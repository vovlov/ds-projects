"""FastAPI + Gradio app for RAG Enterprise."""

from __future__ import annotations

import contextlib
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..cache.semantic_cache import CacheConfig, SemanticCache
from ..generation.chain import generate_answer, generate_answer_with_gate
from ..generation.stream import stream_answer
from ..guardrails.input_guard import InputGuard
from ..guardrails.output_guard import OutputGuard
from ..ingestion.loader import chunk_documents, load_documents
from ..knowledge_graph.graph import KnowledgeGraph
from ..memory.conversation_memory import ConversationMemory, MemoryConfig
from ..retrieval.hybrid import HybridIndex, hybrid_search
from ..retrieval.multi_query import MultiQueryConfig, multi_query_retrieve
from ..retrieval.reranker import RerankConfig, rerank
from ..retrieval.store import get_client, get_or_create_collection, index_chunks

app = FastAPI(title="RAG Enterprise API", version="2.0.0")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"

# Global state
_collection = None
_hybrid_index: HybridIndex | None = None
_indexed_chunks: list[dict] = []
_knowledge_graph: KnowledgeGraph = KnowledgeGraph()
_input_guard: InputGuard = InputGuard()
_output_guard: OutputGuard = OutputGuard()
_semantic_cache: SemanticCache = SemanticCache()
_conversation_memory: ConversationMemory = ConversationMemory()


def _reset_cache() -> None:
    """Сброс кэша для тестовой изоляции."""
    global _semantic_cache
    _semantic_cache = SemanticCache()


def _reset_memory() -> None:
    """Сброс памяти диалога для тестовой изоляции."""
    global _conversation_memory
    _conversation_memory = ConversationMemory()


def _get_collection():
    global _collection, _hybrid_index, _indexed_chunks, _knowledge_graph
    if _collection is None:
        client = get_client()
        _collection = get_or_create_collection(client)

        # Auto-index if collection is empty
        if _collection.count() == 0 and DATA_DIR.exists():
            docs = load_documents(DATA_DIR)
            if docs:
                chunks = chunk_documents(docs)
                index_chunks(chunks, _collection)
                _indexed_chunks = chunks
                _hybrid_index = HybridIndex.build(chunks)
                _knowledge_graph.build_from_chunks(chunks)
    return _collection


class QueryRequest(BaseModel):
    """Запрос к RAG-пайплайну."""

    question: str
    n_results: int = 5
    check_faithfulness: bool = True
    faithfulness_threshold: float = 0.5
    retrieval_method: str = "hybrid"  # "hybrid" | "semantic" | "graph" | "multi_query"
    n_query_variants: int = 3  # для multi_query: число переформулировок
    use_reranking: bool = False  # cross-encoder lexical reranking после первичного поиска
    session_id: str | None = None  # ID сессии для multi-turn диалога (опционально)


class QueryResponse(BaseModel):
    """Ответ RAG-пайплайна с оценкой верности источникам."""

    answer: str
    sources: list[str]
    confidence_score: float
    is_faithful: bool
    faithfulness_method: str
    retrieval_method: str
    query_variants: list[str] | None = None  # заполняется при retrieval_method="multi_query"
    consistency_score: float | None = None  # Jaccard overlap вариантов, 0–1
    reranked: bool = False  # был ли применён cross-encoder reranking
    from_cache: bool = False  # True если ответ взят из semantic cache
    cache_similarity: float | None = None  # TF-IDF cosine similarity к найденной записи кэша
    session_id: str | None = None  # Echo session_id для client-side tracking


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Agentic RAG: ответ на вопрос + faithfulness gate.

    retrieval_method="hybrid" (default) — BM25+vector+RRF, recall@10 ~91%.
    retrieval_method="semantic" — только ChromaDB cosine similarity.

    Возвращает confidence_score — оценку поддержки ответа retrieved документами.
    Низкий score сигнализирует о potential hallucination.

    Запросы кэшируются семантически: повторные/перефразированные вопросы
    отдаются из кэша без retrieval и LLM-вызова (GET /cache/stats для мониторинга).
    """
    collection = _get_collection()
    if collection.count() == 0:
        return QueryResponse(
            answer="No documents indexed. Please add documents to data/documents/ and run /index.",
            sources=[],
            confidence_score=0.0,
            is_faithful=False,
            faithfulness_method="lexical",
            retrieval_method=request.retrieval_method,
            session_id=request.session_id,
        )

    # Query rewriting: follow-up вопросы обогащаем историей диалога
    retrieval_question = request.question
    if request.session_id:
        retrieval_question = _conversation_memory.rewrite_query(
            request.session_id, request.question
        )

    # Semantic cache lookup — пропускаем full RAG pipeline при hit
    cache_result = _semantic_cache.lookup(request.question)
    if cache_result.hit and cache_result.response is not None:
        cached = cache_result.response
        cached_answer = cached["answer"]
        # Также записываем cache-hit в историю диалога
        if request.session_id:
            _conversation_memory.add_turn(
                request.session_id,
                request.question,
                cached_answer,
                cached.get("sources", []),
            )
        return QueryResponse(
            answer=cached_answer,
            sources=cached.get("sources", []),
            confidence_score=cached.get("confidence_score", 1.0),
            is_faithful=cached.get("is_faithful", True),
            faithfulness_method=cached.get("faithfulness_method", "cached"),
            retrieval_method=cached.get("retrieval_method", request.retrieval_method),
            query_variants=cached.get("query_variants"),
            consistency_score=cached.get("consistency_score"),
            reranked=cached.get("reranked", False),
            from_cache=True,
            cache_similarity=round(cache_result.similarity, 4),
            session_id=request.session_id,
        )

    mq_result = None  # populated only for multi_query method

    if request.retrieval_method == "multi_query":
        mq_result = multi_query_retrieve(
            retrieval_question,
            collection,
            _hybrid_index,
            n_results=request.n_results,
            config=MultiQueryConfig(n_variants=request.n_query_variants),
        )
        context = mq_result.chunks
        used_method = "multi_query"
    elif request.retrieval_method == "hybrid":
        context = hybrid_search(
            retrieval_question,
            collection,
            _hybrid_index,
            n_results=request.n_results,
        )
        used_method = "hybrid" if _hybrid_index is not None else "semantic"
    elif request.retrieval_method == "graph":
        context = _knowledge_graph.query_graph(
            retrieval_question,
            _indexed_chunks,
            n_results=request.n_results,
        )
        if not context:
            # Fall back to hybrid when graph finds no entities in query
            context = hybrid_search(
                retrieval_question,
                collection,
                _hybrid_index,
                n_results=request.n_results,
            )
            used_method = "graph_fallback_hybrid"
        else:
            used_method = "graph"
    else:
        from ..retrieval.store import search as semantic_search

        context = semantic_search(retrieval_question, collection, n_results=request.n_results)
        used_method = "semantic"

    # Optional cross-encoder lexical reranking
    did_rerank = False
    if request.use_reranking and context:
        rerank_results = rerank(retrieval_question, context, n_results=request.n_results)
        context = [r.chunk for r in rerank_results]
        did_rerank = True

    if request.check_faithfulness:
        result = generate_answer_with_gate(
            query=request.question,
            context_chunks=context,
            faithfulness_threshold=request.faithfulness_threshold,
        )
    else:
        raw = generate_answer(request.question, context)
        result = {
            **raw,
            "confidence_score": 1.0,
            "is_faithful": True,
            "faithfulness_method": "none",
        }

    response_dict = {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "confidence_score": result["confidence_score"],
        "is_faithful": result["is_faithful"],
        "faithfulness_method": result["faithfulness_method"],
        "retrieval_method": used_method,
        "query_variants": mq_result.query_variants if mq_result else None,
        "consistency_score": mq_result.consistency_score if mq_result else None,
        "reranked": did_rerank,
        "session_id": request.session_id,
    }

    # Кэшируем только faithful ответы — unfaithful могут быть нестабильны
    if result["is_faithful"]:
        _semantic_cache.store(request.question, response_dict)

    # Записываем ход в историю диалога (только оригинальный вопрос, не rewritten)
    if request.session_id:
        _conversation_memory.add_turn(
            request.session_id,
            request.question,
            result["answer"],
            result.get("sources", []),
        )

    return QueryResponse(**response_dict)


@app.post("/query/stream")
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Streaming RAG: токен-за-токеном через Server-Sent Events.

    Возвращает SSE-поток с событиями:
    - ``{"type": "token", "text": "..."}`` — фрагмент ответа
    - ``{"type": "sources", "sources": [...]}`` — список источников
    - ``{"type": "done", "confidence": 0.8, "is_faithful": True, ...}`` — финал
    - ``{"type": "error", "message": "..."}`` — при ошибке

    Используйте ``EventSource`` на клиенте или ``curl -N http://host/query/stream``.

    Без ANTHROPIC_API_KEY возвращает mock-стрим (CI-friendly).
    """
    collection = _get_collection()

    if collection.count() == 0:
        import json

        async def _empty_stream():
            yield f"data: {json.dumps({'type': 'token', 'text': 'No documents indexed.'})}\n\n"
            done = {
                "type": "done",
                "confidence": 0.0,
                "is_faithful": False,
                "faithfulness_method": "none",
                "faithfulness_verdict": "no_documents",
            }
            yield f"data: {json.dumps(done)}\n\n"

        return StreamingResponse(
            _empty_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if request.retrieval_method == "hybrid":
        context = hybrid_search(
            request.question,
            collection,
            _hybrid_index,
            n_results=request.n_results,
        )
    else:
        from ..retrieval.store import search as semantic_search

        context = semantic_search(request.question, collection, n_results=request.n_results)

    return StreamingResponse(
        stream_answer(
            query=request.question,
            context_chunks=context,
            faithfulness_threshold=request.faithfulness_threshold,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class IndexRequest(BaseModel):
    """Параметры переиндексации документов."""

    chunking_strategy: str = "fixed"  # "fixed" | "semantic" | "paragraph"


@app.post("/index")
def index_documents(request: IndexRequest = IndexRequest()):
    """Re-index all documents from data directory.

    chunking_strategy:
    - "fixed"     — RecursiveCharacterTextSplitter 512 chars (default)
    - "semantic"  — TF-IDF cosine boundary detection
    - "paragraph" — split on double newlines

    Очищает semantic cache — после переиндексации старые ответы могут быть устаревшими.
    """
    global _collection, _hybrid_index, _indexed_chunks

    client = get_client()
    with contextlib.suppress(Exception):
        client.delete_collection("documents")

    _collection = get_or_create_collection(client)
    _hybrid_index = None
    _indexed_chunks = []

    if not DATA_DIR.exists():
        return {"error": f"Data directory not found: {DATA_DIR}"}

    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs, chunking_strategy=request.chunking_strategy)
    count = index_chunks(chunks, _collection)

    _indexed_chunks = chunks
    _hybrid_index = HybridIndex.build(chunks)
    _knowledge_graph.build_from_chunks(chunks)

    # Инвалидируем кэш: документы изменились → старые ответы устарели
    evicted = _semantic_cache.clear()

    return {
        "indexed_chunks": count,
        "documents": len(docs),
        "chunking_strategy": request.chunking_strategy,
        "hybrid_index": _hybrid_index._bm25 is not None,
        "knowledge_graph_nodes": _knowledge_graph.stats().n_nodes,
        "cache_evicted": evicted,
    }


class ChunkPreviewRequest(BaseModel):
    """Запрос предварительного просмотра нарезки текста."""

    text: str
    chunking_strategy: str = "semantic"  # "fixed" | "semantic" | "paragraph"
    chunk_size: int = 512
    chunk_overlap: int = 64


class ChunkPreviewResponse(BaseModel):
    """Предварительный просмотр результатов нарезки."""

    chunks: list[str]
    n_chunks: int
    avg_chunk_chars: float
    chunking_strategy: str
    semantic_available: bool


@app.post("/chunk/preview", response_model=ChunkPreviewResponse)
def chunk_preview(request: ChunkPreviewRequest) -> ChunkPreviewResponse:
    """Preview how a text would be split with the chosen chunking strategy.

    Позволяет сравнить стратегии нарезки без переиндексации: передайте
    текстовый фрагмент и получите список чанков с метриками.
    """
    from ..chunking.semantic import is_available as semantic_available_fn

    doc = {"text": request.text, "metadata": {"source": "preview"}}
    chunks = chunk_documents(
        [doc],
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        chunking_strategy=request.chunking_strategy,
    )
    texts = [c["text"] for c in chunks]
    avg = sum(len(t) for t in texts) / max(len(texts), 1)

    return ChunkPreviewResponse(
        chunks=texts,
        n_chunks=len(texts),
        avg_chunk_chars=round(avg, 1),
        chunking_strategy=request.chunking_strategy,
        semantic_available=semantic_available_fn(),
    )


@app.post("/graph/build")
def graph_build():
    """Build (or rebuild) the Knowledge Graph from currently indexed chunks.

    Automatically called on /index. Use this endpoint to rebuild after
    manual updates to indexed_chunks without re-indexing all documents.
    """
    global _knowledge_graph
    if not _indexed_chunks:
        return {"status": "no_chunks", "n_nodes": 0, "n_edges": 0}

    stats = _knowledge_graph.build_from_chunks(_indexed_chunks)
    return {
        "status": "built",
        **stats.to_dict(),
    }


@app.get("/graph/stats")
def graph_stats():
    """Knowledge Graph statistics: node/edge counts and top entities."""
    stats = _knowledge_graph.stats()
    return {"is_built": _knowledge_graph.is_built, **stats.to_dict()}


@app.get("/graph/entity/{entity_key}")
def graph_entity(entity_key: str):
    """Return 1-hop subgraph centred on an entity (D3.js-compatible format).

    entity_key should be lower-cased entity text (e.g. "openai", "machine learning").
    """
    subgraph = _knowledge_graph.get_entity_subgraph(entity_key.lower())
    if not subgraph["found"] and not subgraph["nodes"]:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_key}' not found in graph")
    return subgraph


# ---------------------------------------------------------------------------
# Reranking endpoint
# ---------------------------------------------------------------------------


class RerankRequest(BaseModel):
    """Запрос на cross-encoder reranking списка чанков по запросу."""

    query: str
    candidates: list[dict]  # список чанков {"text": ..., "metadata": {...}}
    n_results: int = 5
    coverage_weight: float = 0.5
    tf_weight: float = 0.35
    position_weight: float = 0.15


class RerankResponseItem(BaseModel):
    """Один переранжированный чанк с объяснением скора."""

    text: str
    rerank_score: float
    original_rank: int
    rerank_rank: int
    coverage: float
    tf_score: float
    position_score: float
    metadata: dict = {}


class RerankResponse(BaseModel):
    """Результат cross-encoder reranking."""

    results: list[RerankResponseItem]
    n_candidates: int
    n_results: int
    query: str


@app.post("/rerank", response_model=RerankResponse)
def rerank_chunks(request: RerankRequest) -> RerankResponse:
    """Переранжировать чанки по запросу через лексический cross-encoder.

    Полезно когда bi-encoder (ChromaDB) возвращает семантически близкие,
    но фактически нерелевантные чанки. Cross-encoder даёт joint scoring
    (query + passage вместе) — лучше precision без GPU/API overhead.

    Алгоритм:
    - Coverage: доля query-термов найденных в passage
    - TF-weighted score с IDF-прокси (редкие термы важнее)
    - Position bonus: термы в первой 25% пассажа (summary-эффект)

    Источники: Nogueira & Cho 2019 (arxiv:1901.04085), Khattab & Zaharia 2020 ColBERT.
    """
    if not request.candidates:
        raise HTTPException(status_code=422, detail="candidates list is empty")

    config = RerankConfig(
        coverage_weight=request.coverage_weight,
        tf_weight=request.tf_weight,
        position_weight=request.position_weight,
    )

    results = rerank(request.query, request.candidates, n_results=request.n_results, config=config)

    items = [
        RerankResponseItem(
            text=r.chunk.get("text", ""),
            rerank_score=r.rerank_score,
            original_rank=r.original_rank,
            rerank_rank=r.rerank_rank,
            coverage=r.coverage,
            tf_score=r.tf_score,
            position_score=r.position_score,
            metadata=r.chunk.get("metadata", {}),
        )
        for r in results
    ]

    return RerankResponse(
        results=items,
        n_candidates=len(request.candidates),
        n_results=len(items),
        query=request.query,
    )


# ---------------------------------------------------------------------------
# Guardrails endpoints
# ---------------------------------------------------------------------------


class InputGuardRequest(BaseModel):
    """Запрос на проверку входного текста через Input Guardrail."""

    query: str
    domain_keywords: list[str] = []
    block_pii: bool = False
    block_off_topic: bool = False
    max_query_length: int = 2000


class InputGuardResponse(BaseModel):
    """Результат проверки входного запроса."""

    is_safe: bool
    threats: list[str]
    sanitized_query: str
    risk_score: float
    details: dict[str, str]


class OutputGuardRequest(BaseModel):
    """Запрос на проверку выходного ответа через Output Guardrail."""

    answer: str
    sources: list[str] = []
    mask_pii: bool = True
    min_answer_length: int = 10


class OutputGuardResponse(BaseModel):
    """Результат проверки выходного ответа."""

    is_safe: bool
    threats: list[str]
    filtered_answer: str
    risk_score: float
    pii_types_found: list[str]


@app.post("/guardrails/check/input", response_model=InputGuardResponse)
def guardrails_check_input(request: InputGuardRequest) -> InputGuardResponse:
    """Проверить запрос через Input Guardrail перед отправкой в RAG.

    Детектирует prompt injection (OWASP LLM01), PII в запросе (GDPR Art.5),
    off-domain запросы. Возвращает sanitized_query с замаскированными PII.

    Prompt injection всегда блокирует (is_safe=False); PII и off-topic — только
    при соответствующих флагах.
    """
    guard = InputGuard(
        max_query_length=request.max_query_length,
        domain_keywords=request.domain_keywords,
        block_pii=request.block_pii,
        block_off_topic=request.block_off_topic,
    )
    result = guard.check(request.query)
    return InputGuardResponse(
        is_safe=result.is_safe,
        threats=[str(t) for t in result.threats],
        sanitized_query=result.sanitized_query,
        risk_score=result.risk_score,
        details=result.details,
    )


@app.post("/guardrails/check/output", response_model=OutputGuardResponse)
def guardrails_check_output(request: OutputGuardRequest) -> OutputGuardResponse:
    """Проверить ответ через Output Guardrail перед отдачей пользователю.

    Маскирует PII в ответе (GDPR Art.5), фильтрует вредоносный контент
    (OWASP LLM02), предупреждает об ответах без источников (риск галлюцинации).

    Вредоносный контент блокирует (is_safe=False, filtered_answer = заглушка).
    PII маскируется в filtered_answer без блокировки.
    """
    guard = OutputGuard(
        mask_pii=request.mask_pii,
        min_answer_length=request.min_answer_length,
    )
    result = guard.check(request.answer, sources=request.sources)
    return OutputGuardResponse(
        is_safe=result.is_safe,
        threats=[str(t) for t in result.threats],
        filtered_answer=result.filtered_answer,
        risk_score=result.risk_score,
        pii_types_found=result.pii_types_found,
    )


@app.get("/guardrails/config")
def guardrails_config() -> dict:
    """Текущая конфигурация глобальных guardrails."""
    return {
        "input_guard": {
            "max_query_length": _input_guard.max_query_length,
            "domain_keywords_count": len(_input_guard.domain_keywords),
            "block_pii": _input_guard.block_pii,
            "block_off_topic": _input_guard.block_off_topic,
            "injection_patterns_count": len(_input_guard._injection_re),
        },
        "output_guard": {
            "mask_pii": _output_guard.mask_pii,
            "min_answer_length": _output_guard.min_answer_length,
            "harmful_patterns_count": len(_output_guard._harmful_re),
        },
        "compliance": ["OWASP LLM Top 10 2025", "GDPR Article 5", "EU AI Act Article 13"],
    }


# ---------------------------------------------------------------------------
# Semantic Cache endpoints
# ---------------------------------------------------------------------------


class CacheConfigRequest(BaseModel):
    """Параметры конфигурации semantic cache."""

    similarity_threshold: float = 0.85
    max_entries: int = 100
    ttl_seconds: float = 3600.0


@app.get("/cache/stats")
def cache_stats() -> dict:
    """Статистика semantic cache для мониторинга.

    Возвращает hit_rate, число записей, eviction/expiration счётчики.
    Hit rate > 0.3 означает хорошее переиспользование при FAQ-нагрузке.
    """
    stats = _semantic_cache.get_stats()
    return {
        "total_queries": stats.total_queries,
        "hits": stats.hits,
        "misses": stats.misses,
        "hit_rate": round(stats.hit_rate, 4),
        "n_entries": stats.n_entries,
        "evictions": stats.evictions,
        "expirations": stats.expirations,
        "config": {
            "similarity_threshold": _semantic_cache.config.similarity_threshold,
            "max_entries": _semantic_cache.config.max_entries,
            "ttl_seconds": _semantic_cache.config.ttl_seconds,
        },
    }


@app.post("/cache/clear")
def cache_clear() -> dict:
    """Принудительно очистить semantic cache.

    Используйте после обновления документов или при проблемах с качеством ответов.
    Автоматически вызывается при POST /index.
    """
    evicted = _semantic_cache.clear()
    return {"cleared": evicted, "status": "ok"}


@app.post("/cache/configure")
def cache_configure(request: CacheConfigRequest) -> dict:
    """Изменить конфигурацию semantic cache без перезапуска.

    Полезно для A/B-тестирования оптимального similarity_threshold:
    0.95 = точное совпадение фраз, 0.75 = широкое семантическое matching.
    Текущие записи сохраняются, новые параметры применяются к следующим lookups.
    """
    global _semantic_cache
    old_entries = dict(_semantic_cache._entries)
    old_stats = {
        "total": _semantic_cache._total,
        "hits": _semantic_cache._hits,
        "misses": _semantic_cache._misses,
        "evictions": _semantic_cache._evictions,
        "expirations": _semantic_cache._expirations,
    }
    new_config = CacheConfig(
        similarity_threshold=request.similarity_threshold,
        max_entries=request.max_entries,
        ttl_seconds=request.ttl_seconds,
    )
    _semantic_cache = SemanticCache(new_config)
    # Перенести существующие записи в новый кэш (с обрезкой до нового max_entries)
    for key, entry in old_entries.items():
        _semantic_cache._entries[key] = entry
    while len(_semantic_cache._entries) > new_config.max_entries:
        oldest, _ = next(iter(_semantic_cache._entries.items()))
        del _semantic_cache._entries[oldest]
    # Восстановить счётчики для непрерывной статистики
    _semantic_cache._total = old_stats["total"]
    _semantic_cache._hits = old_stats["hits"]
    _semantic_cache._misses = old_stats["misses"]
    _semantic_cache._evictions = old_stats["evictions"]
    _semantic_cache._expirations = old_stats["expirations"]
    return {
        "status": "configured",
        "similarity_threshold": new_config.similarity_threshold,
        "max_entries": new_config.max_entries,
        "ttl_seconds": new_config.ttl_seconds,
        "entries_preserved": len(_semantic_cache._entries),
    }


# ---------------------------------------------------------------------------
# Conversational Memory endpoints
# ---------------------------------------------------------------------------


class MemoryConfigRequest(BaseModel):
    """Параметры конфигурации памяти диалога."""

    max_turns: int = 10
    ttl_seconds: float = 3600.0
    context_turns: int = 3


class ConversationTurnResponse(BaseModel):
    """Один ход диалога для API-ответа."""

    question: str
    answer: str
    sources: list[str]
    timestamp: str


class MemoryHistoryResponse(BaseModel):
    """История диалога для API-ответа."""

    session_id: str
    turns: list[ConversationTurnResponse]
    n_turns: int


@app.post("/memory/session")
def memory_create_session(request: MemoryConfigRequest = MemoryConfigRequest()) -> dict:
    """Создать новую сессию диалога.

    Возвращает session_id, который нужно передавать в POST /query
    для сохранения контекста между вопросами.

    Настраиваемые параметры:
    - max_turns: максимум ходов в скользящем окне (default 10)
    - ttl_seconds: время жизни сессии без активности (default 1 час)
    - context_turns: число предыдущих ходов для rewriting follow-up вопросов
    """
    global _conversation_memory
    config = MemoryConfig(
        max_turns=request.max_turns,
        ttl_seconds=request.ttl_seconds,
        context_turns=request.context_turns,
    )
    # Создаём сессию с переданными параметрами
    _conversation_memory_with_config = ConversationMemory(config)
    session_id = _conversation_memory_with_config.create_session()
    # Перенести сессию в глобальный менеджер
    _conversation_memory._sessions[session_id] = _conversation_memory_with_config._sessions[
        session_id
    ]
    return {
        "session_id": session_id,
        "max_turns": config.max_turns,
        "ttl_seconds": config.ttl_seconds,
        "context_turns": config.context_turns,
    }


@app.get("/memory/history/{session_id}", response_model=MemoryHistoryResponse)
def memory_history(session_id: str) -> MemoryHistoryResponse:
    """Получить историю диалога для сессии.

    Возвращает все ходы в хронологическом порядке (oldest first).
    Полезно для отображения диалога в UI или аудита.
    """
    turns = _conversation_memory.get_history(session_id)
    return MemoryHistoryResponse(
        session_id=session_id,
        turns=[
            ConversationTurnResponse(
                question=t.question,
                answer=t.answer,
                sources=t.sources,
                timestamp=t.timestamp,
            )
            for t in turns
        ],
        n_turns=len(turns),
    )


@app.post("/memory/reset/{session_id}")
def memory_reset(session_id: str) -> dict:
    """Сбросить историю диалога для сессии.

    Используйте для начала нового диалога с тем же session_id
    без создания новой сессии. После сброса следующий вопрос
    обрабатывается без истории контекста.
    """
    cleared = _conversation_memory.reset_session(session_id)
    return {"session_id": session_id, "cleared": cleared}


@app.get("/memory/sessions")
def memory_sessions() -> dict:
    """Список активных сессий диалога.

    Возвращает только не истёкшие сессии (TTL не вышел).
    Полезно для мониторинга нагрузки и аудита.
    """
    active = _conversation_memory.list_sessions()
    return {"active_sessions": active, "count": len(active)}


def ask(question: str, n_results: int = 5, use_hybrid: bool = True) -> str:
    """RAG pipeline: retrieve → generate → faithfulness gate."""
    if not question.strip():
        return "Please enter a question."

    collection = _get_collection()
    if collection.count() == 0:
        return "No documents indexed. Please add documents to data/documents/ and run /index."

    if use_hybrid:
        context = hybrid_search(question, collection, _hybrid_index, n_results=n_results)
        retrieval_label = "Hybrid (BM25+Vector+RRF)" if _hybrid_index is not None else "Semantic"
    else:
        from ..retrieval.store import search as semantic_search

        context = semantic_search(question, collection, n_results=n_results)
        retrieval_label = "Semantic only"

    result = generate_answer_with_gate(question, context)

    sources = ", ".join(result["sources"]) if result["sources"] else "N/A"
    confidence = result["confidence_score"]
    verdict = result["faithfulness_verdict"]
    method = result["faithfulness_method"]

    confidence_badge = "✅" if result["is_faithful"] else "⚠️"
    return (
        f"{result['answer']}\n\n"
        f"---\n"
        f"**Sources:** {sources}\n\n"
        f"**Retrieval:** {retrieval_label}\n\n"
        f"**Confidence:** {confidence_badge} {confidence:.2f} ({verdict}, method: {method})"
    )


# Gradio interface
demo = gr.Interface(
    fn=ask,
    inputs=[
        gr.Textbox(label="Question", placeholder="Ask a question about the documents..."),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of context chunks"),
        gr.Checkbox(value=True, label="Hybrid search (BM25+Vector+RRF)"),
    ],
    outputs=gr.Markdown(label="Answer"),
    title="RAG Enterprise — Document Q&A",
    description=(
        "Ask questions about indexed documents. "
        "Hybrid search combines BM25 keyword matching with semantic vector search via RRF."
    ),
)

app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
