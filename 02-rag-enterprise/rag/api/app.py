"""FastAPI + Gradio app for RAG Enterprise."""

from __future__ import annotations

import contextlib
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..generation.chain import generate_answer, generate_answer_with_gate
from ..generation.stream import stream_answer
from ..ingestion.loader import chunk_documents, load_documents
from ..retrieval.hybrid import HybridIndex, hybrid_search
from ..retrieval.store import get_client, get_or_create_collection, index_chunks

app = FastAPI(title="RAG Enterprise API", version="2.0.0")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"

# Global state
_collection = None
_hybrid_index: HybridIndex | None = None
_indexed_chunks: list[dict] = []


def _get_collection():
    global _collection, _hybrid_index, _indexed_chunks
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
    return _collection


class QueryRequest(BaseModel):
    """Запрос к RAG-пайплайну."""

    question: str
    n_results: int = 5
    check_faithfulness: bool = True
    faithfulness_threshold: float = 0.5
    retrieval_method: str = "hybrid"  # "hybrid" | "semantic"


class QueryResponse(BaseModel):
    """Ответ RAG-пайплайна с оценкой верности источникам."""

    answer: str
    sources: list[str]
    confidence_score: float
    is_faithful: bool
    faithfulness_method: str
    retrieval_method: str


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
        )

    if request.retrieval_method == "hybrid":
        context = hybrid_search(
            request.question,
            collection,
            _hybrid_index,
            n_results=request.n_results,
        )
        used_method = "hybrid" if _hybrid_index is not None else "semantic"
    else:
        from ..retrieval.store import search as semantic_search

        context = semantic_search(request.question, collection, n_results=request.n_results)
        used_method = "semantic"

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

    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence_score=result["confidence_score"],
        is_faithful=result["is_faithful"],
        faithfulness_method=result["faithfulness_method"],
        retrieval_method=used_method,
    )


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


@app.post("/index")
def index_documents():
    """Re-index all documents from data directory."""
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
    chunks = chunk_documents(docs)
    count = index_chunks(chunks, _collection)

    _indexed_chunks = chunks
    _hybrid_index = HybridIndex.build(chunks)

    return {
        "indexed_chunks": count,
        "documents": len(docs),
        "hybrid_index": _hybrid_index._bm25 is not None,
    }


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
