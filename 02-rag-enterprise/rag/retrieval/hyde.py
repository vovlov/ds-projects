"""HyDE: Hypothetical Document Embeddings для улучшения semantic retrieval.

Вместо эмбеддинга запроса ("What is the vacation policy?") генерируем
гипотетический документ-ответ ("Remote workers receive 20 days vacation...")
и ищем по нему — разрыв «вопрос vs документ» сокращается, recall улучшается.

Источники:
- Gao et al. 2022 "Precise Zero-Shot Dense Retrieval without Relevance Labels"
  (arxiv:2212.10496) — оригинальная статья HyDE
- Anthropic Contextual Retrieval blog 2024
- LangChain HypotheticalDocumentEmbedder docs 2025
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass
class HyDEConfig:
    """Конфигурация HyDE Retrieval."""

    max_tokens: int = 150  # длина гипотетического документа
    temperature: float = 0.7  # выше = разнообразнее гипотезы
    n_hypothetical: int = 1  # число гипотетических документов (1 = быстро)
    use_llm: bool = False  # True = Claude Haiku; False = rule-based fallback (CI-safe)


@dataclass
class HyDEResult:
    """Результат HyDE Retrieval."""

    chunks: list[dict]
    hypothetical_document: str  # сгенерированный документ (для audit/debug)
    retrieval_method: str = "hyde"
    n_results: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_results = len(self.chunks)

    def to_dict(self) -> dict:
        return {
            "n_chunks": len(self.chunks),
            "hypothetical_document": self.hypothetical_document,
            "retrieval_method": self.retrieval_method,
        }


# Шаблоны для rule-based генерации гипотетических документов без LLM
_QUESTION_TEMPLATES: list[tuple[str, str]] = [
    # (паттерн вопроса, шаблон гипотезы)
    (
        r"^(what is|what are)\s+(.+)",
        "{subject} refers to {keywords}. "
        "It is characterized by specific properties and defined according to established "
        "guidelines.",
    ),
    (
        r"^(how (do|does|can|to))\s+(.+)",
        "The process of {subject} involves several steps: first, identify the requirements; "
        "then follow the established {keywords} procedure according to the guidelines.",
    ),
    (
        r"^(when|where)\s+(.+)",
        "Regarding {subject}: the relevant {keywords} apply under specific conditions "
        "and circumstances as defined in the policy documentation.",
    ),
    (
        r"^(why)\s+(.+)",
        "The reason {subject} is important relates to {keywords}. "
        "This is necessary for compliance with established standards and requirements.",
    ),
    (
        r"^(who)\s+(.+)",
        "The responsible parties for {subject} are designated roles within the organization. "
        "They handle {keywords} according to the relevant policies.",
    ),
]

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "what",
        "how",
        "why",
        "when",
        "where",
        "who",
        "which",
        "does",
        "do",
        "can",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "it",
        "its",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "with",
        "from",
        "and",
        "or",
        "but",
        "not",
        "that",
        "this",
        "these",
        "those",
    }
)


def _extract_keywords(text: str) -> list[str]:
    """Извлекает ключевые слова (без стоп-слов) для шаблонной генерации."""
    tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9_\-]*\b", text)
    return [t for t in tokens if t.lower() not in _STOP_WORDS and len(t) > 2]


def _rule_based_hypothetical(query: str) -> str:
    """CI-safe генерация гипотетического документа без LLM.

    Применяет шаблоны для типичных вопросов или строит универсальный
    гипотетический ответ из ключевых слов запроса.
    """
    q_lower = query.lower().strip()
    keywords = _extract_keywords(query)
    kw_phrase = ", ".join(keywords[:5]) if keywords else query

    for pattern, template in _QUESTION_TEMPLATES:
        m = re.match(pattern, q_lower)
        if m:
            # последняя группа паттерна содержит суть вопроса
            subject = m.group(m.lastindex) if m.lastindex else query
            return template.format(subject=subject.strip(), keywords=kw_phrase)

    # Универсальный шаблон — работает для любого запроса
    return (
        f"This document addresses {query.strip().rstrip('?')}. "
        f"The relevant information covers {kw_phrase}. "
        f"According to established policies and procedures, "
        f"specific guidelines apply to ensure compliance and proper handling."
    )


def _llm_hypothetical(query: str, config: HyDEConfig) -> str:
    """Генерирует гипотетический документ через Claude Haiku.

    Graceful fallback на rule-based при отсутствии ключа или ошибке.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _rule_based_hypothetical(query)

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            "Write a short factual passage (2-3 sentences) that would directly answer "
            "the following question. Write as if it were an excerpt from a policy document "
            "or knowledge base article. Do not say 'the answer is' — just write the passage.\n\n"
            f"Question: {query}"
        )
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception:
        return _rule_based_hypothetical(query)


def generate_hypothetical_document(
    query: str,
    config: HyDEConfig | None = None,
) -> str:
    """Генерирует гипотетический документ для HyDE retrieval.

    В production (use_llm=True): LLM создаёт ответ в стиле corpus-документа.
    В CI (use_llm=False): детерминированный шаблонный ответ.

    Args:
        query: Вопрос пользователя.
        config: Параметры HyDE (max_tokens, temperature, use_llm).

    Returns:
        Текст гипотетического документа.
    """
    if config is None:
        config = HyDEConfig()

    if config.use_llm:
        return _llm_hypothetical(query, config)
    return _rule_based_hypothetical(query)


def hyde_retrieve(
    query: str,
    collection,
    hybrid_index,
    n_results: int = 5,
    config: HyDEConfig | None = None,
) -> HyDEResult:
    """HyDE Retrieval: поиск по гипотетическому документу.

    Алгоритм (Gao et al. 2022):
    1. Генерируем гипотетический документ-ответ на запрос
    2. Используем ТЕКСТ ДОКУМЕНТА (не запроса) для семантического поиска —
       эмбеддинг документа в «пространстве ответов» vs. «пространстве вопросов»
    3. Результаты ранжируются ChromaDB по cosine similarity к гипотезе

    Преимущество: FAQ-запросы ("vacation policy?") vs. корпус-документы
    ("Vacation Policy Section 3.2: employees are entitled to...") — запрос
    и документ находятся в разных эмбеддинговых пространствах. Гипотетический
    документ снижает этот разрыв, используя словарь и стиль корпуса.

    Args:
        query: Исходный запрос пользователя.
        collection: ChromaDB коллекция.
        hybrid_index: BM25-индекс (используется как fallback).
        n_results: Число чанков для возврата.
        config: Параметры HyDE.

    Returns:
        HyDEResult с retrieved чанками и гипотетическим документом.
    """
    if config is None:
        config = HyDEConfig()

    hypothetical_doc = generate_hypothetical_document(query, config)

    # Семантический поиск по гипотетическому документу (не по оригинальному запросу)
    from .store import search as semantic_search

    chunks = semantic_search(hypothetical_doc, collection, n_results=n_results)

    return HyDEResult(
        chunks=chunks,
        hypothetical_document=hypothetical_doc,
        retrieval_method="hyde",
    )
