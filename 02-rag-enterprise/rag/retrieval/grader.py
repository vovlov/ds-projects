"""
Document Grader для Corrective RAG (CRAG).

CRAG паттерн (Yan et al. 2024, arxiv 2401.15884): после retrieval каждый документ
оценивается по релевантности запросу. Нерелевантные документы отфильтровываются,
что снижает hallucination rate в сравнении с vanilla RAG.

Режимы работы:
- LLM mode (ANTHROPIC_API_KEY задан): Haiku как бинарный классификатор — точнее,
  но стоит дополнительного API-вызова.
- Lexical mode (CI): нормализованный overlap токенов запроса с документом —
  работает без внешних зависимостей, достаточно для фильтрации очевидно нерелевантных.

Источники:
- Yan et al. 2024 (CRAG, arxiv 2401.15884)
- Asai et al. 2023 (Self-RAG, arxiv 2310.11511)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

# Функциональные слова, не несущие смысловой нагрузки для matching
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
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "or",
        "and",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "its",
        "our",
        "their",
        "not",
        "no",
        "nor",
    }
)


def _tokenize(text: str) -> list[str]:
    """Базовая токенизация: lowercase + split по non-word символам."""
    return re.findall(r"\w+", text.lower())


@dataclass
class GradeResult:
    """Результат оценки релевантности одного документа.

    Attributes:
        doc: Исходный документ (dict с полями text, metadata).
        relevance_score: Оценка релевантности [0.0, 1.0].
        is_relevant: True если score >= threshold.
        method: Метод оценки — 'lexical' или 'llm'.
    """

    doc: dict
    relevance_score: float
    is_relevant: bool
    method: str = "lexical"


class DocumentGrader:
    """Оценщик релевантности документов для CRAG.

    Grades retrieved documents against the query to filter irrelevant context
    before passing to the LLM generator. Key CRAG invariant: bad retrieval
    triggers query rewrite, not hallucinated generation.

    Args:
        threshold: Минимальный score для признания документа релевантным.
                   0.2 — консервативный порог для коротких запросов.
        model: LLM-модель для grading в production (Haiku — быстро и дёшево).
    """

    def __init__(
        self,
        threshold: float = 0.2,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        self.threshold = threshold
        self.model = model

    def grade_document(self, query: str, doc: dict) -> GradeResult:
        """Оценить релевантность одного документа запросу.

        В production с ANTHROPIC_API_KEY — LLM-классификатор (Haiku).
        В CI (без ключа) — лексическая метрика без внешних зависимостей.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            return self._llm_grade(query, doc, api_key)
        return self._lexical_grade(query, doc)

    def grade_documents(self, query: str, docs: list[dict]) -> list[GradeResult]:
        """Оценить список документов, вернуть все GradeResult."""
        return [self.grade_document(query, doc) for doc in docs]

    def filter_relevant(self, query: str, docs: list[dict]) -> list[dict]:
        """Вернуть только документы с score >= threshold."""
        grades = self.grade_documents(query, docs)
        return [g.doc for g in grades if g.is_relevant]

    def _lexical_score(self, query: str, doc_text: str) -> float:
        """Recall-ориентированный overlap: доля токенов запроса, найденных в документе.

        Стоп-слова исключены — они шумят без смысловой нагрузки и снижают
        точность discriminative matching для коротких запросов.
        """
        q_tokens = {t for t in _tokenize(query) if t not in _STOP_WORDS}
        d_tokens = set(_tokenize(doc_text))
        if not q_tokens:
            return 0.0
        overlap = q_tokens & d_tokens
        return len(overlap) / len(q_tokens)

    def _lexical_grade(self, query: str, doc: dict) -> GradeResult:
        """Лексическая оценка без LLM — CI-safe."""
        doc_text = doc.get("text", "")
        score = round(self._lexical_score(query, doc_text), 4)
        return GradeResult(
            doc=doc,
            relevance_score=score,
            is_relevant=score >= self.threshold,
            method="lexical",
        )

    def _llm_grade(self, query: str, doc: dict, api_key: str) -> GradeResult:
        """LLM-оценка через Haiku — более точная, с graceful fallback на lexical."""
        try:
            from anthropic import Anthropic

            doc_text = doc.get("text", "")[:600]  # Ограничиваем для экономии токенов
            prompt = (
                f"Query: {query}\n\n"
                f"Document excerpt: {doc_text}\n\n"
                "Is this document relevant to answering the query? "
                "Reply ONLY in this format:\n"
                "VERDICT: RELEVANT|NOT_RELEVANT\n"
                "SCORE: 0.XX"
            )
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=self.model,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            return self._parse_llm_grade(raw, doc)
        except Exception:
            # Network error, rate limit, etc. → fallback без потери функциональности
            return self._lexical_grade(query, doc)

    def _parse_llm_grade(self, raw: str, doc: dict) -> GradeResult:
        """Разобрать структурированный ответ LLM-классификатора.

        Консервативный fallback при ошибке парсинга: score=0.0, not relevant.
        """
        lines: dict[str, str] = {}
        for line in raw.splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                lines[key.strip().upper()] = val.strip()

        try:
            score = float(lines.get("SCORE", "0.0"))
            score = max(0.0, min(1.0, score))
        except ValueError:
            score = 0.0

        verdict = lines.get("VERDICT", "NOT_RELEVANT").upper()
        is_relevant = verdict == "RELEVANT" or score >= self.threshold

        return GradeResult(
            doc=doc,
            relevance_score=round(score, 4),
            is_relevant=is_relevant,
            method="llm",
        )
