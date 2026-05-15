"""
Corrective RAG (CRAG) — исправляющий retrieval с grading и query rewriting.

Алгоритм CRAG (Yan et al. 2024, arxiv 2401.15884):
1. Retrieve: получить top-k документов через hybrid/semantic search.
2. Grade: оценить релевантность каждого документа через DocumentGrader.
3. Act:
   - "use_all": все документы релевантны → прямой путь в generation.
   - "filter_relevant": часть нерелевантна → отфильтровать, использовать subset.
   - "rewrite_and_retry": нет релевантных → переписать запрос → retry retrieval.

Query rewriting (без LLM):
Извлекаем ключевые термины (без стоп-слов), дедуплицируем, формируем keyword query.
Это эффективно для случаев когда исходный запрос слишком многословный или содержит
редкие термины, которых нет в корпусе в точном виде.

Источники:
- Yan et al. 2024 (CRAG, arxiv 2401.15884)
- LangGraph CRAG implementation 2025
- Self-RAG (Asai et al. 2023, arxiv 2310.11511)
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from .grader import _STOP_WORDS, DocumentGrader, GradeResult, _tokenize
from .hybrid import HybridIndex, hybrid_search
from .store import search as semantic_search


@dataclass
class CorrectiveResult:
    """Результат corrective retrieval с аудит-следом принятых решений.

    Attributes:
        docs: Финальные документы для generation (после grading/rewrite).
        grades: Оценки документов первого retrieval (для transparency/debugging).
        action: Действие CRAG: "use_all" | "filter_relevant" | "rewrite_and_retry".
        query_rewritten: Переписанный запрос (только если action == "rewrite_and_retry").
        n_relevant: Число релевантных документов из первого retrieval.
        n_total: Общее число документов из первого retrieval.
    """

    docs: list[dict]
    grades: list[GradeResult]
    action: str
    query_rewritten: str | None = None
    n_relevant: int = 0
    n_total: int = 0


class CorrectiveRetriever:
    """CRAG orchestrator: grade → filter/rewrite → (re-)retrieve.

    Встраивается поверх существующего hybrid_search без изменения интерфейса
    ChromaDB или HybridIndex. Singleton-friendly через фабрику _get_corrective_retriever().

    Args:
        grader: DocumentGrader для оценки релевантности. Создаётся по умолчанию.
    """

    def __init__(self, grader: DocumentGrader | None = None) -> None:
        self.grader = grader or DocumentGrader()

    def retrieve_and_grade(
        self,
        query: str,
        collection: chromadb.Collection,
        hybrid_index: HybridIndex | None = None,
        n_results: int = 5,
    ) -> CorrectiveResult:
        """Главный метод CRAG: retrieve → grade → act.

        Args:
            query: Исходный запрос пользователя.
            collection: ChromaDB коллекция с проиндексированными документами.
            hybrid_index: BM25-индекс (None → pure semantic search).
            n_results: Желаемое число финальных документов.

        Returns:
            CorrectiveResult с финальными документами и аудит-следом.
        """
        docs = self._retrieve(query, collection, hybrid_index, n_results)

        if not docs:
            return CorrectiveResult(
                docs=[],
                grades=[],
                action="use_all",
                n_relevant=0,
                n_total=0,
            )

        grades = self.grader.grade_documents(query, docs)
        relevant_grades = [g for g in grades if g.is_relevant]
        n_relevant = len(relevant_grades)
        n_total = len(grades)

        # Все документы прошли grading → не трогаем порядок и состав
        if n_relevant == n_total:
            return CorrectiveResult(
                docs=docs,
                grades=grades,
                action="use_all",
                n_relevant=n_relevant,
                n_total=n_total,
            )

        # Часть документов релевантна → отфильтровать нерелевантные
        if n_relevant > 0:
            filtered = [g.doc for g in grades if g.is_relevant]
            return CorrectiveResult(
                docs=filtered,
                grades=grades,
                action="filter_relevant",
                n_relevant=n_relevant,
                n_total=n_total,
            )

        # Ни один документ не прошёл grading → rewrite + retry
        rewritten = self._rewrite_query(query)
        retry_docs = self._retrieve(rewritten, collection, hybrid_index, n_results)
        retry_grades = self.grader.grade_documents(rewritten, retry_docs)

        # Предпочитаем релевантные из retry, иначе возвращаем лучшее из retry (лучше, чем ничего)
        retry_relevant = [g.doc for g in retry_grades if g.is_relevant]
        final_docs = retry_relevant if retry_relevant else retry_docs[:n_results]

        return CorrectiveResult(
            docs=final_docs,
            grades=grades,  # Grades исходного запроса — для прозрачности
            action="rewrite_and_retry",
            query_rewritten=rewritten,
            n_relevant=n_relevant,
            n_total=n_total,
        )

    def _retrieve(
        self,
        query: str,
        collection: chromadb.Collection,
        hybrid_index: HybridIndex | None,
        n_results: int,
    ) -> list[dict]:
        """Retrieval через hybrid search или pure semantic в зависимости от индекса."""
        if hybrid_index is not None:
            return hybrid_search(query, collection, hybrid_index, n_results=n_results)
        return semantic_search(query, collection, n_results=n_results)

    def _rewrite_query(self, query: str) -> str:
        """Переписать запрос как keyword query (lexical fallback без LLM).

        Удаляем стоп-слова, вопросительные слова (what/how/why) уже в стоп-словах,
        оставляем только содержательные термины длиной > 2 символов.
        Дедуплицируем с сохранением порядка появления.
        """
        tokens = _tokenize(query)
        keywords = [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]

        if not keywords:
            return query

        # Дедупликация с сохранением порядка
        seen: set[str] = set()
        unique: list[str] = []
        for k in keywords:
            if k not in seen:
                seen.add(k)
                unique.append(k)

        return " ".join(unique)
