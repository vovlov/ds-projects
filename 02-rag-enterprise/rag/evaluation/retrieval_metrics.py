"""
Стандартные метрики качества поиска (IR metrics) для оффлайн-оценки RAG-пайплайнов.

Information Retrieval metrics for offline RAG retrieval evaluation.
Позволяет измерить качество поиска независимо от качества генерации.

Метрики:
- Precision@K  — точность: доля релевантных среди первых K результатов
- Recall@K     — полнота: доля найденных релевантных документов из всех существующих
- MRR          — Mean Reciprocal Rank: где впервые встречается релевантный результат
- MAP          — Mean Average Precision: площадь под precision-recall кривой
- NDCG@K       — Normalized DCG: учитывает позицию релевантных (топ важнее дна)

Источники:
    Manning et al. 2008 "Introduction to Information Retrieval" §8 (P@K, MAP, MRR)
    Järvelin & Kekäläinen 2002 ACM TOIS 20(4) (NDCG)
    Croft et al. 2010 "Search Engines: Information Retrieval in Practice" §6
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class QueryMetrics:
    """Метрики качества поиска для одного запроса."""

    query: str
    precision_at_k: dict[int, float]  # {1: 0.8, 3: 0.6, 5: 0.5, 10: 0.4}
    recall_at_k: dict[int, float]  # {1: 0.2, 3: 0.4, 5: 0.6, 10: 0.8}
    mrr: float  # Reciprocal Rank для этого запроса (MRR агрегируется по всем запросам)
    average_precision: float  # AP для этого запроса (MAP = mean по всем запросам)
    ndcg_at_k: dict[int, float]  # {1: 0.9, 3: 0.75, 5: 0.68, 10: 0.71}
    n_relevant: int  # Число известных релевантных документов (для recall)
    n_retrieved: int  # Число фактически retrieved документов

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": round(self.mrr, 4),
            "average_precision": round(self.average_precision, 4),
            "ndcg_at_k": self.ndcg_at_k,
            "n_relevant": self.n_relevant,
            "n_retrieved": self.n_retrieved,
        }


@dataclass
class RetrievalEvalReport:
    """Агрегированный отчёт по качеству retrieval-системы.

    Агрегирует QueryMetrics по всему набору golden queries.
    MRR и MAP — классические IR метрики для сравнения систем поиска.
    """

    n_queries: int
    k_values: list[int]
    mean_precision_at_k: dict[int, float]  # macro-average P@K по всем запросам
    mean_recall_at_k: dict[int, float]  # macro-average R@K по всем запросам
    mrr: float  # Mean Reciprocal Rank по всем запросам
    map_score: float  # Mean Average Precision (поле map — зарезервировано builtin)
    mean_ndcg_at_k: dict[int, float]  # macro-average NDCG@K
    per_query: list[QueryMetrics] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "k_values": self.k_values,
            "mean_precision_at_k": {
                str(k): round(v, 4) for k, v in self.mean_precision_at_k.items()
            },
            "mean_recall_at_k": {str(k): round(v, 4) for k, v in self.mean_recall_at_k.items()},
            "mrr": round(self.mrr, 4),
            "map": round(self.map_score, 4),
            "mean_ndcg_at_k": {str(k): round(v, 4) for k, v in self.mean_ndcg_at_k.items()},
            "per_query": [qm.to_dict() for qm in self.per_query],
        }


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@K — точность первых K результатов.

    P@K = |{relevant} ∩ {retrieved[:K]}| / K

    Args:
        retrieved: Список retrieved doc-IDs (упорядочен по убыванию релевантности).
        relevant: Множество известных релевантных doc-IDs для данного запроса.
        k: Горизонт отсечения.

    Returns:
        P@K ∈ [0, 1].
    """
    if k <= 0 or not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K — полнота: какую долю релевантных мы нашли в первых K.

    R@K = |{relevant} ∩ {retrieved[:K]}| / |relevant|

    Args:
        retrieved: Список retrieved doc-IDs.
        relevant: Множество известных релевантных doc-IDs.
        k: Горизонт отсечения.

    Returns:
        R@K ∈ [0, 1]. Возвращает 0.0 если relevant пуст.
    """
    if not relevant or k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Reciprocal Rank — 1/ранг первого релевантного результата.

    RR = 1 / rank(first relevant), 0 если не найдено.
    MRR = mean(RR) по всем запросам.

    Источник: Voorhees 1999 TREC-8 "The TREC-8 Question Answering Track Report".
    """
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average Precision — площадь под кривой precision-recall.

    AP = Σ P(k) × rel(k) / |relevant|
    MAP = mean(AP) по всем запросам.

    Объединяет precision и recall в одно число.
    Более строга, чем P@K: штрафует за нахождение релевантных ближе к концу.
    """
    if not relevant or not retrieved:
        return 0.0

    hits = 0
    sum_precision = 0.0
    for k, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            sum_precision += hits / k  # P@k только при попадании

    return sum_precision / len(relevant)


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """NDCG@K — нормализованный DCG, учитывает позицию релевантных.

    DCG@K = Σ_{i=1}^{K} rel_i / log2(i+1)
    NDCG@K = DCG@K / IDCG@K,  где IDCG = DCG идеального ранжирования.

    Binary relevance: rel_i ∈ {0, 1}.
    Нормировка делает NDCG сравнимым между запросами с разным числом релевантных.

    Источник: Järvelin & Kekäläinen 2002 ACM TOIS 20(4):422-446.
    """
    if k <= 0 or not retrieved or not relevant:
        return 0.0

    # DCG@K: actual ranking
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 1)

    # IDCG@K: ideal ranking (all relevant documents first)
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_ideal + 1))

    return dcg / idcg if idcg > 0 else 0.0


def compute_query_metrics(
    query: str,
    retrieved: list[str],
    relevant: set[str],
    k_values: list[int],
) -> QueryMetrics:
    """Вычислить все IR-метрики для одного запроса.

    Args:
        query: Текст запроса (для аудита).
        retrieved: Ordered list of retrieved doc-IDs (best first).
        relevant: Set of known-relevant doc-IDs.
        k_values: Список горизонтов, напр. [1, 3, 5, 10].

    Returns:
        QueryMetrics со всеми метриками.
    """
    return QueryMetrics(
        query=query,
        precision_at_k={k: round(precision_at_k(retrieved, relevant, k), 4) for k in k_values},
        recall_at_k={k: round(recall_at_k(retrieved, relevant, k), 4) for k in k_values},
        mrr=round(reciprocal_rank(retrieved, relevant), 4),
        average_precision=round(average_precision(retrieved, relevant), 4),
        ndcg_at_k={k: round(ndcg_at_k(retrieved, relevant, k), 4) for k in k_values},
        n_relevant=len(relevant),
        n_retrieved=len(retrieved),
    )


def aggregate_metrics(
    query_metrics: list[QueryMetrics],
    k_values: list[int],
) -> RetrievalEvalReport:
    """Агрегировать метрики по всем запросам (macro-averaging).

    Macro-averaging (не micro): все запросы равноважны независимо от числа релевантных.
    Стандарт TREC: Manning et al. 2008 §8.4.
    """
    n = len(query_metrics)
    if n == 0:
        return RetrievalEvalReport(
            n_queries=0,
            k_values=k_values,
            mean_precision_at_k={k: 0.0 for k in k_values},
            mean_recall_at_k={k: 0.0 for k in k_values},
            mrr=0.0,
            map_score=0.0,
            mean_ndcg_at_k={k: 0.0 for k in k_values},
            per_query=[],
        )

    mean_p = {k: round(sum(qm.precision_at_k[k] for qm in query_metrics) / n, 4) for k in k_values}
    mean_r = {k: round(sum(qm.recall_at_k[k] for qm in query_metrics) / n, 4) for k in k_values}
    mrr_val = round(sum(qm.mrr for qm in query_metrics) / n, 4)
    map_val = round(sum(qm.average_precision for qm in query_metrics) / n, 4)
    mean_ndcg = {k: round(sum(qm.ndcg_at_k[k] for qm in query_metrics) / n, 4) for k in k_values}

    return RetrievalEvalReport(
        n_queries=n,
        k_values=k_values,
        mean_precision_at_k=mean_p,
        mean_recall_at_k=mean_r,
        mrr=mrr_val,
        map_score=map_val,
        mean_ndcg_at_k=mean_ndcg,
        per_query=query_metrics,
    )
