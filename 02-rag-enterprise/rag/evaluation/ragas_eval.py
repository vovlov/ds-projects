"""
Лёгковесная реализация RAGAS-метрик для оценки RAG-пайплайнов.

Inspired by: https://arxiv.org/abs/2309.15217
(RAGAS: Automated Evaluation of Retrieval Augmented Generation)

Оригинальный RAGAS использует LLM для семантической оценки. Здесь реализованы
лексические приближения, которые работают без API-ключей — для CI и быстрой разработки.
В production следует заменить на полный RAGAS с LLM-оценщиком (см. ragas Python package).

Метрики:
- context_precision: Доля релевантных чанков среди retrieved (precision@k)
- context_recall: Доля ответа, подкреплённая retrieved контекстом
- answer_relevance: Насколько ответ отвечает на вопрос (по ключевым словам)
- faithfulness: Доля утверждений ответа, поддержанных контекстом
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any

# Стоп-слова (English + Russian) — не несут смысловой нагрузки для overlap-метрик
_STOPWORDS = frozenset(
    {
        # English
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
        "need",
        "dare",
        "ought",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "from",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "and",
        "or",
        "but",
        "if",
        "because",
        "as",
        "until",
        "while",
        "that",
        "this",
        "it",
        "its",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "just",
        "than",
        "then",
        "when",
        "where",
        "who",
        "which",
        "how",
        "what",
        # Russian
        "и",
        "в",
        "на",
        "с",
        "по",
        "из",
        "за",
        "к",
        "о",
        "от",
        "до",
        "у",
        "под",
        "над",
        "при",
        "без",
        "для",
        "или",
        "но",
        "а",
        "не",
        "это",
        "что",
        "как",
        "то",
        "все",
        "он",
        "она",
        "они",
        "мы",
        "вы",
        "я",
        "его",
        "её",
        "их",
        "нас",
        "вас",
        "ему",
        "ей",
        "им",
        "же",
        "ли",
        "уже",
        "ещё",
        "даже",
        "тоже",
        "также",
        "себя",
    }
)


def _tokenize(text: str) -> set[str]:
    """Токенизировать текст: нижний регистр, без пунктуации, без стоп-слов.

    Простая лексическая токенизация без внешних зависимостей.
    Подходит для метрик на основе overlap.
    """
    text = text.lower()
    # Убираем пунктуацию
    extra_punct = "\u00ab\u00bb\u2014\u2013\u201e\u201c\u201d"  # «»—–„""
    text = text.translate(str.maketrans("", "", string.punctuation + extra_punct))
    tokens = set(text.split())
    return tokens - _STOPWORDS


def _sentence_split(text: str) -> list[str]:
    """Разбить текст на предложения по знакам препинания."""
    # Простой сплиттер без NLTK: по . ! ?
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def context_precision(question: str, contexts: list[str]) -> float:
    """Precision@k: доля контекстных чанков, релевантных вопросу.

    Чанк считается релевантным, если у него есть хотя бы одно общее
    содержательное слово с вопросом (лексический overlap).

    Args:
        question: Вопрос пользователя.
        contexts: Список retrieved текстов (чанков).

    Returns:
        Значение от 0.0 до 1.0. 1.0 = все чанки релевантны.
    """
    if not contexts:
        return 0.0

    question_tokens = _tokenize(question)
    if not question_tokens:
        return 0.0

    relevant = sum(1 for ctx in contexts if _tokenize(ctx) & question_tokens)
    return relevant / len(contexts)


def context_recall(answer: str, contexts: list[str]) -> float:
    """Recall: какая доля предложений ответа подкреплена контекстом.

    Предложение ответа считается поддержанным, если у него есть overlap
    хотя бы с одним чанком контекста.

    Args:
        answer: Сгенерированный ответ.
        contexts: Список retrieved текстов (чанков).

    Returns:
        Значение от 0.0 до 1.0. 1.0 = весь ответ поддержан контекстом.
    """
    if not answer or not contexts:
        return 0.0

    sentences = _sentence_split(answer)
    if not sentences:
        return 0.0

    context_tokens = set()
    for ctx in contexts:
        context_tokens |= _tokenize(ctx)

    supported = sum(1 for sent in sentences if _tokenize(sent) & context_tokens)
    return supported / len(sentences)


def answer_relevance(question: str, answer: str) -> float:
    """Насколько ответ адресует вопрос (лексический overlap).

    Мера того, что ответ вообще связан с заданным вопросом,
    а не уходит в сторону.

    Args:
        question: Вопрос пользователя.
        answer: Сгенерированный ответ.

    Returns:
        Значение от 0.0 до 1.0.
    """
    if not question or not answer:
        return 0.0

    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)

    if not q_tokens or not a_tokens:
        return 0.0

    intersection = q_tokens & a_tokens
    # Jaccard similarity
    union = q_tokens | a_tokens
    return len(intersection) / len(union)


def faithfulness(answer: str, contexts: list[str]) -> float:
    """Верность источникам: доля предложений ответа, поддержанных контекстом.

    Похоже на context_recall, но с акцентом на то, что ответ не «придумывает»
    информацию сверх того, что есть в контексте.

    В полном RAGAS faithfulness оценивает LLM. Здесь — лексическое приближение.

    Args:
        answer: Сгенерированный ответ.
        contexts: Список retrieved текстов (чанков).

    Returns:
        Значение от 0.0 до 1.0. 1.0 = каждое предложение подкреплено контекстом.
    """
    # Алгоритмически идентично context_recall — оставляем как отдельную метрику
    # для совместимости с RAGAS-интерфейсом (разные семантические смыслы в LLM-версии)
    return context_recall(answer, contexts)


@dataclass
class RAGASResult:
    """Результат RAGAS-оценки одного Q&A сэмпла.

    Attributes:
        context_precision: Precision@k retrieved chunks.
        context_recall: Recall на уровне предложений ответа.
        answer_relevance: Overlap вопроса и ответа.
        faithfulness: Доля ответа, подкреплённая контекстом.
        overall: Среднее всех 4 метрик (composite score).
        metadata: Любые дополнительные данные (вопрос, длина ответа, etc.).
    """

    context_precision: float
    context_recall: float
    answer_relevance: float
    faithfulness: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def overall(self) -> float:
        """Composite RAGAS score — среднее арифметическое 4 метрик."""
        return (
            self.context_precision + self.context_recall + self.answer_relevance + self.faithfulness
        ) / 4

    def as_dict(self) -> dict[str, float]:
        """Вернуть метрики в виде словаря для логирования."""
        return {
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "answer_relevance": round(self.answer_relevance, 4),
            "faithfulness": round(self.faithfulness, 4),
            "overall": round(self.overall, 4),
        }


def evaluate_sample(
    question: str,
    answer: str,
    contexts: list[str],
    metadata: dict[str, Any] | None = None,
) -> RAGASResult:
    """Оценить один Q&A сэмпл по всем RAGAS-метрикам.

    Args:
        question: Вопрос пользователя.
        answer: Сгенерированный ответ.
        contexts: Список retrieved текстов (чанков).
        metadata: Произвольные метаданные (id сэмпла, модель, etc.).

    Returns:
        RAGASResult с рассчитанными метриками.

    Example:
        >>> result = evaluate_sample(
        ...     question="What is the VPN policy?",
        ...     answer="All employees must use VPN when working remotely.",
        ...     contexts=["VPN is required for all remote connections."],
        ... )
        >>> result.overall > 0.3
        True
    """
    return RAGASResult(
        context_precision=context_precision(question, contexts),
        context_recall=context_recall(answer, contexts),
        answer_relevance=answer_relevance(question, answer),
        faithfulness=faithfulness(answer, contexts),
        metadata=metadata or {},
    )


def evaluate_dataset(
    samples: list[dict[str, Any]],
) -> dict[str, float]:
    """Оценить набор Q&A сэмплов и вернуть средние метрики.

    Args:
        samples: Список словарей с ключами:
            - "question": str
            - "answer": str
            - "contexts": list[str]
            - "metadata": dict (опционально)

    Returns:
        Словарь со средними значениями каждой метрики по датасету.

    Example:
        >>> samples = [
        ...     {"question": "What is remote work?", "answer": "Remote work...",
        ...      "contexts": ["..."]},
        ... ]
        >>> metrics = evaluate_dataset(samples)
        >>> "overall" in metrics
        True
    """
    if not samples:
        return {
            "context_precision": 0.0,
            "context_recall": 0.0,
            "answer_relevance": 0.0,
            "faithfulness": 0.0,
            "overall": 0.0,
            "num_samples": 0,
        }

    results = [
        evaluate_sample(
            question=s["question"],
            answer=s["answer"],
            contexts=s.get("contexts", []),
            metadata=s.get("metadata"),
        )
        for s in samples
    ]

    n = len(results)
    return {
        "context_precision": round(sum(r.context_precision for r in results) / n, 4),
        "context_recall": round(sum(r.context_recall for r in results) / n, 4),
        "answer_relevance": round(sum(r.answer_relevance for r in results) / n, 4),
        "faithfulness": round(sum(r.faithfulness for r in results) / n, 4),
        "overall": round(sum(r.overall for r in results) / n, 4),
        "num_samples": n,
    }
