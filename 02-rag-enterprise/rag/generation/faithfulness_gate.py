"""
Faithfulness Gate — верификация ответа RAG по retrieved-контексту.

Agentic RAG pattern: второй LLM-вызов (или лексическая эвристика без API)
проверяет, поддержан ли сгенерированный ответ retrieved-чанками.

Если ANTHROPIC_API_KEY доступен — используется быстрая модель (Haiku) как
судья. Иначе — лексические метрики из ragas_eval.py (без внешних зависимостей).

Источники:
- Self-RAG: https://arxiv.org/abs/2310.11511
- RAGFlow year-end review 2025
- DEV.to RAG Blueprint 2026
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..evaluation.ragas_eval import faithfulness as lexical_faithfulness

# Промпт для LLM-судьи: структурированный вывод с вердиктом и оценкой.
# Haiku используется как судья (быстро и дёшево) — отдельно от основной генерации.
_JUDGE_SYSTEM_PROMPT = """You are a faithfulness judge for RAG systems.

Your task: determine if every factual claim in the given answer is supported
by the provided context chunks.

Strict rules:
1. Answer FAITHFUL only if ALL factual claims trace back to the context.
2. Answer UNFAITHFUL if ANY claim is not in the context (hallucination).
3. Ignore style, grammar, or rephrasing — focus only on factual grounding.

Respond ONLY in this exact format (no extra text):
VERDICT: FAITHFUL|UNFAITHFUL
SCORE: 0.XX
REASON: one concise sentence"""

_JUDGE_PROMPT_TEMPLATE = """Context chunks:
{context}

---

Answer to verify:
{answer}

---

Is this answer fully supported by the context above?"""


@dataclass
class FaithfulnessResult:
    """Результат проверки верности ответа источникам.

    Attributes:
        score: Оценка верности от 0.0 до 1.0.
        is_faithful: True если score >= threshold (ответ считается верным).
        verdict: 'FAITHFUL', 'UNFAITHFUL' или 'UNKNOWN'.
        reason: Краткое объяснение вердикта.
        method: 'llm' (второй вызов API) или 'lexical' (без API).
    """

    score: float
    is_faithful: bool
    verdict: str
    reason: str
    method: str  # 'llm' | 'lexical'


def check_faithfulness(
    answer: str,
    contexts: list[str],
    threshold: float = 0.5,
    model: str = "claude-haiku-4-5-20251001",
) -> FaithfulnessResult:
    """Проверить верность RAG-ответа retrieved-контексту (faithfulness gate).

    Ключевой паттерн Agentic RAG: пайплайн сам верифицирует качество ответа
    перед тем, как вернуть его пользователю. Это снижает hallucination rate
    и даёт пользователю явный сигнал о надёжности ответа (confidence_score).

    Режимы работы:
    - LLM mode (ANTHROPIC_API_KEY задан): второй вызов к быстрой Haiku-модели.
    - Lexical mode (без API): лексическое приближение из ragas_eval.

    Args:
        answer: Сгенерированный RAG-ответ для проверки.
        contexts: Список retrieved текстов (чанков).
        threshold: Порог для is_faithful (по умолчанию 0.5).
        model: Модель-судья для LLM-режима (Haiku — быстро и дёшево).

    Returns:
        FaithfulnessResult с оценкой, вердиктом и методом.
    """
    if not answer or not contexts:
        return FaithfulnessResult(
            score=0.0,
            is_faithful=False,
            verdict="UNFAITHFUL",
            reason="Empty answer or no context provided.",
            method="lexical",
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _lexical_check(answer, contexts, threshold)

    return _llm_check(answer, contexts, threshold, model, api_key)


def _lexical_check(answer: str, contexts: list[str], threshold: float) -> FaithfulnessResult:
    """Лексическая проверка без LLM — работает в CI без API-ключа."""
    score = lexical_faithfulness(answer, contexts)
    score = round(score, 4)
    is_faithful = score >= threshold

    return FaithfulnessResult(
        score=score,
        is_faithful=is_faithful,
        verdict="FAITHFUL" if is_faithful else "UNFAITHFUL",
        reason="Lexical overlap between answer sentences and context chunks.",
        method="lexical",
    )


def _llm_check(
    answer: str,
    contexts: list[str],
    threshold: float,
    model: str,
    api_key: str,
) -> FaithfulnessResult:
    """Второй LLM-вызов для оценки верности (agentic режим)."""
    from anthropic import Anthropic

    context_text = "\n\n---\n\n".join(contexts)
    prompt = _JUDGE_PROMPT_TEMPLATE.format(context=context_text, answer=answer)

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=128,
        system=_JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    return _parse_judge_response(raw, threshold)


def _parse_judge_response(raw: str, threshold: float) -> FaithfulnessResult:
    """Разобрать структурированный ответ LLM-судьи.

    Ожидаемый формат:
        VERDICT: FAITHFUL|UNFAITHFUL
        SCORE: 0.XX
        REASON: краткое объяснение

    При ошибке парсинга возвращает консервативный результат (unfaithful, score=0).
    Это намеренно: лучше ложная тревога, чем пропущенный hallucination.
    """
    lines: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            lines[key.strip().upper()] = val.strip()

    # Парсим score с защитой от некорректного ответа
    try:
        score = float(lines.get("SCORE", "0.0"))
        score = max(0.0, min(1.0, score))  # clamping в [0, 1]
    except ValueError:
        score = 0.0

    verdict = lines.get("VERDICT", "UNFAITHFUL").upper()
    if verdict not in ("FAITHFUL", "UNFAITHFUL"):
        verdict = "UNFAITHFUL"

    reason = lines.get("REASON", "Could not parse judge response.")
    is_faithful = score >= threshold

    return FaithfulnessResult(
        score=round(score, 4),
        is_faithful=is_faithful,
        verdict=verdict,
        reason=reason,
        method="llm",
    )
