"""
LLM Output Guardrails — фильтрация и защита ответов RAG пайплайна.

Проверяет сгенерированный ответ перед отдачей пользователю:
  - PII маскирование (GDPR Article 5 — data minimization)
  - Детектирование вредоносного контента в ответе
  - Предупреждение об ответах без источников (risk of hallucination)

Источники:
- orq.ai/blog/llm-guardrails (Production Guardrails Guide 2026)
- GDPR Article 5 (data minimization principle)
- OWASP LLM Top 10 2025 (LLM02: Insecure Output Handling)
- EU AI Act Article 13 (transparency, logging)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class OutputThreatType(StrEnum):
    """Типы угроз, обнаруживаемых выходным защитником."""

    PII_IN_ANSWER = "pii_in_answer"
    HARMFUL_CONTENT = "harmful_content"
    ANSWER_TOO_SHORT = "answer_too_short"
    NO_SOURCES = "no_sources"


@dataclass
class OutputGuardResult:
    """Результат проверки выходного ответа RAG.

    Attributes:
        is_safe: False только при обнаружении вредоносного контента.
        threats: Список всех угроз (блокирующих и предупреждений).
        filtered_answer: Ответ с замаскированными PII (или фильтр-сообщение при harmful).
        risk_score: Суммарный риск 0.0-1.0.
        pii_types_found: Типы PII, найденные и замаскированные в ответе.
    """

    is_safe: bool
    threats: list[OutputThreatType]
    filtered_answer: str
    risk_score: float
    pii_types_found: list[str] = field(default_factory=list)


# PII — те же паттерны что и в input_guard для симметрии
_PII_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "phone_ru": r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b",
    "credit_card": r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
    "ssn": r"\b\d{3}[\-\s]\d{2}[\-\s]\d{4}\b",
}

# Паттерны вредоносного контента в ответах LLM (консервативный список)
_HARMFUL_PATTERNS: list[str] = [
    r"how\s+to\s+(make|build|create|synthesize)\s+(?:a\s+)?(bomb|explosive|weapon|poison|drug)",
    r"(suicide\s+method|how\s+to\s+self[\s\-]?harm|steps?\s+to\s+(kill|harm)\s+(yourself|oneself))",
    r"(how\s+to\s+hack\s+into|unauthorized\s+access\s+to\s+|bypass\s+authentication)",
    r"step[\s\-]by[\s\-]step\s+(instructions?|guide)\s+(to|for)\s+(attack|exploit|compromise)",
]

_HARMFUL_REPLACEMENT = (
    "[Content filtered by output guardrail — potential harmful information detected]"
)


class OutputGuard:
    """Выходной защитник ответов RAG пайплайна.

    Применяется после генерации ответа перед отдачей пользователю.
    Маскирует PII, фильтрует вредоносный контент.

    Args:
        mask_pii: Заменять PII на [TYPE_REDACTED] (default: True).
        min_answer_length: Минимальная длина ответа в символах (default: 10).
    """

    def __init__(self, mask_pii: bool = True, min_answer_length: int = 10) -> None:
        self.mask_pii = mask_pii
        self.min_answer_length = min_answer_length
        self._pii_re = {k: re.compile(v) for k, v in _PII_PATTERNS.items()}
        self._harmful_re = [re.compile(p, re.IGNORECASE) for p in _HARMFUL_PATTERNS]

    def check(self, answer: str, sources: list[str] | None = None) -> OutputGuardResult:
        """Проверить и отфильтровать ответ RAG перед отдачей пользователю.

        Args:
            answer: Сгенерированный ответ LLM.
            sources: Список источников (имена файлов / URL).

        Returns:
            OutputGuardResult с маскированным ответом и флагами угроз.
        """
        threats: list[OutputThreatType] = []
        risk_score = 0.0
        filtered = answer
        pii_found: list[str] = []

        # --- Длина ответа ---
        if len(answer.strip()) < self.min_answer_length:
            threats.append(OutputThreatType.ANSWER_TOO_SHORT)
            risk_score = max(risk_score, 0.2)

        # --- Проверка вредоносного контента (до PII — если harmful → заменяем всё) ---
        is_harmful = any(rx.search(answer) for rx in self._harmful_re)
        if is_harmful:
            threats.append(OutputThreatType.HARMFUL_CONTENT)
            risk_score = 1.0
            filtered = _HARMFUL_REPLACEMENT
            # PII не маскируем — ответ уже заменён
            return OutputGuardResult(
                is_safe=False,
                threats=threats,
                filtered_answer=filtered,
                risk_score=risk_score,
                pii_types_found=[],
            )

        # --- PII маскирование ---
        if self.mask_pii:
            for pii_type, rx in self._pii_re.items():
                if rx.search(filtered):
                    pii_found.append(pii_type)
                    filtered = rx.sub(f"[{pii_type.upper()}_REDACTED]", filtered)
            if pii_found:
                threats.append(OutputThreatType.PII_IN_ANSWER)
                risk_score = max(risk_score, 0.6)

        # --- Ответ без источников (риск галлюцинации) ---
        if not sources:
            threats.append(OutputThreatType.NO_SOURCES)
            risk_score = max(risk_score, 0.1)

        return OutputGuardResult(
            is_safe=True,  # только HARMFUL_CONTENT → is_safe=False
            threats=threats,
            filtered_answer=filtered,
            risk_score=round(risk_score, 4),
            pii_types_found=pii_found,
        )

    def mask_answer(self, answer: str) -> str:
        """Быстрое PII-маскирование без полной проверки (для логирования)."""
        result = answer
        for pii_type, rx in self._pii_re.items():
            result = rx.sub(f"[{pii_type.upper()}_REDACTED]", result)
        return result
