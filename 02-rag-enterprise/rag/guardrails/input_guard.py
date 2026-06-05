"""
LLM Input Guardrails — защита RAG пайплайна от вредоносных входных запросов.

Архитектура Production Safety Layers (digitalapplied.com/blog/llm-guardrails 2026):
  Layer 1: Input Validation — длина, PII, injection patterns
  Layer 2: Topic Classification — фильтрация off-domain запросов

Все проверки работают на regex / эвристиках без LLM-вызова (CI-friendly).
PII маскирование на входе снижает риск утечки персональных данных через логи.

Источники:
- OWASP LLM Top 10 2025 (LLM01: Prompt Injection)
- EU AI Act Article 10 (data governance, PII minimization)
- futureagi.com/blog/ultimate-guide-llm-guardrails-2026
- NIST AI RMF 2.0 (GOVERN 1.2, MANAGE 2.2)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum


class ThreatType(StrEnum):
    """Типы угроз, обнаруживаемых входным защитником."""

    PROMPT_INJECTION = "prompt_injection"
    PII_IN_QUERY = "pii_in_query"
    OFF_TOPIC = "off_topic"
    QUERY_TOO_LONG = "query_too_long"
    EMPTY_QUERY = "empty_query"


@dataclass
class InputGuardResult:
    """Результат проверки входного запроса.

    Attributes:
        is_safe: False если обнаружены блокирующие угрозы (injection, пустой запрос).
        threats: Список всех обнаруженных угроз (блокирующих и предупреждений).
        sanitized_query: Запрос с замаскированными PII (или оригинал если PII нет).
        risk_score: Суммарный риск от 0.0 до 1.0 (max по всем угрозам).
        details: Словарь с деталями по каждой угрозе для аудит-лога.
    """

    is_safe: bool
    threats: list[ThreatType]
    sanitized_query: str
    risk_score: float
    details: dict[str, str] = field(default_factory=dict)


# Паттерны prompt injection (OWASP LLM01 — наиболее критичная угроза для RAG)
_INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(?:(?:previous|all|prior|above)\s+){1,2}(?:instructions?|prompts?|context|rules?)",
    r"forget\s+(everything|all|previous|prior|your\s+instructions)",
    r"you\s+are\s+now\s+(a|an)\s+\w+",
    r"act\s+as\s+(if\s+you\s+are|a|an)\s+",
    r"(system\s*prompt|system\s*message)\s*[:=]",
    r"\bjailbreak\b",
    r"\bdan\s*(mode|prompt|jailbreak)\b",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"\[system\]|\[user\]|\[assistant\]",  # role confusion через markdown-теги
    r"override\s+(safety|guidelines|rules|restrictions|filters|constraints)",
    r"disregard\s+your\s+(training|guidelines|instructions|constraints)",
    r"new\s+instruction[s]?:\s*",
    r"</?(s|system|instruction|context)>",  # XML/HTML injection в промпт
]

# PII regex — работают без внешних библиотек
_PII_PATTERNS: dict[str, str] = {
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
    "phone_ru": r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b",
    "ssn": r"\b\d{3}[\-\s]\d{2}[\-\s]\d{4}\b",
    "credit_card": r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
    "passport_ru": r"\b\d{4}\s?\d{6}\b",  # Серия + номер паспорта РФ
}


class InputGuard:
    """Входной защитник RAG запросов.

    Проверяет пользовательский запрос перед отправкой в RAG-пайплайн:
    - Prompt injection (блокирующая угроза)
    - PII в тексте запроса (предупреждение + маскирование; блокирует при block_pii=True)
    - Off-domain запросы (предупреждение; блокирует при block_off_topic=True)
    - Длина запроса (усечение; предупреждение)

    Args:
        max_query_length: Максимальная длина запроса в символах (default: 2000).
        domain_keywords: Ключевые слова домена — отсутствие всех → off_topic сигнал.
        block_pii: Блокировать запросы с PII (default: False — только предупреждение).
        block_off_topic: Блокировать off-domain запросы (default: False).
    """

    def __init__(
        self,
        max_query_length: int = 2000,
        domain_keywords: list[str] | None = None,
        block_pii: bool = False,
        block_off_topic: bool = False,
    ) -> None:
        self.max_query_length = max_query_length
        self.domain_keywords = domain_keywords or []
        self.block_pii = block_pii
        self.block_off_topic = block_off_topic
        self._injection_re = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
        self._pii_re = {k: re.compile(v) for k, v in _PII_PATTERNS.items()}

    def check(self, query: str) -> InputGuardResult:
        """Проверить запрос на безопасность.

        Args:
            query: Пользовательский запрос для проверки.

        Returns:
            InputGuardResult с флагами безопасности, маскированным запросом и деталями.
        """
        if not query or not query.strip():
            return InputGuardResult(
                is_safe=False,
                threats=[ThreatType.EMPTY_QUERY],
                sanitized_query="",
                risk_score=1.0,
                details={"empty_query": "Query is empty or whitespace-only"},
            )

        threats: list[ThreatType] = []
        details: dict[str, str] = {}
        risk_score = 0.0
        sanitized = query

        # --- Длина ---
        if len(query) > self.max_query_length:
            threats.append(ThreatType.QUERY_TOO_LONG)
            details["query_too_long"] = (
                f"Query length {len(query)} exceeds max {self.max_query_length}"
            )
            risk_score = max(risk_score, 0.3)
            sanitized = query[: self.max_query_length]

        # --- Prompt injection ---
        matched: list[str] = []
        for rx in self._injection_re:
            m = rx.search(query)
            if m:
                matched.append(m.group(0)[:60])
        if matched:
            threats.append(ThreatType.PROMPT_INJECTION)
            details["prompt_injection"] = (
                f"Detected {len(matched)} injection pattern(s): {matched[0]!r}"
            )
            risk_score = max(risk_score, 0.95)

        # --- PII в запросе ---
        pii_found: list[str] = []
        for pii_type, rx in self._pii_re.items():
            if rx.search(sanitized):
                pii_found.append(pii_type)
                sanitized = rx.sub(f"[{pii_type.upper()}_REDACTED]", sanitized)
        if pii_found:
            threats.append(ThreatType.PII_IN_QUERY)
            details["pii_in_query"] = f"PII types detected: {', '.join(pii_found)}"
            risk_score = max(risk_score, 0.4)

        # --- Off-topic (только если настроены ключевые слова домена) ---
        if self.domain_keywords:
            query_lower = query.lower()
            on_topic = any(kw.lower() in query_lower for kw in self.domain_keywords)
            if not on_topic:
                threats.append(ThreatType.OFF_TOPIC)
                details["off_topic"] = (
                    f"Query does not match any of {len(self.domain_keywords)} domain keyword(s)"
                )
                risk_score = max(risk_score, 0.5)

        # --- Определяем блокирующие угрозы ---
        blocking: set[ThreatType] = {ThreatType.PROMPT_INJECTION, ThreatType.EMPTY_QUERY}
        if self.block_pii:
            blocking.add(ThreatType.PII_IN_QUERY)
        if self.block_off_topic:
            blocking.add(ThreatType.OFF_TOPIC)

        is_safe = not any(t in blocking for t in threats)

        return InputGuardResult(
            is_safe=is_safe,
            threats=threats,
            sanitized_query=sanitized,
            risk_score=round(risk_score, 4),
            details=details,
        )

    def is_injection_attempt(self, query: str) -> bool:
        """Быстрая проверка только на prompt injection (без полной валидации)."""
        return any(rx.search(query) for rx in self._injection_re)
