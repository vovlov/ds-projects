"""
Write-Audit-Publish (WAP) drift gate для feature store.
Write-Audit-Publish drift gate for the feature store.

Паттерн WAP гарантирует, что новые батчи фичей не вызывают
статистического дрейфа перед публикацией в production.

The WAP pattern ensures new feature batches don't introduce statistical
drift before being promoted to production.

Три шага:
  1. Write  — записываем батч в изолированный staging-буфер
  2. Audit  — вычисляем PSI относительно reference-распределения
  3. Publish — если PSI < порог, продвигаем в production; иначе — карантин

Sources:
  - WAP pattern: lakefs.io/blog/data-engineering-patterns-write-audit-publish (2024)
  - Dagster WAP: dagster.io/blog/python-write-audit-publish (2025)
  - PSI threshold BCBS 2011: PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 significant
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Вычисление PSI без внешних зависимостей
# PSI computation without external dependencies
# ---------------------------------------------------------------------------


def _compute_psi(
    reference: list[float] | np.ndarray,
    current: list[float] | np.ndarray,
    bins: int = 10,
) -> float:
    """
    Population Stability Index (PSI) для одного признака.
    Population Stability Index for a single feature.

    Реализуем локально, чтобы не создавать кросс-проектную зависимость
    на quality/quality/drift.py — проекты должны быть изолированы.
    Implemented locally to keep project boundaries clean.

    Интерпретация / Interpretation:
      PSI < 0.1   — нет дрейфа / stable
      0.1–0.25    — умеренный дрейф / moderate drift
      > 0.25      — значительный дрейф, нужен карантин / quarantine
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    if len(ref) == 0 or len(cur) == 0:
        return 0.0

    # Границы бинов — по объединённому диапазону
    lo = min(ref.min(), cur.min())
    hi = max(ref.max(), cur.max())

    if lo == hi:
        # Все значения одинаковые — дрейфа нет
        return 0.0

    breakpoints = np.linspace(lo, hi, bins + 1)

    ref_counts = np.histogram(ref, bins=breakpoints)[0].astype(float)
    cur_counts = np.histogram(cur, bins=breakpoints)[0].astype(float)

    # Epsilon защищает от деления на ноль / guards against zero division
    eps = 1e-8
    ref_pct = ref_counts / ref_counts.sum() + eps
    cur_pct = cur_counts / cur_counts.sum() + eps

    psi_value = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(psi_value, 6)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AuditResult:
    """
    Результат WAP-аудита для одного батча фичей.
    WAP audit result for a feature batch.

    Attributes:
        draft_id: Уникальный ID staging-батча / Unique staging batch ID.
        feature_name: Имя проверяемой фичи / Feature being audited.
        status: "published" | "quarantined" | "no_reference"
        psi: Значение PSI / PSI value (0.0 if no reference).
        threshold: Порог PSI / PSI threshold used.
        passed: Прошёл ли аудит / Whether audit passed.
        n_reference: Размер reference-выборки / Reference sample size.
        n_current: Размер текущей выборки / Current sample size.
        timestamp: ISO-8601 UTC / Timestamp.
        reason: Человекочитаемая причина / Human-readable reason.
    """

    draft_id: str
    feature_name: str
    status: str  # "published" | "quarantined" | "no_reference"
    psi: float
    threshold: float
    passed: bool
    n_reference: int
    n_current: int
    timestamp: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-сериализуемый словарь / JSON-serializable dict."""
        return {
            "draft_id": self.draft_id,
            "feature_name": self.feature_name,
            "status": self.status,
            "psi": self.psi,
            "threshold": self.threshold,
            "passed": self.passed,
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# WAP Gate
# ---------------------------------------------------------------------------


class WAPGate:
    """
    Write-Audit-Publish gate для feature store.
    WAP drift gate — validates new feature batches before publishing.

    Жизненный цикл батча / Batch lifecycle:
      write() → audit() → publish() или quarantine()

    Или одним вызовом / Or all-in-one:
      write_audit_publish()

    Args:
        psi_threshold: PSI выше этого значения → карантин.
                       PSI above this → quarantine. Default: 0.2 (BCBS-based).
        bins: Количество бинов для PSI / Number of bins for PSI histogram.
    """

    def __init__(self, psi_threshold: float = 0.2, bins: int = 10) -> None:
        self.psi_threshold = psi_threshold
        self.bins = bins

        # Reference distributions: feature_name → array of floats
        self._reference: dict[str, list[float]] = {}

        # Staging area: draft_id → {feature_name, values}
        self._staging: dict[str, dict[str, Any]] = {}

        # Production store: feature_name → latest published values
        self._production: dict[str, list[float]] = {}

        # Audit log: все результаты аудита / full audit trail
        self._audit_log: list[AuditResult] = []

    # ------------------------------------------------------------------
    # Reference management
    # ------------------------------------------------------------------

    def set_reference(self, feature_name: str, values: list[float] | np.ndarray) -> None:
        """
        Устанавливает эталонное распределение для фичи.
        Set reference distribution for a feature.

        Reference — это обычно обучающая выборка или первый батч в production.
        Typically the training distribution or first production batch.
        """
        self._reference[feature_name] = list(np.asarray(values, dtype=float))

    def has_reference(self, feature_name: str) -> bool:
        """Есть ли reference-распределение для фичи / Is reference set."""
        return feature_name in self._reference

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write(self, feature_name: str, values: list[float] | np.ndarray) -> str:
        """
        Шаг 1 — запись в staging-буфер.
        Step 1 — write to isolated staging buffer.

        Возвращает draft_id для последующего аудита / Returns draft_id for auditing.
        """
        draft_id = str(uuid.uuid4())
        self._staging[draft_id] = {
            "feature_name": feature_name,
            "values": list(np.asarray(values, dtype=float)),
        }
        return draft_id

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit(self, draft_id: str) -> AuditResult:
        """
        Шаг 2 — PSI-аудит staging-батча.
        Step 2 — PSI audit of the staged batch.

        Если reference не задан — статус "no_reference", батч пропускается
        (нельзя заблокировать первый батч в холодный старт).
        If no reference set — status "no_reference", batch is let through
        (can't block the first cold-start batch).
        """
        if draft_id not in self._staging:
            raise KeyError(f"draft_id '{draft_id}' not found in staging")

        batch = self._staging[draft_id]
        feature_name = batch["feature_name"]
        current_vals = batch["values"]
        ts = datetime.now(tz=UTC).isoformat()

        # Холодный старт: нет reference → публикуем без аудита
        # Cold start: no reference → publish without audit (first batch sets the baseline)
        if not self.has_reference(feature_name):
            result = AuditResult(
                draft_id=draft_id,
                feature_name=feature_name,
                status="no_reference",
                psi=0.0,
                threshold=self.psi_threshold,
                passed=True,
                n_reference=0,
                n_current=len(current_vals),
                timestamp=ts,
                reason=(
                    "No reference distribution set — cold start. "
                    "Batch published and set as reference."
                ),
            )
            # Первый батч становится reference для следующих
            # First batch becomes the reference for subsequent checks
            self.set_reference(feature_name, current_vals)
            self._audit_log.append(result)
            return result

        ref_vals = self._reference[feature_name]
        psi_value = _compute_psi(ref_vals, current_vals, bins=self.bins)
        passed = psi_value < self.psi_threshold

        if passed:
            status = "published"
            reason = f"PSI={psi_value:.4f} < threshold={self.psi_threshold} — batch approved."
        else:
            status = "quarantined"
            reason = (
                f"PSI={psi_value:.4f} >= threshold={self.psi_threshold} — "
                f"significant drift detected, batch quarantined."
            )

        result = AuditResult(
            draft_id=draft_id,
            feature_name=feature_name,
            status=status,
            psi=psi_value,
            threshold=self.psi_threshold,
            passed=passed,
            n_reference=len(ref_vals),
            n_current=len(current_vals),
            timestamp=ts,
            reason=reason,
        )
        self._audit_log.append(result)
        return result

    # ------------------------------------------------------------------
    # Publish / Quarantine
    # ------------------------------------------------------------------

    def publish(self, draft_id: str, audit_result: AuditResult) -> bool:
        """
        Шаг 3 — публикация, если аудит прошёл.
        Step 3 — promote to production if audit passed.

        Возвращает True если данные опубликованы, False если в карантине.
        Returns True if published, False if quarantined.
        """
        if not audit_result.passed:
            return False

        batch = self._staging.pop(draft_id, None)
        if batch is None:
            return False

        feature_name = batch["feature_name"]
        self._production[feature_name] = batch["values"]
        return True

    # ------------------------------------------------------------------
    # All-in-one WAP
    # ------------------------------------------------------------------

    def write_audit_publish(
        self,
        feature_name: str,
        values: list[float] | np.ndarray,
    ) -> AuditResult:
        """
        Полный WAP-цикл за один вызов.
        Complete WAP cycle in a single call.

        Write → Audit → Publish (или Quarantine).
        Возвращает AuditResult с финальным статусом.
        Returns AuditResult with final status.
        """
        draft_id = self.write(feature_name, values)
        result = self.audit(draft_id)
        self.publish(draft_id, result)
        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_production(self, feature_name: str) -> list[float] | None:
        """Получить production-значения фичи / Get production feature values."""
        return self._production.get(feature_name)

    def get_audit_log(self) -> list[AuditResult]:
        """Полный аудит-лог (иммутабельная копия) / Full immutable audit log."""
        return list(self._audit_log)

    @property
    def n_staging(self) -> int:
        """Количество батчей в staging / Number of batches in staging."""
        return len(self._staging)

    @property
    def n_production(self) -> int:
        """Количество фичей в production / Number of features in production."""
        return len(self._production)
