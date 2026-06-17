"""Ensemble anomaly detector — voting aggregator over multiple detector outputs.

Паттерн: production-системы агрегируют скоры нескольких моделей и применяют
стратегию голосования. Stateless aggregator — не хранит историю, применяет
стратегию к текущим голосам.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnsembleConfig:
    """Конфигурация ансамблевого голосования.

    Attributes:
        strategy: "majority" | "weighted" | "any" | "all"
        weights: веса детекторов для weighted-стратегии (нормируются автоматически).
            Ключ = имя детектора, значение > 0.
        min_agreement: порог для majority/weighted стратегий (0 < x ≤ 1)
    """

    strategy: str = "majority"
    weights: dict[str, float] | None = None
    min_agreement: float = 0.5


@dataclass
class DetectorVote:
    """Голос одного детектора аномалий.

    Attributes:
        name: имя детектора (например "cusum", "kalman", "isolation_forest", "esn")
        is_anomaly: булево решение детектора
        score: аномальный скор в [0, 1]; используется в weighted-стратегии
    """

    name: str
    is_anomaly: bool
    score: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclass
class EnsembleResult:
    """Результат ансамблевого голосования.

    Attributes:
        is_anomaly: итоговое решение ансамбля
        confidence: уверенность [0, 1] — взвешенная доля голосов "anomaly"
        strategy: использованная стратегия
        agreement_ratio: доля детекторов, проголосовавших "anomaly"
        n_votes: общее число детекторов
        n_anomaly_votes: число детекторов, сигнализирующих аномалию
        votes: оригинальные голоса
    """

    is_anomaly: bool
    confidence: float
    strategy: str
    agreement_ratio: float
    n_votes: int
    n_anomaly_votes: int
    votes: list[DetectorVote] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_anomaly": self.is_anomaly,
            "confidence": round(self.confidence, 4),
            "strategy": self.strategy,
            "agreement_ratio": round(self.agreement_ratio, 4),
            "n_votes": self.n_votes,
            "n_anomaly_votes": self.n_anomaly_votes,
            "votes": [
                {"name": v.name, "is_anomaly": v.is_anomaly, "score": round(v.score, 4)}
                for v in self.votes
            ],
        }


STRATEGIES: dict[str, str] = {
    "majority": "Аномалия если доля детекторов > min_agreement (default 0.5). "
    "Баланс precision/recall.",
    "weighted": "Взвешенное среднее score_i×w_i ≥ min_agreement. "
    "Опытные детекторы (высокий AUC) получают больший вес.",
    "any": "Аномалия при хотя бы одном голосе 'да'. "
    "Минимальный miss rate — высокая чувствительность.",
    "all": "Аномалия только если все детекторы согласны. "
    "Минимальный false alarm rate — высокая специфичность.",
}


class AnomalyEnsemble:
    """Ансамбль детекторов аномалий с конфигурируемым голосованием.

    Stateless: каждый вызов aggregate() независим, состояние не накапливается.
    Это позволяет использовать класс в многопоточных API без блокировок.

    Strategies:
        majority  — аномалия если ≥ min_agreement (default 50%) детекторов согласны.
                    Оптимально для balanced datasets (равные precision/recall).
        weighted  — взвешенная сумма anomaly scores; детекторы с высоким AUC получают
                    больший вес через config.weights.
        any       — OR-логика: один сигнал = тревога. Используется в safety-critical
                    системах (лучше лишний alert, чем пропущенная аварий).
        all       — AND-логика: консенсус. Применяется когда False Positives дороги
                    (например, автоматическое прерывание производственного процесса).
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        self.config = config or EnsembleConfig()

    def aggregate(self, votes: list[DetectorVote]) -> EnsembleResult:
        """Агрегировать голоса детекторов в итоговое решение.

        Args:
            votes: список голосов DetectorVote от отдельных детекторов

        Returns:
            EnsembleResult с решением, уверенностью и метаданными голосования

        Raises:
            ValueError: если список пуст или стратегия неизвестна
        """
        if not votes:
            raise ValueError("Нужен хотя бы один голос детектора")

        n = len(votes)
        n_anomaly = sum(1 for v in votes if v.is_anomaly)
        agreement_ratio = n_anomaly / n
        strategy = self.config.strategy

        if strategy == "majority":
            is_anomaly = agreement_ratio > self.config.min_agreement
            confidence = agreement_ratio

        elif strategy == "weighted":
            weights = self.config.weights or {}
            # Неизвестным детекторам назначаем вес 1.0 (равный вклад)
            w = [weights.get(v.name, 1.0) for v in votes]
            w_sum = sum(w)
            if w_sum == 0:
                w = [1.0] * n
                w_sum = float(n)
            weighted_score = sum(v.score * wi for v, wi in zip(votes, w)) / w_sum
            is_anomaly = weighted_score > self.config.min_agreement
            confidence = weighted_score

        elif strategy == "any":
            is_anomaly = n_anomaly > 0
            # Confidence = agreement_ratio даёт сигнал о силе сигнала (1 vs все)
            confidence = agreement_ratio

        elif strategy == "all":
            is_anomaly = n_anomaly == n
            confidence = agreement_ratio

        else:
            raise ValueError(
                f"Неизвестная стратегия: {strategy!r}. Допустимые: {list(STRATEGIES)}"
            )

        return EnsembleResult(
            is_anomaly=is_anomaly,
            confidence=confidence,
            strategy=strategy,
            agreement_ratio=agreement_ratio,
            n_votes=n,
            n_anomaly_votes=n_anomaly,
            votes=votes,
        )
