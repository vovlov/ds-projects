"""
Инкрементальное обучение для предсказания оттока с ADWIN детектором дрейфа.

Incremental Learning for customer churn prediction with ADWIN drift detection.

## Зачем нужно инкрементальное обучение?

Batch-обучение раз в квартал упускает быстрые изменения в поведении клиентов
(новый тариф, кампания конкурентов, изменение политики). Incremental learning
обновляет модель с каждым новым размеченным примером — latency обновления
падает с «дней» до «секунд».

## Архитектура

1. River HoeffdingTreeClassifier — онлайн-классификатор:
   - VFDT (Very Fast Decision Tree, Hulten et al. 2001)
   - Гарантированно сходится к тому же дереву, что и batch C4.5
   - Naive Bayes на листьях — лучше при малом числе примеров на лист

2. River ADWIN — Adaptive Windowing (Bifet & Gavalda 2007):
   - Следит за потоком ошибок модели (0/1 per prediction)
   - Автоматически адаптирует длину окна к текущему распределению
   - drift_detected = True → сигнал для запуска batch-переобучения

3. Periodic snapshot — защита от потери прогресса при рестарте:
   - Каждые snapshot_interval примеров → pickle сериализация модели
   - load_snapshot() восстанавливает полное состояние (модель + ADWIN + scaler)

## Graceful degradation

Если river не установлен (CI без полного окружения) — SimpleFallbackClassifier:
простой байесовский счётчик на основе наблюдённой частоты классов.
API работает, тесты зелёные, но без настоящей адаптации.

Источники:
- Bifet & Gavalda 2007, ADWIN (SDM'07): doi.org/10.1137/1.9781611972795.62
- Hulten, Spencer & Domingos 2001, Hoeffding Tree (KDD): doi.org/10.1145/502512.502529
- River docs: riverml.xyz/latest/
"""

from __future__ import annotations

import logging
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def is_available() -> bool:
    """True если River установлен в текущем окружении."""
    try:
        import river  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class IncrementalConfig:
    """Конфигурация инкрементального классификатора оттока.

    Configuration for the incremental churn classifier.
    """

    model_type: str = "hoeffding_tree"
    # ADWIN delta: вероятность ложной тревоги (меньше = чувствительнее к дрейфу)
    adwin_delta: float = 0.002
    # Сохранять снапшот каждые N примеров
    snapshot_interval: int = 100
    snapshot_dir: str = "/tmp/churn_online_snapshots"
    feature_names: list[str] = field(default_factory=list)


@dataclass
class DriftState:
    """Текущее состояние детектора ADWIN.

    Current state of the ADWIN drift detector.
    """

    n_detected: int = 0
    n_samples_since_last: int = 0
    last_detected_at: str | None = None
    # Оценка текущей ошибки из внутреннего состояния ADWIN
    current_error_rate: float = 0.0


@dataclass
class SnapshotInfo:
    """Метаданные последнего снапшота модели.

    Metadata for the last saved model snapshot.
    """

    snapshot_id: str
    path: str
    n_samples: int
    timestamp: str
    drift_count_at_snapshot: int


@dataclass
class IncrementalResult:
    """Результат предсказания инкрементального классификатора.

    Prediction result from the incremental classifier.
    """

    churn_probability: float
    churn_prediction: bool
    risk_level: str
    model_type: str
    n_samples_seen: int
    drift_detected_this_step: bool = False
    drift_state: DriftState | None = None


@dataclass
class LearnResult:
    """Результат шага обучения на одном примере.

    Result of a single incremental learning step.
    """

    n_samples_seen: int
    error: float
    drift_detected: bool
    snapshot_saved: bool
    snapshot_info: SnapshotInfo | None
    drift_state: DriftState


class IncrementalChurnLearner:
    """Инкрементальный классификатор оттока на River с ADWIN детектором дрейфа.

    Incremental churn classifier using River with ADWIN drift detection.

    Пример использования / Example usage:

        learner = IncrementalChurnLearner()

        # Предсказать до обучения (fallback: 50/50)
        result = learner.predict_one({"tenure": 12.0, "MonthlyCharges": 70.0})

        # Обучить на размеченном примере
        learn_result = learner.learn_one({"tenure": 12.0, "MonthlyCharges": 70.0}, label=1)

        if learn_result.drift_detected:
            # ADWIN обнаружил дрейф → запустить batch-переобучение
            trigger_batch_retraining()
    """

    def __init__(self, config: IncrementalConfig | None = None) -> None:
        self._config = config or IncrementalConfig()
        self._n_samples = 0
        self._drift_state = DriftState()
        self._last_snapshot: SnapshotInfo | None = None
        self._model: Any = None
        self._adwin: Any = None
        self._scaler: Any = None
        # Fallback: счётчики классов для CI без River
        self._fallback_counts: dict[int, int] = {0: 0, 1: 0}
        self._init_model()

    def _init_model(self) -> None:
        """Инициализировать River-модель или Simple Fallback."""
        if not is_available():
            logger.warning(
                "River not installed — using SimpleFallbackClassifier (class-frequency baseline)"
            )
            return

        from river import drift, preprocessing, tree  # type: ignore[import-untyped]

        # ADWIN следит за потоком 0/1 ошибок (ошибка = предсказание ≠ метка)
        self._adwin = drift.ADWIN(delta=self._config.adwin_delta)
        # Онлайн-нормировка признаков: скользящее среднее + стандартное отклонение
        self._scaler = preprocessing.StandardScaler()

        # HoeffdingTree — гарантированно сходится к batch-дереву при n→∞
        # grace_period=100: строим лист после 100 примеров (меньше шума)
        # delta=1e-7: вероятность принятия неверного разбиения < 1e-7
        self._model = tree.HoeffdingTreeClassifier(
            grace_period=100,
            delta=1e-7,
            leaf_prediction="nb",
        )

    def _features_to_dict(self, features: dict[str, float] | list[float]) -> dict[str, float]:
        """Конвертировать признаки в словарь River {str: float}."""
        if isinstance(features, dict):
            return {str(k): float(v) for k, v in features.items()}
        names = self._config.feature_names or [f"f{i}" for i in range(len(features))]
        return {names[i]: float(features[i]) for i in range(min(len(names), len(features)))}

    def predict_one(self, features: dict[str, float] | list[float]) -> IncrementalResult:
        """Предсказать вероятность оттока для одного примера.

        Predict churn probability for a single example.

        Args:
            features: Словарь {feature_name: value} или список значений.

        Returns:
            IncrementalResult с вероятностью, предсказанием и состоянием дрейфа.
        """
        x = self._features_to_dict(features)

        if not is_available() or self._model is None:
            total = sum(self._fallback_counts.values()) or 1
            # Laplace smoothing: избегаем 0 и 1 при малом числе примеров
            proba = (self._fallback_counts[1] + 1) / (total + 2)
        else:
            x_scaled = self._scaler.transform_one(x) or x
            proba_dict = self._model.predict_proba_one(x_scaled) or {0: 0.5, 1: 0.5}
            proba = float(proba_dict.get(1, 0.5))

        risk = "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"
        model_label = ("river_" + self._config.model_type) if is_available() else "fallback_freq"

        return IncrementalResult(
            churn_probability=round(proba, 4),
            churn_prediction=proba >= 0.5,
            risk_level=risk,
            model_type=model_label,
            n_samples_seen=self._n_samples,
            drift_state=DriftState(
                n_detected=self._drift_state.n_detected,
                n_samples_since_last=self._drift_state.n_samples_since_last,
                last_detected_at=self._drift_state.last_detected_at,
                current_error_rate=self._drift_state.current_error_rate,
            ),
        )

    def learn_one(self, features: dict[str, float] | list[float], label: int) -> LearnResult:
        """Обновить модель одним размеченным примером и проверить ADWIN.

        Update model with one labeled example and check ADWIN for drift.

        Сначала предсказываем (для измерения ошибки), затем обучаем.
        Predict-then-learn порядок обязателен для честной оценки ошибки.

        Args:
            features: Числовые признаки клиента.
            label: 0 = не ушёл, 1 = ушёл (churn).

        Returns:
            LearnResult с n_samples_seen, ошибкой и состоянием дрейфа.
        """
        x = self._features_to_dict(features)
        y = int(label)

        # Обновляем fallback-счётчики независимо от River
        self._fallback_counts[y] = self._fallback_counts.get(y, 0) + 1

        error = 0.5
        drift_detected = False

        if is_available() and self._model is not None:
            # Predict-then-learn: предсказываем ДО обновления модели
            x_scaled = self._scaler.transform_one(x) or x
            pred = self._model.predict_one(x_scaled)
            error = 0.0 if pred == y else 1.0

            # ADWIN обновляется ошибкой 0 или 1
            self._adwin.update(error)

            if self._adwin.drift_detected:
                drift_detected = True
                self._drift_state.n_detected += 1
                self._drift_state.last_detected_at = datetime.now(UTC).isoformat()
                self._drift_state.n_samples_since_last = 0
                logger.warning(
                    "ADWIN concept drift detected at sample %d (total: %d)",
                    self._n_samples,
                    self._drift_state.n_detected,
                )

            # Сначала обновляем scaler, затем трансформируем и обучаем модель
            self._scaler.learn_one(x)
            x_scaled = self._scaler.transform_one(x) or x
            self._model.learn_one(x_scaled, y)

            # ADWIN.estimation — оценка среднего ошибки в текущем окне
            adwin_est = getattr(self._adwin, "estimation", None)
            if adwin_est is not None:
                self._drift_state.current_error_rate = float(adwin_est)

        self._n_samples += 1
        self._drift_state.n_samples_since_last += 1

        # Периодический снапшот — защита от потери прогресса
        snapshot_saved = False
        snapshot_info = None
        if self._n_samples % self._config.snapshot_interval == 0:
            snapshot_info = self._save_snapshot()
            snapshot_saved = True

        return LearnResult(
            n_samples_seen=self._n_samples,
            error=error,
            drift_detected=drift_detected,
            snapshot_saved=snapshot_saved,
            snapshot_info=snapshot_info,
            drift_state=DriftState(
                n_detected=self._drift_state.n_detected,
                n_samples_since_last=self._drift_state.n_samples_since_last,
                last_detected_at=self._drift_state.last_detected_at,
                current_error_rate=self._drift_state.current_error_rate,
            ),
        )

    def _save_snapshot(self) -> SnapshotInfo:
        """Сохранить полное состояние модели в pickle-файл.

        Save full model state (model + ADWIN + scaler) to a pickle file.
        """
        snap_dir = Path(self._config.snapshot_dir)
        snap_dir.mkdir(parents=True, exist_ok=True)

        snapshot_id = str(uuid.uuid4())[:8]
        path = snap_dir / f"churn_online_{self._n_samples}_{snapshot_id}.pkl"

        state = {
            "model": self._model,
            "adwin": self._adwin,
            "scaler": self._scaler,
            "n_samples": self._n_samples,
            "drift_state": self._drift_state,
            "fallback_counts": self._fallback_counts,
            "config": self._config,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

        info = SnapshotInfo(
            snapshot_id=snapshot_id,
            path=str(path),
            n_samples=self._n_samples,
            timestamp=datetime.now(UTC).isoformat(),
            drift_count_at_snapshot=self._drift_state.n_detected,
        )
        self._last_snapshot = info
        logger.info("Snapshot saved: %s (n=%d)", path, self._n_samples)
        return info

    @classmethod
    def load_snapshot(cls, path: str) -> IncrementalChurnLearner:
        """Восстановить модель из снапшота.

        Restore model from a snapshot file.

        Args:
            path: Путь к pickle-файлу снапшота.

        Returns:
            IncrementalChurnLearner с восстановленным состоянием.
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        learner = cls(config=state["config"])
        learner._model = state["model"]
        learner._adwin = state["adwin"]
        learner._scaler = state["scaler"]
        learner._n_samples = state["n_samples"]
        learner._drift_state = state["drift_state"]
        learner._fallback_counts = state["fallback_counts"]
        return learner

    def get_status(self) -> dict[str, Any]:
        """Вернуть текущее состояние модели для health-check и мониторинга.

        Return current model status for health-check and monitoring.
        """
        return {
            "model_type": (
                ("river_" + self._config.model_type) if is_available() else "fallback_freq"
            ),
            "river_available": is_available(),
            "n_samples_seen": self._n_samples,
            "n_drift_detections": self._drift_state.n_detected,
            "current_error_rate": round(self._drift_state.current_error_rate, 4),
            "last_drift_at": self._drift_state.last_detected_at,
            "adwin_delta": self._config.adwin_delta,
            "snapshot_interval": self._config.snapshot_interval,
            "last_snapshot": (
                {
                    "snapshot_id": self._last_snapshot.snapshot_id,
                    "path": self._last_snapshot.path,
                    "n_samples": self._last_snapshot.n_samples,
                    "timestamp": self._last_snapshot.timestamp,
                }
                if self._last_snapshot
                else None
            ),
            "class_distribution": {
                "n_non_churn": self._fallback_counts.get(0, 0),
                "n_churn": self._fallback_counts.get(1, 0),
            },
        }
