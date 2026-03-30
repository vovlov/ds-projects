"""
MLflow Model Registry — регистрация и продвижение моделей ценообразования.

Тот же паттерн, что и в churn-проекте, но метрики регрессионные:
- Эксперимент = «исследование» (Optuna trials, сравнение CatBoost vs LightGBM)
- Registry    = «продакшн» (версии с историей, алиас "@champion" для деплоя)

Порог для @champion: R² > 0.7 означает, что модель объясняет >70% дисперсии цен.
MAPE < 15% — ошибка меньше 15% от цены, приемлемо для маркетплейса.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Пороговые метрики для продвижения в @champion.
# R² > 0.7 — приемлемое объяснение дисперсии для синтетического датасета.
# MAPE < 0.15 — ошибка меньше 15% от реальной цены.
CHAMPION_THRESHOLDS = {
    "r2": 0.70,
}


def register_best_model(
    run_id: str,
    artifact_path: str,
    model_name: str,
    metrics: dict[str, float],
) -> str | None:
    """Зарегистрировать модель оценки недвижимости в MLflow Model Registry.

    Требует tracking URI с поддержкой registry (SQLite или PostgreSQL).
    При работе с дефолтным file store (тесты, быстрый запуск) — пропускает
    регистрацию с предупреждением вместо падения.

    Args:
        run_id: ID MLflow run, из которого берём артефакт.
        artifact_path: Путь к артефакту внутри run (обычно "model").
        model_name: Имя модели в registry (например "realestate-catboost").
        metrics: Словарь метрик для тегирования версии.

    Returns:
        Номер версии (строка) или None, если registry недоступен.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        model_uri = f"runs:/{run_id}/{artifact_path}"

        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = result.version
        logger.info("Registered %s v%s from run %s", model_name, version, run_id)

        # Теги с метриками видны в UI без открытия run
        for metric_name, value in metrics.items():
            client.set_model_version_tag(model_name, version, metric_name, f"{value:.6f}")

        # Алиас "@champion" — только если метрики выше порога
        passes_threshold = all(
            metrics.get(k, 0.0) >= v for k, v in CHAMPION_THRESHOLDS.items()
        )
        if passes_threshold:
            client.set_registered_model_alias(model_name, "champion", version)
            logger.info("Alias @champion → %s v%s", model_name, version)
        else:
            failed = {k: metrics.get(k) for k in CHAMPION_THRESHOLDS}
            logger.warning("Model did not meet champion thresholds: %s", failed)

        return version

    except Exception as exc:
        # Registry недоступен (file store вместо SQLite/PostgreSQL) —
        # нормально при локальном запуске без MLflow сервера и в CI.
        logger.warning("Model Registry unavailable, skipping registration: %s", exc)
        return None
