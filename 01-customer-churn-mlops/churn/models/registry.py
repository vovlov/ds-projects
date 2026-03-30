"""
MLflow Model Registry — регистрация и продвижение моделей.

Model Registry отделяет жизненный цикл модели от эксперимента:
- Эксперимент = «исследование» (много runs, разные гиперпараметры)
- Registry    = «продакшн» (конкретные версии с историей и аудитом)

Паттерн: best run → registered model → alias "@champion" → деплой из registry.
Алиас "@champion" указывает на текущую лучшую версию — сервис загружает модель
по алиасу, а не по версии, что позволяет откатиться без перекомпиляции образа.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Минимальные метрики для регистрации как champion.
# Модель ниже порога попадает в registry, но не получает алиас.
CHAMPION_THRESHOLDS = {
    "roc_auc": 0.75,
    "f1_score": 0.50,
}


def register_best_model(
    run_id: str,
    artifact_path: str,
    model_name: str,
    metrics: dict[str, float],
) -> str | None:
    """Зарегистрировать модель в MLflow Model Registry.

    Требует tracking URI с поддержкой registry (SQLite или PostgreSQL).
    При работе с дефолтным file store (тесты, быстрый запуск) — пропускает
    регистрацию с предупреждением вместо падения.

    Args:
        run_id: ID MLflow run, из которого берём артефакт.
        artifact_path: Путь к артефакту внутри run (обычно "model").
        model_name: Имя модели в registry (например "churn-catboost").
        metrics: Словарь метрик для тегирования версии.

    Returns:
        Номер версии (строка) или None, если registry недоступен.
    """
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        model_uri = f"runs:/{run_id}/{artifact_path}"

        # Регистрируем — создаём новую версию (или первую, если модели нет)
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
        version = result.version
        logger.info("Registered %s v%s from run %s", model_name, version, run_id)

        # Теги с метриками — видны в UI без открытия run
        for metric_name, value in metrics.items():
            client.set_model_version_tag(model_name, version, metric_name, f"{value:.6f}")

        # Алиас "@champion" присваиваем только если метрики выше порога.
        # Это защита от деградации: плохая модель не попадёт в продакшн автоматически.
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
