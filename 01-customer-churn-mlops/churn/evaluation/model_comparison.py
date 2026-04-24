"""
Automated model comparison reports for churn prediction models.
Автоматические отчёты сравнения моделей для предсказания оттока.

Comparing models on aggregate metrics is sufficient for champion selection
in the churn domain where test sets are large (n > 1000) and AUC variance
is low. Margin of 0.02 corresponds to ~2 SE of AUC (Hanley & McNeil 1982)
for balanced telco datasets — differences below this are practically noise.

Сравнение моделей по агрегированным метрикам достаточно для выбора чемпиона
в задаче предсказания оттока при больших тестовых выборках (n > 1000).
Порог 0.02 соответствует ~2 SE AUC (Hanley & McNeil 1982) для телеком-данных —
меньшие различия практически неотличимы от шума.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# AUC margin that justifies declaring a clear winner.
# Smaller differences might be sampling noise, so we recommend A/B testing instead.
_AUC_SIGNIFICANCE_MARGIN: float = 0.02


@dataclass
class ModelResult:
    """Metrics and metadata for a single trained model.
    Метрики и метаданные одной обученной модели.
    """

    name: str
    roc_auc: float
    f1_score: float
    precision: float = 0.0
    recall: float = 0.0
    training_time_sec: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    run_id: str | None = None


@dataclass
class ComparisonSummary:
    """High-level comparison outcome.
    Итог сравнения моделей.
    """

    winner: str
    winner_auc: float
    runner_up: str | None
    auc_margin: float
    is_significant: bool
    recommendation: str


@dataclass
class ComparisonReport:
    """Full model comparison report with leaderboard.
    Полный отчёт сравнения моделей с таблицей лидеров.
    """

    models: list[ModelResult]
    summary: ComparisonSummary
    timestamp: str
    leaderboard: list[dict[str, Any]] = field(default_factory=list)


def compare_models(results: list[ModelResult]) -> ComparisonReport:
    """Compare multiple trained models and produce a ranking report.
    Сравнить обученные модели и создать отчёт с ранжированием.

    Models are ranked by ROC AUC (primary) then F1 (tiebreaker).
    Winner is significant when AUC gap to runner-up exceeds the margin.
    When only one model is provided, it wins by default.

    Args:
        results: List of model results to compare.

    Returns:
        ComparisonReport with leaderboard and winner recommendation.

    Raises:
        ValueError: If results list is empty.
    """
    if not results:
        raise ValueError("Need at least one ModelResult to compare")

    ranked = sorted(results, key=lambda r: (r.roc_auc, r.f1_score), reverse=True)

    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    auc_margin = winner.roc_auc - runner_up.roc_auc if runner_up else 0.0
    is_significant = runner_up is None or auc_margin >= _AUC_SIGNIFICANCE_MARGIN

    if runner_up is None:
        recommendation = (
            f"Only model: {winner.name} (AUC={winner.roc_auc:.4f}). "
            "No comparison available — register as champion if above threshold."
        )
    elif is_significant:
        recommendation = (
            f"Deploy {winner.name}: AUC={winner.roc_auc:.4f}, "
            f"margin +{auc_margin:.4f} over {runner_up.name} exceeds significance threshold."
        )
    else:
        recommendation = (
            f"Models are close (margin {auc_margin:.4f} < {_AUC_SIGNIFICANCE_MARGIN}). "
            f"Prefer {winner.name} by AUC but validate with A/B test before promoting."
        )

    summary = ComparisonSummary(
        winner=winner.name,
        winner_auc=winner.roc_auc,
        runner_up=runner_up.name if runner_up else None,
        auc_margin=round(auc_margin, 6),
        is_significant=is_significant,
        recommendation=recommendation,
    )

    leaderboard = [
        {
            "rank": i + 1,
            "name": r.name,
            "roc_auc": round(r.roc_auc, 4),
            "f1_score": round(r.f1_score, 4),
            "precision": round(r.precision, 4),
            "recall": round(r.recall, 4),
            "training_time_sec": round(r.training_time_sec, 2),
            "run_id": r.run_id,
        }
        for i, r in enumerate(ranked)
    ]

    return ComparisonReport(
        models=results,
        summary=summary,
        timestamp=datetime.now(UTC).isoformat(),
        leaderboard=leaderboard,
    )


def generate_markdown_report(report: ComparisonReport) -> str:
    """Generate a Markdown comparison table for human review and docs.
    Сформировать Markdown-таблицу сравнения для человека или документации.

    Args:
        report: ComparisonReport from compare_models().

    Returns:
        Markdown string with leaderboard table and summary.
    """
    lines = [
        "# Model Comparison Report",
        f"Generated: {report.timestamp}",
        "",
        "## Leaderboard",
        "",
        "| Rank | Model | AUC | F1 | Precision | Recall | Train (s) |",
        "|------|-------|-----|----|-----------|--------|-----------|",
    ]

    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    for row in report.leaderboard:
        medal = medals.get(row["rank"], "  ")
        lines.append(
            f"| {row['rank']} {medal} | {row['name']} "
            f"| {row['roc_auc']:.4f} | {row['f1_score']:.4f} "
            f"| {row['precision']:.4f} | {row['recall']:.4f} "
            f"| {row['training_time_sec']:.1f} |"
        )

    s = report.summary
    sig_badge = "✅ significant" if s.is_significant else "⚠️ not significant"
    lines += [
        "",
        "## Summary",
        "",
        f"**Winner:** {s.winner} (AUC={s.winner_auc:.4f})",
        f"**AUC Margin:** {s.auc_margin:.4f} — {sig_badge}",
        "",
        f"**Recommendation:** {s.recommendation}",
    ]

    # Top-5 feature importances per model (where available)
    models_with_imp = [m for m in report.models if m.feature_importances]
    if models_with_imp:
        lines += ["", "## Feature Importance (top 5)", ""]
        for m in models_with_imp:
            top5 = sorted(m.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append(f"**{m.name}:** " + ", ".join(f"{k} ({v:.2f})" for k, v in top5))

    return "\n".join(lines)


def generate_json_report(report: ComparisonReport) -> dict[str, Any]:
    """Generate a JSON-serializable comparison report for API responses.
    Сформировать JSON-совместимый отчёт для API-ответов и аудит-лога.

    Args:
        report: ComparisonReport from compare_models().

    Returns:
        Dict ready for json.dumps() or FastAPI response_model.
    """
    return {
        "timestamp": report.timestamp,
        "summary": {
            "winner": report.summary.winner,
            "winner_auc": report.summary.winner_auc,
            "runner_up": report.summary.runner_up,
            "auc_margin": report.summary.auc_margin,
            "is_significant": report.summary.is_significant,
            "recommendation": report.summary.recommendation,
        },
        "leaderboard": report.leaderboard,
    }
