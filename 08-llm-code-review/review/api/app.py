"""FastAPI app for LLM code review service."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ..data.pr_dataset import get_pr_dataset, get_pr_stats
from ..evaluation.golden_dataset import get_golden_dataset
from ..evaluation.judge import JudgeVerdict, RegressionResult
from ..evaluation.judge import evaluate_review as _evaluate_review
from ..evaluation.judge import run_regression_suite as _run_regression_suite
from ..models.classifier import build_classifier, classify_comment
from ..models.lora_adapter import LoRAAdapter, LoRAConfig
from ..models.multi_review import MultiReviewResult as _MultiReviewResult
from ..models.multi_review import multi_model_review
from ..models.reviewer import review_code

app = FastAPI(title="LLM Code Review API", version="1.0.0")

# Pre-train classifier at startup
_pipeline = build_classifier()

# LoRA adapter (lazy-initialised on first /adapter/train call)
_adapter: LoRAAdapter | None = None


def _reset_adapter() -> None:
    """Сбросить адаптер (для тестовой изоляции)."""
    global _adapter
    _adapter = None


# ── Request / Response schemas ───────────────────────────────────────────────


class ReviewRequest(BaseModel):
    diff: str
    model: str = "claude-sonnet-4-20250514"


class ReviewComment(BaseModel):
    line: str
    category: str
    comment: str
    severity: str


class ReviewResponse(BaseModel):
    comments: list[ReviewComment]


class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    category: str
    confidence: float
    all_probabilities: dict[str, float]


class MultiReviewRequest(BaseModel):
    diff: str
    correctness_model: str = "claude-haiku-4-20250414"
    security_model: str = "claude-haiku-4-20250414"


class EvaluateReviewRequest(BaseModel):
    example_id: str
    review_comments: list[dict]
    judge_model: str = "claude-haiku-4-20250414"


class RegressionRequest(BaseModel):
    threshold: float = 0.5
    use_lexical: bool = True  # default True — CI-safe without API key


class MultiReviewSummary(BaseModel):
    total: int
    correctness_issues: int
    security_issues: int
    by_severity: dict[str, int]
    consistency_score: float
    verdict: str


class MultiReviewResponse(BaseModel):
    correctness_comments: list[ReviewComment]
    security_comments: list[ReviewComment]
    all_comments: list[ReviewComment]
    consistency_score: float
    summary: MultiReviewSummary


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "healthy", "service": "llm-code-review"}


@app.post("/review", response_model=ReviewResponse)
def post_review(req: ReviewRequest) -> ReviewResponse:
    """Generate AI review comments for a code diff."""
    comments = review_code(req.diff, model=req.model)
    return ReviewResponse(
        comments=[
            ReviewComment(
                line=str(c.get("line", "")),
                category=c.get("category", "bug"),
                comment=c.get("comment", ""),
                severity=c.get("severity", "major"),
            )
            for c in comments
        ]
    )


@app.post("/classify", response_model=ClassifyResponse)
def post_classify(req: ClassifyRequest) -> ClassifyResponse:
    """Classify a review comment into a category (sklearn, no LLM needed)."""
    result = classify_comment(req.text, _pipeline)
    return ClassifyResponse(**result)


@app.post("/review/multi", response_model=MultiReviewResponse)
def post_multi_review(req: MultiReviewRequest) -> MultiReviewResponse:
    """Two-pass multi-model review: correctness pass + security pass.

    Два независимых прохода снижают перекрёстное загрязнение. Возвращает
    self-consistency score и verdict (pass/review_required/fail/api_key_missing).
    """
    result: _MultiReviewResult = multi_model_review(
        req.diff,
        correctness_model=req.correctness_model,
        security_model=req.security_model,
    )

    def _to_comment(c: dict) -> ReviewComment:
        return ReviewComment(
            line=str(c.get("line") or ""),
            category=c.get("category") or "bug",
            comment=c.get("comment") or "",
            severity=c.get("severity") or "major",
        )

    return MultiReviewResponse(
        correctness_comments=[_to_comment(c) for c in result.correctness_comments],
        security_comments=[_to_comment(c) for c in result.security_comments],
        all_comments=[_to_comment(c) for c in result.all_comments],
        consistency_score=result.consistency_score,
        summary=MultiReviewSummary(**result.summary),
    )


@app.get("/evaluate/dataset")
def get_evaluate_dataset() -> dict:
    """Return metadata about the golden evaluation dataset.

    Возвращает метаданные золотого датасета: число примеров, разбивку по доменам.
    """
    examples = get_golden_dataset()
    by_domain: dict[str, int] = {}
    for ex in examples:
        by_domain[ex.domain] = by_domain.get(ex.domain, 0) + 1
    return {
        "n_examples": len(examples),
        "by_domain": by_domain,
        "ids": [ex.id for ex in examples],
    }


@app.post("/evaluate/review")
def post_evaluate_review(req: EvaluateReviewRequest) -> dict:
    """Judge a code review against a golden dataset example.

    Оценивает качество ревью (faithfulness/helpfulness/FPR) по одному примеру
    из золотого датасета. Использует LLM-as-Judge или лексическую эвристику.
    """
    examples = {ex.id: ex for ex in get_golden_dataset()}
    if req.example_id not in examples:
        raise HTTPException(
            status_code=404,
            detail=f"Example '{req.example_id}' not found. Available IDs: {list(examples.keys())}",
        )
    example = examples[req.example_id]
    # Force lexical judge when no API key (CI-safe)
    use_lexical = not os.environ.get("ANTHROPIC_API_KEY", "")
    from ..evaluation.judge import _lexical_judge

    verdict: JudgeVerdict = (
        _lexical_judge(example, req.review_comments)
        if use_lexical
        else _evaluate_review(example, req.review_comments, model=req.judge_model)
    )
    return verdict.to_dict()


@app.post("/evaluate/regression")
def post_evaluate_regression(req: RegressionRequest) -> dict:
    """Run LLM-as-Judge regression suite across all 20 golden examples.

    Запускает регрессионный тест по полному золотому датасету. Возвращает
    агрегатные метрики и pass/fail по порогу avg_overall_score >= threshold.
    """
    result: RegressionResult = _run_regression_suite(
        threshold=req.threshold,
        use_lexical=req.use_lexical or not os.environ.get("ANTHROPIC_API_KEY", ""),
    )
    return result.to_dict()


# ── LoRA Adapter endpoints ────────────────────────────────────────────────────


class AdapterTrainRequest(BaseModel):
    """Запрос на обучение LoRA адаптера. / Train a LoRA adapter on domain examples."""

    domain: str = "security"
    """Категория/домен для специализации: security/bug/performance/style/documentation."""

    rank: int = 4
    """Ранг адаптера (r). Чем меньше — тем меньше параметров. / Adapter rank."""

    alpha: float = 16.0
    """Масштабирующий коэффициент LoRA. / LoRA scaling factor."""

    n_epochs: int = 100
    """Число эпох обучения. / Training epochs."""

    learning_rate: float = 0.05
    """Шаг обучения. / Learning rate."""

    custom_texts: list[str] = []
    """Кастомные тексты для fine-tuning (опционально). / Custom texts for fine-tuning."""

    custom_labels: list[str] = []
    """Метки для кастомных текстов. / Labels for custom texts."""


class AdapterPredictRequest(BaseModel):
    """Запрос на классификацию с адаптером. / Predict using the active adapter."""

    text: str


@app.post("/adapter/train")
def post_adapter_train(req: AdapterTrainRequest) -> dict:
    """Обучить LoRA адаптер на domain-specific PR примерах.

    Использует синтетический PR датасет для заданного домена.
    Если переданы custom_texts + custom_labels — использует их.

    Train a LoRA adapter on domain-specific PR examples from the synthetic dataset.
    Pass custom_texts + custom_labels to override with your own fine-tuning data.
    """
    global _adapter

    valid_domains = {"security", "bug", "performance", "style", "documentation", "general"}
    if req.domain not in valid_domains:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown domain '{req.domain}'. Valid: {sorted(valid_domains)}",
        )
    if req.rank < 1 or req.rank > 64:
        raise HTTPException(status_code=422, detail="rank must be between 1 and 64")

    if req.custom_texts and req.custom_labels:
        if len(req.custom_texts) != len(req.custom_labels):
            raise HTTPException(
                status_code=422,
                detail="custom_texts and custom_labels must have the same length",
            )
        texts = req.custom_texts
        labels = req.custom_labels
    else:
        # Use synthetic PR dataset filtered by domain
        examples = get_pr_dataset()
        if req.domain != "general":
            examples = [ex for ex in examples if ex.category == req.domain]
        if not examples:
            raise HTTPException(
                status_code=422,
                detail=f"No examples found for domain '{req.domain}'",
            )
        texts = [ex.review_comment for ex in examples]
        labels = [ex.category for ex in examples]

    config = LoRAConfig(
        rank=req.rank,
        alpha=req.alpha,
        n_epochs=req.n_epochs,
        learning_rate=req.learning_rate,
        target_domain=req.domain,
    )
    _adapter = LoRAAdapter(_pipeline, config)
    result = _adapter.fit(texts, labels)
    return {
        **result.to_dict(),
        "adapter_norm": _adapter.adapter_norm(),
        "n_training_texts": len(texts),
    }


@app.post("/adapter/predict")
def post_adapter_predict(req: AdapterPredictRequest) -> dict:
    """Классифицировать текст с активным LoRA адаптером.

    Возвращает category, confidence, base_confidence и adaptation_delta
    (насколько адаптер изменил уверенность базовой модели).

    Classify a review comment using the active LoRA adapter.
    Returns confidence delta to show adapter impact vs base model.
    """
    if _adapter is None or not _adapter.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="No adapter trained yet. Call POST /adapter/train first.",
        )
    result = _adapter.predict(req.text)
    return result.to_dict()


@app.get("/adapter/status")
def get_adapter_status() -> dict:
    """Состояние активного LoRA адаптера.

    LoRA adapter status: fitted, domain, rank, loss metrics.
    """
    if _adapter is None or not _adapter.is_fitted:
        return {"fitted": False, "domain": None, "rank": None}

    tr = _adapter.train_result
    return {
        "fitted": True,
        "domain": _adapter.config.target_domain,
        "rank": _adapter.config.rank,
        "alpha": _adapter.config.alpha,
        "adapter_norm": _adapter.adapter_norm(),
        "train_result": tr.to_dict() if tr else None,
    }


@app.get("/adapter/dataset/stats")
def get_adapter_dataset_stats() -> dict:
    """Статистика синтетического PR датасета для LoRA fine-tuning.

    PR dataset statistics: total, by_category, by_domain, by_severity.
    """
    return get_pr_stats()
