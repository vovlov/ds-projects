"""FastAPI app for LLM code review service."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ..models.classifier import build_classifier, classify_comment
from ..models.multi_review import MultiReviewResult as _MultiReviewResult
from ..models.multi_review import multi_model_review
from ..models.reviewer import review_code

app = FastAPI(title="LLM Code Review API", version="1.0.0")

# Pre-train classifier at startup
_pipeline = build_classifier()


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
