"""FastAPI app for LLM code review service."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from ..models.classifier import build_classifier, classify_comment
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
