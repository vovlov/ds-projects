"""Review comment category classifier (TF-IDF + LogisticRegression)."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..data.samples import CATEGORIES, get_sample_reviews


def build_classifier() -> Pipeline:
    """Train a TF-IDF + LogisticRegression pipeline on sample review data."""
    samples = get_sample_reviews()
    texts = [s["review_comment"] for s in samples]
    labels = [s["category"] for s in samples]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipeline.fit(texts, labels)
    return pipeline


def classify_comment(text: str, pipeline: Pipeline | None = None) -> dict:
    """Classify a single review comment into a category.

    Returns dict with keys: category, confidence, all_probabilities.
    """
    if pipeline is None:
        pipeline = build_classifier()

    proba = pipeline.predict_proba([text])[0]
    classes = pipeline.classes_
    predicted = classes[proba.argmax()]
    confidence = float(proba.max())

    return {
        "category": predicted,
        "confidence": round(confidence, 3),
        "all_probabilities": {c: round(float(p), 3) for c, p in zip(classes, proba)},
    }


def classify_batch(texts: list[str], pipeline: Pipeline | None = None) -> list[dict]:
    """Classify multiple review comments."""
    if pipeline is None:
        pipeline = build_classifier()
    return [classify_comment(t, pipeline) for t in texts]


def get_categories() -> tuple[str, ...]:
    """Return the valid category labels."""
    return CATEGORIES
