"""Gradio dashboard for LLM code review."""

from __future__ import annotations

import gradio as gr
from fastapi import FastAPI

from ..data.samples import get_sample_reviews
from ..models.classifier import build_classifier, classify_comment
from ..models.reviewer import review_code

app = FastAPI(title="LLM Code Review Dashboard", version="1.0.0")


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


# ── Helpers ──────────────────────────────────────────────────────────────────

_pipeline = build_classifier()

SAMPLE_DIFFS = {s["category"]: s["code_diff"] for s in get_sample_reviews()}


def _format_comments(comments: list[dict]) -> str:
    if not comments:
        return "No issues found."
    parts = []
    for i, c in enumerate(comments, 1):
        sev = c.get("severity", "?")
        cat = c.get("category", "?")
        line = c.get("line", "")
        text = c.get("comment", "")
        parts.append(f"### {i}. [{sev.upper()}] {cat}\n**Line:** `{line}`\n\n{text}")
    return "\n\n---\n\n".join(parts)


def _category_distribution(comments: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in comments:
        cat = c.get("category", "other")
        counts[cat] = counts.get(cat, 0) + 1
    return counts


# ── Gradio callbacks ─────────────────────────────────────────────────────────


def run_review(diff: str, model: str) -> tuple[str, str]:
    """Run AI review and return formatted comments + category distribution."""
    if not diff.strip():
        return "Please paste a code diff.", ""
    comments = review_code(diff, model=model)
    formatted = _format_comments(comments)
    dist = _category_distribution(comments)
    dist_text = "\n".join(f"- **{k}**: {v}" for k, v in sorted(dist.items()))
    return formatted, dist_text


def run_classify(text: str) -> str:
    """Classify a review comment text."""
    if not text.strip():
        return "Please enter a review comment."
    result = classify_comment(text, _pipeline)
    lines = [
        f"**Predicted category:** {result['category']}",
        f"**Confidence:** {result['confidence']:.1%}",
        "",
        "All probabilities:",
    ]
    for cat, prob in sorted(result["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 30)
        lines.append(f"- {cat}: {prob:.1%} {bar}")
    return "\n".join(lines)


def load_sample(category: str) -> str:
    """Load a sample diff for the given category."""
    return SAMPLE_DIFFS.get(category, "No sample for this category.")


# ── Build UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="LLM Code Review") as demo:
    gr.Markdown("# LLM Code Review Assistant\nPaste a code diff to get AI-powered review.")

    with gr.Tab("AI Review"):
        with gr.Row():
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLE_DIFFS.keys()),
                label="Load sample diff",
            )
            model_dropdown = gr.Dropdown(
                choices=[
                    "claude-sonnet-4-20250514",
                    "claude-haiku-4-20250414",
                ],
                value="claude-sonnet-4-20250514",
                label="Model",
            )
        diff_input = gr.Textbox(lines=12, label="Code Diff", placeholder="Paste diff here...")
        review_btn = gr.Button("Review", variant="primary")
        review_output = gr.Markdown(label="Review Comments")
        dist_output = gr.Markdown(label="Category Distribution")

        sample_dropdown.change(load_sample, inputs=sample_dropdown, outputs=diff_input)
        review_btn.click(
            run_review, inputs=[diff_input, model_dropdown], outputs=[review_output, dist_output]
        )

    with gr.Tab("Comment Classifier"):
        gr.Markdown("Classify a review comment into bug / security / style / performance / docs.")
        comment_input = gr.Textbox(lines=3, label="Review comment text")
        classify_btn = gr.Button("Classify", variant="primary")
        classify_output = gr.Markdown(label="Classification Result")
        classify_btn.click(run_classify, inputs=comment_input, outputs=classify_output)


app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7868)
