"""Code review generation using Claude API."""

from __future__ import annotations

import json
import os

SYSTEM_PROMPT = """You are a senior code reviewer with 10+ years of experience in Python.

Analyze the given code diff and produce review comments. For each issue found, return a JSON object with:
- "line": the relevant line content or line number from the diff
- "category": one of "bug", "security", "style", "performance", "documentation"
- "comment": a clear, actionable explanation of the issue
- "severity": one of "critical", "major", "minor", "suggestion"

Rules:
1. Focus on real problems, not nitpicks.
2. Explain *why* something is wrong and suggest a fix.
3. Be specific — reference variable names, function signatures, etc.
4. Return a JSON array of objects. If the code looks fine, return an empty array [].
5. Do NOT wrap the JSON in markdown code fences.
"""


def review_code(
    diff: str,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 2048,
) -> list[dict]:
    """Send a code diff to Claude API and return structured review comments.

    Returns a list of dicts with keys: line, category, comment, severity.
    When the API key is missing, returns a single-item list with an error message.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return [
            {
                "line": "",
                "category": "documentation",
                "comment": (
                    "Error: ANTHROPIC_API_KEY not set. "
                    "Please export it in your environment to enable AI reviews."
                ),
                "severity": "critical",
            }
        ]

    # Lazy import — no anthropic dependency needed at import time
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Review the following diff:\n\n```diff\n{diff}\n```",
            }
        ],
    )

    raw = response.content[0].text.strip()

    try:
        comments = json.loads(raw)
    except json.JSONDecodeError:
        # If model returned markdown-fenced JSON, try stripping fences
        cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            comments = json.loads(cleaned)
        except json.JSONDecodeError:
            comments = [
                {
                    "line": "",
                    "category": "bug",
                    "comment": f"Failed to parse model output as JSON: {raw[:200]}",
                    "severity": "major",
                }
            ]

    if isinstance(comments, dict):
        comments = [comments]

    return comments
