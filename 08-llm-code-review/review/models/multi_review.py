"""Multi-model cross-check for code review.

Implements a two-pass review strategy inspired by:
- Ericsson 2025 (arxiv 2507.19115): context-aware LLM review in production
- CISC (Confidence-Informed Self-Consistency): quality scoring without a second LLM call
- Semgrep AI (2025): structural analysis as deterministic anchors for LLM reasoning

Два независимых прохода снижают перекрёстное загрязнение между типами проблем:
correctness pass не отвлекается на безопасность, security pass — на баги.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# ── System prompts ────────────────────────────────────────────────────────────

_CORRECTNESS_PROMPT = """You are a senior engineer focused exclusively on correctness review.

Analyze the given code diff for:
- Logic errors (off-by-one, inverted conditions, wrong operators)
- Null/None dereferences and missing edge case handling
- Type mismatches and invalid assumptions about input
- Race conditions and concurrency bugs
- Algorithmic correctness (wrong result, infinite loops)

Rules:
1. ONLY report correctness issues — skip style and security (handled separately).
2. For each issue return JSON with keys: "line", "category", "comment", "severity".
3. Categories: "bug" (logic error), "performance" (inefficiency), "documentation".
4. Severity: "critical", "major", "minor", "suggestion".
5. Explain *why* the code is wrong and suggest a concrete fix.
6. Return a JSON array. If no correctness issues: return [].
7. Do NOT wrap JSON in markdown fences.
"""

_SECURITY_PROMPT = """You are a security engineer focused on OWASP Top 10 and secure coding.

Analyze the given code diff for:
- Injection (SQL, command, LDAP, SSTI, template injection)
- Authentication and authorization bypass
- Sensitive data exposure (hardcoded secrets, PII, tokens)
- Path traversal and arbitrary file access
- Insecure deserialization
- Cryptographic weaknesses (weak algorithms, hardcoded keys)
- SSRF and unvalidated redirects

Rules:
1. ONLY report security vulnerabilities — skip logic and style (handled separately).
2. For each issue return JSON with keys: "line", "category", "comment", "severity".
3. Use category "security" for all findings. Reference CWE or OWASP category when known.
4. Severity: "critical" (exploitable), "major" (likely vulnerability), "minor", "suggestion".
5. Return a JSON array. If no security issues: return [].
6. Do NOT wrap JSON in markdown fences.
"""

# Semgrep findings are injected as deterministic anchors for the LLM to reason about.
# Inspired by Semgrep AI 2025: structural pattern detection + LLM contextualisation.
_SEMGREP_SUFFIX = """
Additionally, the following static analysis findings from semgrep were detected:

{findings}

For each semgrep finding, confirm whether it is a real vulnerability in context
and explain the impact. If false-positive, say "False positive: <reason>".
"""


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class SemgrepFinding:
    """A single finding from semgrep static analysis.

    Represents one pattern match found by semgrep that gets passed to the
    security pass as a deterministic anchor for LLM reasoning.
    """

    rule_id: str
    message: str
    severity: str
    line: int
    code_snippet: str = ""


@dataclass
class MultiReviewResult:
    """Result of a two-pass multi-model code review.

    Combines the correctness pass and security pass outputs with a
    self-consistency score that measures overall review quality.
    """

    correctness_comments: list[dict] = field(default_factory=list)
    security_comments: list[dict] = field(default_factory=list)
    all_comments: list[dict] = field(default_factory=list)
    consistency_score: float = 0.0
    summary: dict = field(default_factory=dict)


# ── Public API ────────────────────────────────────────────────────────────────


def correctness_pass(
    diff: str,
    model: str = "claude-haiku-4-20250414",
    max_tokens: int = 1024,
) -> list[dict]:
    """Run a correctness-focused review pass on a code diff.

    Первый проход — только баги и логические ошибки. Безопасность
    проверяется отдельно в security_pass() чтобы избежать «ролевого смешения».

    Returns:
        List of dicts with keys: line, category, comment, severity.
        Returns a single-item error list if ANTHROPIC_API_KEY is not set (CI-safe).
    """
    return _llm_call(
        diff=diff,
        system_prompt=_CORRECTNESS_PROMPT,
        model=model,
        max_tokens=max_tokens,
        pass_name="correctness",
    )


def security_pass(
    diff: str,
    model: str = "claude-haiku-4-20250414",
    semgrep_findings: list[SemgrepFinding] | None = None,
    max_tokens: int = 1024,
) -> list[dict]:
    """Run a security-focused review pass on a code diff.

    Второй проход — только уязвимости OWASP Top 10. Семафор-решение
    принимает результаты semgrep как детерминированные якоря, которые LLM
    контекстуализирует и подтверждает (снижение false-positive rate).

    Args:
        diff: Unified diff to review.
        model: Claude model ID.
        semgrep_findings: Optional static analysis findings from semgrep.
        max_tokens: Max response tokens.

    Returns:
        List of dicts with keys: line, category, comment, severity.
    """
    system_prompt = _SECURITY_PROMPT
    if semgrep_findings:
        findings_text = "\n".join(
            f"- [{f.severity.upper()}] {f.rule_id} at line {f.line}: {f.message}"
            + (f"\n  Code: {f.code_snippet}" if f.code_snippet else "")
            for f in semgrep_findings
        )
        system_prompt = system_prompt + _SEMGREP_SUFFIX.format(findings=findings_text)

    return _llm_call(
        diff=diff,
        system_prompt=system_prompt,
        model=model,
        max_tokens=max_tokens,
        pass_name="security",
    )


def self_consistency_score(
    diff: str,  # noqa: ARG001 — reserved for future context-aware scoring
    comments: list[dict],
) -> float:
    """Compute a self-consistency / review quality score (0.0 – 1.0).

    Эвристический скорер, инспирированный CISC (Confidence-Informed
    Self-Consistency). Оценивает качество ревью без дополнительного вызова
    LLM — работает в CI без API-ключа.

    Signals (each contributes a fraction to the final score):
    - Line references: comments point to specific code lines
    - Severity variety: review found issues of different impact levels
    - Comment substance: comments are long enough to be actionable
    - Critical/major findings: at least one non-trivial issue identified
    - Category variety: review covers multiple problem dimensions

    Returns:
        float in [0.0, 1.0] where 1.0 = highly specific, multi-faceted review.
    """
    if not comments:
        return 0.0

    # Error fallback: API key missing or parse failure
    error_markers = ("ANTHROPIC_API_KEY", "Failed to parse", "Error:")
    for comment in comments:
        text = comment.get("comment", "") or ""
        if any(m in text for m in error_markers):
            return 0.0

    n = len(comments)
    score = 0.0

    # 25 pts — at least one comment references a specific line
    line_refs = sum(1 for c in comments if (c.get("line") or "").strip()) / n
    score += 0.25 * min(line_refs * 2, 1.0)

    # 20 pts — severity variety (more = reviewer considered multiple risk levels)
    severity_variety = len({c.get("severity", "") for c in comments}) / 4
    score += 0.20 * min(severity_variety * 2, 1.0)

    # 25 pts — average comment length (substantive explanations > 40 chars)
    avg_len = sum(len(c.get("comment") or "") for c in comments) / n
    score += 0.25 * min(avg_len / 80, 1.0)

    # 15 pts — at least one critical or major finding
    high_sev = sum(1 for c in comments if c.get("severity") in ("critical", "major"))
    score += 0.15 * min(high_sev / max(n, 1), 1.0)

    # 15 pts — category variety (bug + security + performance = well-rounded review)
    cat_variety = len({c.get("category", "") for c in comments}) / 3
    score += 0.15 * min(cat_variety, 1.0)

    return round(min(score, 1.0), 3)


def multi_model_review(
    diff: str,
    correctness_model: str = "claude-haiku-4-20250414",
    security_model: str = "claude-haiku-4-20250414",
    semgrep_findings: list[SemgrepFinding] | None = None,
) -> MultiReviewResult:
    """Orchestrate a two-pass multi-model code review with self-consistency scoring.

    Архитектура двух проходов инспирирована Ericsson 2025 (arxiv 2507.19115):
    два независимых ревьюера устраняют «ролевое смешение» и повышают полноту
    покрытия. Self-consistency score (CISC) позволяет ранжировать ревью по
    качеству без дополнительного LLM-вызова.

    Args:
        diff: Unified diff to review.
        correctness_model: Claude model for correctness pass.
        security_model: Claude model for security pass (can differ for specialisation).
        semgrep_findings: Optional semgrep findings injected into security pass.

    Returns:
        MultiReviewResult with both pass outputs, merged comments, quality score,
        and a summary dict (total, by_severity, verdict).
    """
    # Run passes independently — separate prompts prevent cross-contamination
    correctness = correctness_pass(diff, model=correctness_model)
    security = security_pass(diff, model=security_model, semgrep_findings=semgrep_findings)

    # Deduplicate by (line, category) to avoid surfacing the same issue twice
    seen: set[tuple[str, str]] = set()
    merged: list[dict] = []
    for comment in correctness + security:
        key = (comment.get("line") or "", comment.get("category") or "")
        if key not in seen:
            seen.add(key)
            merged.append(comment)

    score = self_consistency_score(diff, merged)

    # Count by severity for quick triage
    by_severity: dict[str, int] = {}
    for c in merged:
        sev = c.get("severity") or "unknown"
        by_severity[sev] = by_severity.get(sev, 0) + 1

    critical_count = by_severity.get("critical", 0)
    security_count = sum(1 for c in security if c.get("category") == "security")

    # Determine verdict: api_key_missing > fail (critical) > review_required > pass
    has_api_error = any(
        "ANTHROPIC_API_KEY" in (c.get("comment") or "") for c in merged
    )
    if has_api_error:
        verdict = "api_key_missing"
    elif critical_count > 0:
        verdict = "fail"
    elif merged:
        verdict = "review_required"
    else:
        verdict = "pass"

    summary: dict = {
        "total": len(merged),
        "correctness_issues": len(correctness),
        "security_issues": security_count,
        "by_severity": by_severity,
        "consistency_score": score,
        "verdict": verdict,
    }

    return MultiReviewResult(
        correctness_comments=correctness,
        security_comments=security,
        all_comments=merged,
        consistency_score=score,
        summary=summary,
    )


# ── Internal helper ───────────────────────────────────────────────────────────


def _llm_call(
    diff: str,
    system_prompt: str,
    model: str,
    max_tokens: int,
    pass_name: str,
) -> list[dict]:
    """Shared LLM call with graceful API-key fallback.

    Вынесен отдельно, чтобы не дублировать обработку ошибок между
    correctness_pass() и security_pass().
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return [
            {
                "line": "",
                "category": "security" if pass_name == "security" else "documentation",
                "comment": (
                    f"Error: ANTHROPIC_API_KEY not set. "
                    f"Cannot run {pass_name} pass. "
                    "Export ANTHROPIC_API_KEY to enable AI-powered reviews."
                ),
                "severity": "critical",
            }
        ]

    # Lazy import — no hard dependency on anthropic at import time
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Review the following diff for {pass_name} issues:\n\n"
                    f"```diff\n{diff}\n```"
                ),
            }
        ],
    )

    raw = response.content[0].text.strip()
    try:
        comments = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = (
            raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        try:
            comments = json.loads(cleaned)
        except json.JSONDecodeError:
            comments = [
                {
                    "line": "",
                    "category": "bug",
                    "comment": f"[{pass_name}] Failed to parse model output: {raw[:200]}",
                    "severity": "major",
                }
            ]

    if isinstance(comments, dict):
        comments = [comments]

    return comments
