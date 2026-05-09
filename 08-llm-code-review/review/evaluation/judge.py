"""LLM-as-Judge pipeline for evaluating code review quality.

Реализует паттерн LLM-as-Judge (Zheng et al. 2023, MT-Bench) адаптированный
для оценки качества code review. Три метрики:

- faithfulness: доля комментариев, корректно идентифицирующих реальные проблемы
  (нет галлюцинаций о несуществующих багах)
- helpfulness: насколько комментарии actionable — предлагают конкретный фикс
- false_positive_rate: доля комментариев о несуществующих проблемах

Graceful degradation: без ANTHROPIC_API_KEY используется лексическая эвристика
(детерминированная, работает в CI без API-ключей).

Sources:
- Zheng et al. 2023 "Judging LLM-as-a-Judge" (MT-Bench, arxiv 2306.05685)
- LLM Code Review Evaluation 2025 (arxiv 2505.20206)
- Confident AI LLM Eval Guide 2026
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .golden_dataset import GoldenExample

# ── Judge system prompt ────────────────────────────────────────────────

_JUDGE_PROMPT = """\
You are an expert code reviewer evaluating the quality of AI-generated
code review comments.

Given:
1. CODE DIFF: the original change being reviewed
2. REVIEW COMMENTS: AI-generated review findings to evaluate
3. GROUND TRUTH: known real issues in this diff (empty = clean code)

Evaluate the review on three dimensions (each 0.0-1.0):

FAITHFULNESS (are comments accurate and grounded in the diff?):
- 1.0 = all comments correctly identify real issues in the diff
- 0.5 = mix of accurate and inaccurate comments
- 0.0 = comments hallucinate issues not present in the diff

HELPFULNESS (are comments actionable with concrete fixes?):
- 1.0 = each comment explains WHY and suggests a concrete fix
- 0.5 = comments are correct but vague, no fix suggested
- 0.0 = comments are confusing, wrong direction, or unhelpful

FALSE_POSITIVE_RATE (fraction of flagged issues that are NOT real problems):
- 0.0 = all flagged issues are genuine (perfect precision)
- 0.5 = half the comments are false alarms
- 1.0 = all comments are false positives

For clean code (no ground truth issues), any "critical" or "major"
severity comment counts as a false positive.

Return ONLY valid JSON (no markdown fences):
{"faithfulness": 0.0-1.0, "helpfulness": 0.0-1.0,
 "false_positive_rate": 0.0-1.0, "reasoning": "brief explanation"}
"""

# ── Dataclasses ─────────────────────────────────────────────────

# Action words that indicate a comment suggests a concrete fix
_HELPFULNESS_SIGNALS = frozenset(
    {
        "should",
        "fix",
        "change",
        "replace",
        "use",
        "avoid",
        "remove",
        "add",
        "consider",
        "refactor",
        "switch",
        "instead",
        "recommend",
        "suggest",
        "prefer",
    }
)

# Error placeholder text injected when API key is missing
_API_ERROR_MARKERS = ("api_key", "anthropic_api_key", "api key not set", "api_key_missing")


@dataclass
class JudgeVerdict:
    """Результат оценки одного code review по трём метрикам.

    Result of LLM-as-Judge evaluation for a single code review.
    """

    faithfulness: float
    helpfulness: float
    false_positive_rate: float
    overall_score: float
    reasoning: str
    api_key_used: bool
    example_id: str = ""

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "faithfulness": round(self.faithfulness, 3),
            "helpfulness": round(self.helpfulness, 3),
            "false_positive_rate": round(self.false_positive_rate, 3),
            "overall_score": round(self.overall_score, 3),
            "reasoning": self.reasoning,
            "api_key_used": self.api_key_used,
        }


@dataclass
class RegressionResult:
    """Результат регрессионного тестирования по всему золотому датасету.

    Aggregate evaluation across the full golden dataset. Tracks whether
    average quality dropped below the acceptance threshold.
    """

    n_examples: int
    avg_faithfulness: float
    avg_helpfulness: float
    avg_false_positive_rate: float
    avg_overall_score: float
    by_domain: dict[str, dict]
    verdicts: list[JudgeVerdict] = field(default_factory=list)
    passed: bool = True
    threshold: float = 0.5

    def to_dict(self) -> dict:
        return {
            "n_examples": self.n_examples,
            "avg_faithfulness": round(self.avg_faithfulness, 3),
            "avg_helpfulness": round(self.avg_helpfulness, 3),
            "avg_false_positive_rate": round(self.avg_false_positive_rate, 3),
            "avg_overall_score": round(self.avg_overall_score, 3),
            "by_domain": self.by_domain,
            "passed": self.passed,
            "threshold": self.threshold,
        }


# ── Internal helpers ────────────────────────────────────────────────


def _compute_overall(faithfulness: float, helpfulness: float, fpr: float) -> float:
    """Weighted aggregate: faithfulness (40%) + helpfulness (30%) + precision (30%)."""
    return 0.4 * faithfulness + 0.3 * helpfulness + 0.3 * (1.0 - fpr)


def _is_api_error_comment(comment: dict) -> bool:
    text = comment.get("comment", "").lower()
    return any(marker in text for marker in _API_ERROR_MARKERS)


def _compute_helpfulness(comments: list[dict]) -> float:
    """Fraction of comments containing action-oriented language."""
    if not comments:
        return 0.0
    helpful = sum(
        1 for c in comments if any(w in c.get("comment", "").lower() for w in _HELPFULNESS_SIGNALS)
    )
    return helpful / len(comments)


# ── Lexical judge (CI-safe, no API key) ────────────────────────────────────


def _lexical_judge(example: GoldenExample, review_comments: list[dict]) -> JudgeVerdict:
    """Детерминированная лексическая оценка качества ревью без API-ключа.

    Lexical fallback judge using keyword overlap with ground truth.
    Used in CI where ANTHROPIC_API_KEY is not available.
    """
    real_comments = [c for c in review_comments if not _is_api_error_comment(c)]

    if not real_comments:
        if example.is_clean:
            # No comments on clean code — correct behaviour
            return JudgeVerdict(
                faithfulness=1.0,
                helpfulness=1.0,
                false_positive_rate=0.0,
                overall_score=_compute_overall(1.0, 1.0, 0.0),
                reasoning="No comments on clean code — correct.",
                api_key_used=False,
                example_id=example.id,
            )
        else:
            # Missed all real issues — bad recall but FPR is 0
            return JudgeVerdict(
                faithfulness=0.0,
                helpfulness=0.0,
                false_positive_rate=0.0,
                overall_score=_compute_overall(0.0, 0.0, 0.0),
                reasoning="No real comments generated despite known issues.",
                api_key_used=False,
                example_id=example.id,
            )

    helpfulness = _compute_helpfulness(real_comments)

    if example.is_clean:
        high_severity = {"critical", "major"}
        fp_count = sum(1 for c in real_comments if c.get("severity", "") in high_severity)
        fpr = fp_count / len(real_comments)
        faithfulness = 1.0 - fpr
        reasoning = (
            f"Clean diff: {fp_count}/{len(real_comments)} "
            "high-severity comments are false positives."
        )
    else:
        # Keyword overlap: comment covers the issue if it mentions any ground truth keyword
        all_keywords: set[str] = set()
        for issue in example.ground_truth_issues:
            for kw in issue.get("keywords", []):
                all_keywords.add(kw.lower())

        matched = sum(
            1
            for c in real_comments
            if any(kw in c.get("comment", "").lower() for kw in all_keywords)
        )
        faithfulness = matched / len(real_comments) if real_comments else 0.0
        fpr = 1.0 - faithfulness
        sample_kws = sorted(all_keywords)[:4]
        reasoning = (
            f"{matched}/{len(real_comments)} comments matched ground truth keywords "
            f"(e.g. {sample_kws})."
        )

    return JudgeVerdict(
        faithfulness=faithfulness,
        helpfulness=helpfulness,
        false_positive_rate=fpr,
        overall_score=_compute_overall(faithfulness, helpfulness, fpr),
        reasoning=reasoning,
        api_key_used=False,
        example_id=example.id,
    )


# ── LLM judge ─────────────────────────────────────────────────────────────


def _llm_judge(
    example: GoldenExample,
    review_comments: list[dict],
    model: str,
) -> JudgeVerdict:
    """Оценка качества ревью с помощью Claude в роли судьи.

    Uses Claude as an independent judge to score faithfulness, helpfulness,
    and false_positive_rate. Falls back to lexical judge on any error.
    """
    try:
        import anthropic
    except ImportError:
        return _lexical_judge(example, review_comments)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _lexical_judge(example, review_comments)

    ground_truth_text = (
        json.dumps(example.ground_truth_issues, indent=2)
        if example.ground_truth_issues
        else "[] (clean code — no real issues)"
    )
    comments_text = (
        json.dumps(review_comments, indent=2) if review_comments else "[] (no comments generated)"
    )

    user_content = (
        f"CODE DIFF:\n{example.diff}\n\n"
        f"REVIEW COMMENTS (to evaluate):\n{comments_text}\n\n"
        f"GROUND TRUTH ISSUES:\n{ground_truth_text}\n\n"
        f"Domain: {example.domain}\n"
        f"Is clean code: {example.is_clean}\n"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=_JUDGE_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if the model wraps the JSON
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw
        verdict_dict = json.loads(raw)
        faithfulness = float(verdict_dict.get("faithfulness", 0.5))
        helpfulness = float(verdict_dict.get("helpfulness", 0.5))
        fpr = float(verdict_dict.get("false_positive_rate", 0.5))
        reasoning = str(verdict_dict.get("reasoning", ""))
        # Clamp to [0, 1]
        faithfulness = max(0.0, min(1.0, faithfulness))
        helpfulness = max(0.0, min(1.0, helpfulness))
        fpr = max(0.0, min(1.0, fpr))
        return JudgeVerdict(
            faithfulness=faithfulness,
            helpfulness=helpfulness,
            false_positive_rate=fpr,
            overall_score=_compute_overall(faithfulness, helpfulness, fpr),
            reasoning=reasoning,
            api_key_used=True,
            example_id=example.id,
        )
    except Exception as exc:
        verdict = _lexical_judge(example, review_comments)
        verdict.reasoning = f"LLM judge error ({exc!s}); lexical fallback. " + verdict.reasoning
        return verdict


# ── Public API ────────────────────────────────────────────────────────────


def evaluate_review(
    example: GoldenExample,
    review_comments: list[dict],
    model: str = "claude-haiku-4-20250414",
) -> JudgeVerdict:
    """Оценить одно ревью с помощью LLM-as-Judge или лексического fallback.

    Evaluate a single code review. Uses the LLM judge when ANTHROPIC_API_KEY
    is set, otherwise falls back to deterministic lexical scoring (CI-safe).

    Args:
        example: annotated golden example with ground truth issues
        review_comments: list of comment dicts from multi_model_review()
        model: Claude model to use as the judge

    Returns:
        JudgeVerdict with faithfulness, helpfulness, false_positive_rate
    """
    return _llm_judge(example, review_comments, model)


def run_regression_suite(
    model: str = "claude-haiku-4-20250414",
    judge_model: str = "claude-haiku-4-20250414",
    threshold: float = 0.5,
    use_lexical: bool = False,
) -> RegressionResult:
    """Регрессионный тест: запустить LLM-as-Judge по всему золотому датасету.

    Run LLM-as-Judge evaluation across all 20 golden examples. A drop in
    avg_overall_score below ``threshold`` signals a quality regression.

    Args:
        model: Claude model used to generate the reviews under test
        judge_model: Claude model used to judge review quality
        threshold: minimum acceptable avg_overall_score to pass
        use_lexical: force lexical judge (skip LLM calls, always CI-safe)

    Returns:
        RegressionResult with per-domain breakdown and pass/fail verdict
    """
    from ..models.multi_review import multi_model_review
    from .golden_dataset import get_golden_dataset

    examples = get_golden_dataset()
    verdicts: list[JudgeVerdict] = []

    for example in examples:
        result = multi_model_review(example.diff, correctness_model=model, security_model=model)
        review_comments = result.all_comments

        if use_lexical:
            verdict = _lexical_judge(example, review_comments)
        else:
            verdict = evaluate_review(example, review_comments, model=judge_model)
        verdict.example_id = example.id
        verdicts.append(verdict)

    n = len(verdicts)
    avg_faithfulness = sum(v.faithfulness for v in verdicts) / n
    avg_helpfulness = sum(v.helpfulness for v in verdicts) / n
    avg_fpr = sum(v.false_positive_rate for v in verdicts) / n
    avg_overall = sum(v.overall_score for v in verdicts) / n

    # Per-domain aggregation
    domains: dict[str, list[JudgeVerdict]] = {}
    for ex, v in zip(examples, verdicts, strict=True):
        domains.setdefault(ex.domain, []).append(v)

    by_domain: dict[str, dict] = {}
    for domain, dverdicts in domains.items():
        nd = len(dverdicts)
        by_domain[domain] = {
            "n": nd,
            "avg_faithfulness": round(sum(v.faithfulness for v in dverdicts) / nd, 3),
            "avg_helpfulness": round(sum(v.helpfulness for v in dverdicts) / nd, 3),
            "avg_false_positive_rate": round(
                sum(v.false_positive_rate for v in dverdicts) / nd, 3
            ),
            "avg_overall_score": round(sum(v.overall_score for v in dverdicts) / nd, 3),
        }

    return RegressionResult(
        n_examples=n,
        avg_faithfulness=avg_faithfulness,
        avg_helpfulness=avg_helpfulness,
        avg_false_positive_rate=avg_fpr,
        avg_overall_score=avg_overall,
        by_domain=by_domain,
        verdicts=verdicts,
        passed=avg_overall >= threshold,
        threshold=threshold,
    )
