"""
Golden query set для оффлайн-оценки RAG-пайплайна.

Queries align with the actual document corpus in data/documents/:
- data_governance.txt   — data classification, retention, GDPR, access control
- engineering_standards.txt — code review, testing, branching, deployment
- sample_policy.txt     — remote work eligibility, schedule, equipment, expenses
- sample_onboarding.txt — onboarding schedule, tools, key contacts
- product_faq.txt       — product Q&A, passwords, billing

Каждый GoldenQuery содержит:
- query:            вопрос пользователя (natural language)
- relevant_keywords: ключевые слова из релевантных чанков (keyword overlap proxy)
- category:         тематика для группового анализа
- difficulty:       easy/medium/hard — насколько прямое ключевое совпадение

Принцип оценки релевантности:
    chunk_is_relevant() — keyword overlap: чанк считается релевантным если
    содержит ≥ min_keywords ключевых слов из relevant_keywords.
    CI-safe, воспроизводимо, не требует labeled judgments.

Источники:
    Manning et al. 2008 "Introduction to Information Retrieval" §8 (offline eval)
    Thakur et al. 2021 BEIR Benchmark (arxiv:2104.08663) — golden query design
    Voorhees & Harman 2005 NIST/TREC — reproducible IR evaluation methodology
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenQuery:
    """Аннотированный запрос для оффлайн-оценки retrieval."""

    query: str
    relevant_keywords: tuple[str, ...]  # tuple для frozen dataclass
    category: str  # governance / engineering / remote / onboarding / product
    difficulty: str  # easy / medium / hard

    @property
    def keyword_set(self) -> frozenset[str]:
        """Нормализованные ключевые слова для сравнения."""
        return frozenset(kw.lower() for kw in self.relevant_keywords)


# 20 golden queries, покрывают все 5 документов и 3 уровня сложности
# easy:   ключевые слова буквально присутствуют в одном чанке
# medium: синонимы, перефразировки или несколько чанков нужны
# hard:   multi-hop, нет прямого упоминания, нужен контекст
GOLDEN_QUERIES: tuple[GoldenQuery, ...] = (
    # ── DATA GOVERNANCE ───────────────────────────────────────────────────────
    GoldenQuery(
        query="What are the four data classification categories?",
        relevant_keywords=("classification", "public", "internal", "confidential", "restricted"),
        category="governance",
        difficulty="easy",
    ),
    GoldenQuery(
        query="How long must confidential data be retained before deletion?",
        relevant_keywords=("confidential", "retention", "years", "deletion", "secure"),
        category="governance",
        difficulty="easy",
    ),
    GoldenQuery(
        query="What are the GDPR requirements for collecting personal data?",
        relevant_keywords=("GDPR", "personal data", "consent", "right", "access"),
        category="governance",
        difficulty="medium",
    ),
    GoldenQuery(
        query="How should a data breach be reported and what are the timelines?",
        relevant_keywords=("breach", "report", "72 hours", "GDPR", "notify"),
        category="governance",
        difficulty="medium",
    ),
    GoldenQuery(
        query="What access control principle applies to company data?",
        relevant_keywords=("access", "least privilege", "data owner", "review", "approved"),
        category="governance",
        difficulty="hard",
    ),
    # ── ENGINEERING STANDARDS ─────────────────────────────────────────────────
    GoldenQuery(
        query="What is the minimum code coverage requirement for new code?",
        relevant_keywords=("coverage", "80%", "unit test", "new code", "testing"),
        category="engineering",
        difficulty="easy",
    ),
    GoldenQuery(
        query="How quickly must code reviews be completed?",
        relevant_keywords=("review", "24 hours", "submission", "completed", "code"),
        category="engineering",
        difficulty="easy",
    ),
    GoldenQuery(
        query="What branch naming convention should I follow?",
        relevant_keywords=("branch", "naming", "feature", "bugfix", "hotfix"),
        category="engineering",
        difficulty="easy",
    ),
    GoldenQuery(
        query="What is the deployment process and who needs to approve production deploys?",
        relevant_keywords=("deployment", "staging", "production", "approval", "blue-green"),
        category="engineering",
        difficulty="medium",
    ),
    GoldenQuery(
        query="What severity levels exist for incidents and what is the P1 response?",
        relevant_keywords=("severity", "P1", "incident", "post-mortem", "response"),
        category="engineering",
        difficulty="medium",
    ),
    # ── REMOTE WORK POLICY ────────────────────────────────────────────────────
    GoldenQuery(
        query="Who is eligible to work remotely?",
        relevant_keywords=("eligible", "remote", "probationary", "full-time", "90 days"),
        category="remote",
        difficulty="easy",
    ),
    GoldenQuery(
        query="What are the core working hours for remote employees?",
        relevant_keywords=("core", "10:00 AM", "4:00 PM", "hours", "time zone"),
        category="remote",
        difficulty="easy",
    ),
    GoldenQuery(
        query="What equipment does the company provide for remote workers?",
        relevant_keywords=("laptop", "monitor", "keyboard", "equipment", "provides"),
        category="remote",
        difficulty="easy",
    ),
    GoldenQuery(
        query="How much does the company reimburse for internet expenses when working from home?",
        relevant_keywords=("reimburse", "internet", "$50", "month", "expenses"),
        category="remote",
        difficulty="medium",
    ),
    GoldenQuery(
        query="What security requirements apply when working remotely?",
        relevant_keywords=("VPN", "two-factor", "authentication", "security", "company data"),
        category="remote",
        difficulty="medium",
    ),
    # ── ONBOARDING ────────────────────────────────────────────────────────────
    GoldenQuery(
        query="What should I do on my first day at the company?",
        relevant_keywords=("day 1", "manager", "HR paperwork", "accounts", "team"),
        category="onboarding",
        difficulty="easy",
    ),
    GoldenQuery(
        query="What tools and systems does the company use for project management?",
        relevant_keywords=("Jira", "GitHub", "Slack", "Confluence", "tools"),
        category="onboarding",
        difficulty="easy",
    ),
    GoldenQuery(
        query="When will I get my first ticket and what kind of task is it?",
        relevant_keywords=("ticket", "good-first-issue", "backlog", "pull request", "sprint"),
        category="onboarding",
        difficulty="medium",
    ),
    # ── PRODUCT FAQ ───────────────────────────────────────────────────────────
    GoldenQuery(
        query="How do I reset my password if I forgot it?",
        relevant_keywords=("password", "reset", "forgot", "login", "email"),
        category="product",
        difficulty="easy",
    ),
    GoldenQuery(
        query="What payment methods are supported for enterprise plans?",
        relevant_keywords=("payment", "enterprise", "billing", "invoice", "NET-30"),
        category="product",
        difficulty="medium",
    ),
)


def get_golden_queries(
    category: str | None = None,
    difficulty: str | None = None,
) -> list[GoldenQuery]:
    """Получить golden queries с опциональной фильтрацией.

    Args:
        category: Фильтр (governance/engineering/remote/onboarding/product).
        difficulty: Фильтр (easy/medium/hard).

    Returns:
        Отфильтрованный список GoldenQuery.
    """
    result = list(GOLDEN_QUERIES)
    if category is not None:
        result = [q for q in result if q.category == category]
    if difficulty is not None:
        result = [q for q in result if q.difficulty == difficulty]
    return result


def chunk_is_relevant(chunk_text: str, query: GoldenQuery, min_keywords: int = 2) -> bool:
    """Определить, является ли чанк релевантным для данного запроса.

    Keyword overlap proxy — CI-safe без labeled relevance judgments.
    Чанк считается релевантным если содержит ≥ min_keywords ключевых слов.
    Multi-word keywords ("core hours", "least privilege") ищутся как подстрока.

    Args:
        chunk_text: Текст чанка.
        query: Golden query с relevant_keywords.
        min_keywords: Порог совпадений для признания релевантным.

    Returns:
        True если ≥ min_keywords ключевых слов найдено в тексте.
    """
    text_lower = chunk_text.lower()
    hits = sum(1 for kw in query.relevant_keywords if kw.lower() in text_lower)
    return hits >= min_keywords


def find_relevant_chunks(
    chunks: list[dict],
    query: GoldenQuery,
    min_keywords: int = 2,
) -> set[str]:
    """Найти множество релевантных chunk-IDs для данного запроса.

    Строит ground truth при оффлайн-оценке без labeled relevance judgments.
    chunk_id = "chunk_{i}" — стандарт индексации в RAG Enterprise (store.py).

    Args:
        chunks: Список чанков [{"text": ..., "metadata": ...}].
        query: Golden query с relevant_keywords.
        min_keywords: Порог для chunk_is_relevant().

    Returns:
        Set of relevant chunk IDs, e.g. {"chunk_0", "chunk_3"}.
    """
    relevant: set[str] = set()
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if chunk_is_relevant(text, query, min_keywords):
            relevant.add(f"chunk_{i}")
    return relevant


def get_dataset_stats() -> dict:
    """Статистика golden dataset по категориям и сложностям."""
    from collections import Counter

    cats = Counter(q.category for q in GOLDEN_QUERIES)
    diffs = Counter(q.difficulty for q in GOLDEN_QUERIES)
    return {
        "n_queries": len(GOLDEN_QUERIES),
        "categories": dict(cats),
        "difficulties": dict(diffs),
        "categories_list": sorted(cats.keys()),
    }
