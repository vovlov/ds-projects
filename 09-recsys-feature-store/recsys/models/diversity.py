"""MMR Diversity Reranking for Recommendation Systems.

Implements Maximal Marginal Relevance (Carbonell & Goldstein 1998, SIGIR)
to balance relevance and diversity in recommendation lists.

MMR score = λ · sim(item, query) - (1-λ) · max_{s∈S} sim(item, s)

Greedy selection: at each step, pick the candidate that maximises the
trade-off between being relevant to the query and dissimilar to already
selected items.  λ=1.0 → pure relevance, λ=0.0 → pure diversity.

Business rationale: pure relevance leads to near-identical items
(filter bubble), hurting serendipity and long-term user satisfaction.
MMR provides a tunable knob between the two extremes.

Sources:
  Carbonell & Goldstein 1998, SIGIR (original MMR paper)
  Kunaver & Požrl 2017 (ILS survey, Information Processing & Management)
  Google RecSys 2024: diversity-aware retrieval for YouTube
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DiversityConfig:
    """Configuration for MMR diversity reranker."""

    lambda_param: float = 0.5
    """Trade-off: 1.0 = pure relevance, 0.0 = pure diversity."""

    n_items: int = 10
    """Number of items to return."""

    embedding_dim: int = 8
    """Dimension of item content embeddings."""


@dataclass
class DiverseItem:
    """Single item in a diversity-reranked recommendation list."""

    item_id: int
    relevance_score: float
    """Normalised relevance score in [0, 1]."""

    diversity_contribution: float
    """Marginal diversity added: 1 - max_cosine_sim to previously selected items."""

    mmr_score: float
    """Combined MMR objective value."""

    rank: int
    """1-based position in the final list."""


@dataclass
class DiversityMetrics:
    """Post-hoc diversity metrics for the selected recommendation list."""

    intra_list_diversity: float
    """Mean pairwise cosine distance (higher = more diverse)."""

    coverage: float
    """Fraction of distinct embedding quadrants covered — category proxy."""

    novelty: float
    """Mean distance from the most popular candidate item."""

    effective_diversity: float
    """Shannon entropy of pairwise-distance distribution — penalises concentration."""


@dataclass
class DiversityResult:
    """Full result of an MMR diversity-reranking run."""

    items: list[DiverseItem]
    metrics: DiversityMetrics
    lambda_param: float
    n_candidates: int


class MMRDiversifier:
    """Maximal Marginal Relevance reranker (Carbonell & Goldstein 1998).

    Usage::

        diversifier = MMRDiversifier(DiversityConfig(lambda_param=0.6))
        embs = diversifier.build_item_embeddings([1, 2, 3], categories=["books", "books", "sports"])
        result = diversifier.rerank([1, 2, 3], [0.9, 0.85, 0.8], embs)
    """

    def __init__(self, config: DiversityConfig | None = None) -> None:
        self.config = config or DiversityConfig()

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    def rerank(
        self,
        candidate_ids: list[int],
        relevance_scores: list[float],
        item_embeddings: dict[int, np.ndarray],
        lambda_param: float | None = None,
        n_items: int | None = None,
    ) -> DiversityResult:
        """Greedy MMR selection from a candidate set.

        Args:
            candidate_ids: Ordered list of candidate item IDs.
            relevance_scores: Corresponding relevance scores (arbitrary scale, will be normalised).
            item_embeddings: Map from item_id to content embedding vector.
            lambda_param: Override config.lambda_param (0 = diversity, 1 = relevance).
            n_items: Override config.n_items.

        Returns:
            DiversityResult with reranked items and aggregate diversity metrics.
        """
        lam = lambda_param if lambda_param is not None else self.config.lambda_param
        n = min(n_items if n_items is not None else self.config.n_items, len(candidate_ids))

        if not candidate_ids:
            return DiversityResult(
                items=[],
                metrics=DiversityMetrics(0.0, 0.0, 0.0, 0.0),
                lambda_param=lam,
                n_candidates=0,
            )

        if len(candidate_ids) != len(relevance_scores):
            raise ValueError("candidate_ids and relevance_scores must have equal length")

        # Normalise relevance to [0, 1] for stable MMR objective
        norm_scores = self._normalise(relevance_scores)
        score_map: dict[int, float] = dict(zip(candidate_ids, norm_scores))

        dim = self.config.embedding_dim
        remaining = list(candidate_ids)
        selected: list[int] = []
        selected_embs: list[np.ndarray] = []
        diverse_items: list[DiverseItem] = []

        for rank in range(1, n + 1):
            if not remaining:
                break

            best_id: int | None = None
            best_mmr = float("-inf")
            best_div = 0.0

            for item_id in remaining:
                rel = score_map[item_id]
                emb = item_embeddings.get(item_id, np.zeros(dim))

                if selected_embs:
                    max_sim = max(
                        self._cosine_sim(emb, sel_emb) for sel_emb in selected_embs
                    )
                else:
                    # первый элемент — нет конкурентов за diversity
                    max_sim = 0.0

                mmr = lam * rel - (1 - lam) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_id = item_id
                    best_div = 1.0 - max_sim

            if best_id is None:
                break

            sel_emb = item_embeddings.get(best_id, np.zeros(dim))
            selected.append(best_id)
            selected_embs.append(sel_emb)
            remaining.remove(best_id)

            diverse_items.append(
                DiverseItem(
                    item_id=best_id,
                    relevance_score=round(score_map[best_id], 4),
                    diversity_contribution=round(best_div, 4),
                    mmr_score=round(best_mmr, 4),
                    rank=rank,
                )
            )

        metrics = self._compute_metrics(selected_embs, candidate_ids, item_embeddings)

        return DiversityResult(
            items=diverse_items,
            metrics=metrics,
            lambda_param=lam,
            n_candidates=len(candidate_ids),
        )

    # ------------------------------------------------------------------
    # Helper: build content embeddings from item metadata
    # ------------------------------------------------------------------

    def build_item_embeddings(
        self,
        item_ids: list[int],
        categories: list[str] | None = None,
        price_tiers: list[str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[int, np.ndarray]:
        """Build lightweight content embeddings from item metadata.

        Produces L2-normalised embeddings where the first half encodes category
        (one-hot) and the second half encodes price tier (one-hot).  Small
        Gaussian noise ensures identical-metadata items are not exactly the same.

        Args:
            item_ids: List of product IDs.
            categories: Optional category label per item (same order as item_ids).
            price_tiers: Optional price tier per item ("low" / "medium" / "high").
            rng: Optional seeded RNG for reproducibility.

        Returns:
            Dict mapping item_id → L2-normalised embedding (shape: (embedding_dim,)).
        """
        rng = rng or np.random.default_rng(42)
        dim = self.config.embedding_dim

        CAT_MAP = {
            "electronics": 0,
            "books": 1,
            "clothing": 2,
            "food": 3,
            "sports": 4,
        }
        PT_MAP = {"low": 0, "medium": 1, "high": 2}

        half = dim // 2
        embeddings: dict[int, np.ndarray] = {}

        for idx, item_id in enumerate(item_ids):
            emb = rng.normal(0.0, 0.1, dim)

            if categories and idx < len(categories):
                cat_pos = CAT_MAP.get(categories[idx], -1)
                if 0 <= cat_pos < half:
                    emb[cat_pos] += 1.0

            if price_tiers and idx < len(price_tiers):
                pt_pos = PT_MAP.get(price_tiers[idx], -1)
                slot = half + pt_pos
                if 0 <= pt_pos < dim - half and slot < dim:
                    emb[slot] += 1.0

            norm = np.linalg.norm(emb)
            if norm > 1e-10:
                emb = emb / norm
            embeddings[item_id] = emb

        return embeddings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity; returns 0 for zero vectors."""
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))

    @staticmethod
    def _normalise(scores: list[float]) -> list[float]:
        """Min-max normalise to [0, 1]; returns uniform 1.0 when all equal."""
        lo, hi = min(scores), max(scores)
        span = hi - lo
        if span < 1e-10:
            return [1.0] * len(scores)
        return [(s - lo) / span for s in scores]

    def _compute_metrics(
        self,
        selected_embs: list[np.ndarray],
        all_candidates: list[int],
        item_embeddings: dict[int, np.ndarray],
    ) -> DiversityMetrics:
        """Compute post-hoc diversity metrics for the selected list."""
        # 1. Intra-list diversity: mean pairwise cosine distance
        if len(selected_embs) < 2:
            ild = 0.0
        else:
            dists = [
                1.0 - self._cosine_sim(selected_embs[i], selected_embs[j])
                for i in range(len(selected_embs))
                for j in range(i + 1, len(selected_embs))
            ]
            ild = float(np.mean(dists))

        # 2. Coverage: fraction of unique "quadrant signatures" (cheap category proxy)
        if selected_embs:
            half = max(len(selected_embs[0]) // 2, 1)
            sigs: set[tuple[int, ...]] = set()
            for emb in selected_embs:
                sig = (
                    int(np.sign(np.sum(emb[:half]))),
                    int(np.sign(np.sum(emb[half:]))),
                )
                sigs.add(sig)
            coverage = len(sigs) / max(1, len(selected_embs))
        else:
            coverage = 0.0

        # 3. Novelty: mean distance from the most popular (first) candidate
        dim = self.config.embedding_dim
        if all_candidates and selected_embs:
            pop_emb = item_embeddings.get(all_candidates[0], np.zeros(dim))
            novelty = float(
                np.mean([1.0 - self._cosine_sim(e, pop_emb) for e in selected_embs])
            )
        else:
            novelty = 0.0

        # 4. Effective diversity: Shannon entropy over histogram of pairwise distances
        if len(selected_embs) >= 2:
            dists_arr = np.array(
                [
                    1.0 - self._cosine_sim(selected_embs[i], selected_embs[j])
                    for i in range(len(selected_embs))
                    for j in range(i + 1, len(selected_embs))
                ]
            )
            counts, _ = np.histogram(dists_arr, bins=5, range=(0.0, 1.0))
            total = counts.sum()
            if total > 0:
                probs = counts / total
                eff_div = float(-sum(p * math.log2(p) for p in probs if p > 0))
            else:
                eff_div = 0.0
        else:
            eff_div = 0.0

        return DiversityMetrics(
            intra_list_diversity=round(max(0.0, ild), 4),
            coverage=round(min(1.0, max(0.0, coverage)), 4),
            novelty=round(max(0.0, novelty), 4),
            effective_diversity=round(max(0.0, eff_div), 4),
        )
