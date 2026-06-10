"""Comparable Properties Analysis (Market Comps) для AVM.

Поиск K ближайших аналогов — ключевой компонент AVM (Automated Valuation Model).
Используется Zillow Zestimate, CoreLogic, ЦИАН: «оценка подтверждается реальными
продажами похожих объектов в том же районе».

Техника: взвешенная нормализованная евклидова дистанция.
Веса отражают важность каждого признака для московского рынка недвижимости:
район объясняет ~60% вариации цены → самый большой вес.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CompsConfig:
    """Конфигурация поиска аналогов."""

    n_comps: int = 5
    market_at_threshold_pct: float = 5.0
    # Веса признаков: район → самый важный, санузлы → наименее важный
    feature_weights: dict[str, float] = field(
        default_factory=lambda: {
            "sqft": 2.0,
            "year_built": 1.0,
            "bedrooms": 0.5,
            "neighborhood": 3.0,
            "condition": 1.5,
        }
    )


@dataclass
class ComparableProperty:
    """Один аналог из базы данных объектов."""

    sqft: int
    bedrooms: int
    year_built: int
    neighborhood: str
    condition: str
    price: int
    price_per_sqft: float
    similarity_score: float
    distance: float

    def to_dict(self) -> dict:
        return {
            "sqft": self.sqft,
            "bedrooms": self.bedrooms,
            "year_built": self.year_built,
            "neighborhood": self.neighborhood,
            "condition": self.condition,
            "price": self.price,
            "price_per_sqft": round(self.price_per_sqft, 0),
            "similarity_score": round(self.similarity_score, 4),
            "distance": round(self.distance, 4),
        }


@dataclass
class CompsResult:
    """Результат поиска аналогов с рыночными метриками."""

    comparables: list[ComparableProperty]
    subject_price: int | None
    median_comp_price: int
    mean_comp_price: int
    price_deviation_pct: float | None
    market_position: str | None
    n_comparables: int

    def to_dict(self) -> dict:
        return {
            "comparables": [c.to_dict() for c in self.comparables],
            "subject_price": self.subject_price,
            "median_comp_price": self.median_comp_price,
            "mean_comp_price": self.mean_comp_price,
            "price_deviation_pct": (
                round(self.price_deviation_pct, 2) if self.price_deviation_pct is not None else None
            ),
            "market_position": self.market_position,
            "n_comparables": self.n_comparables,
        }


class ComparableSearch:
    """Поиск аналогов через взвешенную нормализованную евклидову дистанцию.

    Кодирование признаков:
    - Числовые (sqft, year_built, bedrooms): min-max нормализация по базе
    - Neighborhood: ценовой коэффициент из NEIGHBORHOODS (чем ближе коэффициент, тем похожее)
    - Condition: ценовой коэффициент из CONDITION_MAP
    Это семантически точнее алфавитного кодирования: «Раменки» (1.2) ближе к «Строгино» (0.95),
    чем к «Арбату» (1.9), и расстояние пропорционально разнице рыночного уровня.
    """

    def __init__(self, config: CompsConfig | None = None) -> None:
        self.config = config or CompsConfig()
        self._database: list[dict] = []
        self._is_fitted = False
        self._num_stats: dict[str, tuple[float, float]] = {}
        self._nbh_coeffs: dict[str, float] = {}
        self._cond_coeffs: dict[str, float] = {}

    def fit(self, data: list[dict]) -> "ComparableSearch":
        """Загрузить базу данных и вычислить статистики для нормализации."""
        if not data:
            raise ValueError("Database must have at least 1 property")

        self._database = list(data)

        for feat in ("sqft", "year_built", "bedrooms"):
            vals = [float(d[feat]) for d in data]
            mn, mx = min(vals), max(vals)
            self._num_stats[feat] = (mn, mx if mx > mn else mn + 1.0)

        # Ценовые коэффициенты районов и состояний — из load.py (можно передать или импортировать)
        try:
            from ..data.load import CONDITION_MAP, NEIGHBORHOODS

            self._nbh_coeffs = dict(NEIGHBORHOODS)
            self._cond_coeffs = dict(CONDITION_MAP)
        except ImportError:
            # Fallback: собираем уникальные значения и нормируем позицией
            nbh_vals = sorted(set(d["neighborhood"] for d in data))
            self._nbh_coeffs = {n: float(i) / max(len(nbh_vals) - 1, 1) for i, n in enumerate(nbh_vals)}
            cond_vals = sorted(set(d["condition"] for d in data))
            self._cond_coeffs = {c: float(i) / max(len(cond_vals) - 1, 1) for i, c in enumerate(cond_vals)}

        self._is_fitted = True
        return self

    def _encode(self, prop: dict) -> np.ndarray:
        """Преобразовать объект в нормализованный вектор признаков."""
        w = self.config.feature_weights
        vec: list[float] = []

        for feat in ("sqft", "year_built", "bedrooms"):
            mn, mx = self._num_stats.get(feat, (0.0, 1.0))
            val = float(prop.get(feat, mn))
            vec.append(w.get(feat, 1.0) * (val - mn) / (mx - mn))

        # Neighborhood через ценовой коэффициент
        nbh_vals = list(self._nbh_coeffs.values())
        nbh_min = min(nbh_vals) if nbh_vals else 0.0
        nbh_max = max(nbh_vals) if nbh_vals else 1.0
        nbh_range = nbh_max - nbh_min if nbh_max > nbh_min else 1.0
        nbh_coeff = self._nbh_coeffs.get(prop.get("neighborhood", ""), nbh_min)
        vec.append(w.get("neighborhood", 3.0) * (nbh_coeff - nbh_min) / nbh_range)

        # Condition через ценовой коэффициент
        cond_vals = list(self._cond_coeffs.values())
        cond_min = min(cond_vals) if cond_vals else 0.0
        cond_max = max(cond_vals) if cond_vals else 1.0
        cond_range = cond_max - cond_min if cond_max > cond_min else 1.0
        cond_coeff = self._cond_coeffs.get(prop.get("condition", ""), cond_min)
        vec.append(w.get("condition", 1.5) * (cond_coeff - cond_min) / cond_range)

        return np.array(vec, dtype=np.float64)

    def find_comps(self, subject: dict, n_comps: int | None = None) -> CompsResult:
        """Найти K наиболее похожих объектов для subject.

        Args:
            subject: словарь с признаками оцениваемого объекта (price опциональна)
            n_comps: число аналогов (переопределяет config.n_comps)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before find_comps()")

        k = min(n_comps or self.config.n_comps, len(self._database))
        subject_vec = self._encode(subject)

        dists = []
        for prop in self._database:
            d = float(np.linalg.norm(subject_vec - self._encode(prop)))
            dists.append((d, prop))

        dists.sort(key=lambda x: x[0])
        top_k = dists[:k]

        # Similarity = exp(-dist): 0 дистанция → 1.0, большая → ≈ 0
        comparables = []
        for dist, prop in top_k:
            sim = float(np.exp(-dist))
            price = int(prop["price"])
            sqft = int(prop["sqft"])
            comparables.append(
                ComparableProperty(
                    sqft=sqft,
                    bedrooms=int(prop["bedrooms"]),
                    year_built=int(prop["year_built"]),
                    neighborhood=str(prop["neighborhood"]),
                    condition=str(prop["condition"]),
                    price=price,
                    price_per_sqft=price / sqft if sqft > 0 else 0.0,
                    similarity_score=sim,
                    distance=dist,
                )
            )

        comp_prices = [c.price for c in comparables]
        median_price = int(np.median(comp_prices))
        mean_price = int(np.mean(comp_prices))

        subject_price: int | None = subject.get("price") if subject.get("price") else None
        dev_pct: float | None = None
        market_pos: str | None = None

        if subject_price is not None and median_price > 0:
            dev_pct = (subject_price - median_price) / median_price * 100.0
            thr = self.config.market_at_threshold_pct
            if dev_pct > thr:
                market_pos = "above_market"
            elif dev_pct < -thr:
                market_pos = "below_market"
            else:
                market_pos = "at_market"

        return CompsResult(
            comparables=comparables,
            subject_price=subject_price,
            median_comp_price=median_price,
            mean_comp_price=mean_price,
            price_deviation_pct=dev_pct,
            market_position=market_pos,
            n_comparables=len(comparables),
        )
