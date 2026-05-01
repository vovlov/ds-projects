"""
H3 геопространственные признаки для оценки стоимости недвижимости.

Зачем H3 вместо просто координат:
- Шестиугольные ячейки: все 6 соседей на одинаковом расстоянии (в отличие от квадратов)
- Иерархическая структура: r7 (район, ~5 км²) → r8 (микрорайон, ~0.74 км²)
- Uber использует H3 для surge pricing; Zillow/ЦИАН — для AVM (Automated Valuation Models)

price_vs_district = price / h3_r7_median_price
  > 1.0 → квартира дороже своего района (возможно завышена)
  < 1.0 → дешевле рынка (возможна скрытая ценность или ремонт)

ВНИМАНИЕ data leakage: h3_r7_median_price вычисляется из тех же данных, что и таргет.
Это аналог target encoding — требует кросс-валидационного вычисления при обучении.
В продакшне используется внешний исторический датасет цен (например, ЦИАН за 12 мес.).

Источники: Uber H3 blog 2018, h3geo.org docs, Zillow AVM research 2024.
"""

from __future__ import annotations

import numpy as np
import polars as pl

try:
    import h3 as _h3_lib

    _H3_AVAILABLE = True
except ImportError:
    _H3_AVAILABLE = False

# Приближённые координаты WGS84 центров московских районов.
# Точность ~500 м — достаточно для определения правильной H3-ячейки на уровне r7/r8.
NEIGHBORHOOD_COORDS: dict[str, tuple[float, float]] = {
    "Хамовники": (55.730, 37.570),
    "Арбат": (55.752, 37.600),
    "Тверской": (55.772, 37.620),
    "Пресненский": (55.760, 37.575),
    "Басманный": (55.762, 37.680),
    "Замоскворечье": (55.730, 37.630),
    "Дорогомилово": (55.742, 37.520),
    "Раменки": (55.720, 37.430),
    "Бутово Южное": (55.580, 37.600),
    "Марьино": (55.650, 37.750),
    "Люблино": (55.670, 37.770),
    "Бирюлёво": (55.600, 37.680),
    "Митино": (55.830, 37.360),
    "Строгино": (55.790, 37.370),
    "Куркино": (55.880, 37.380),
}

# Признаки, добавляемые этим модулем
GEO_FEATURES: list[str] = ["h3_r7", "h3_r8"]
GEO_MARKET_FEATURES: list[str] = ["h3_r7_median_price", "h3_r7_count", "price_vs_district"]


def is_available() -> bool:
    """Проверить наличие h3 библиотеки."""
    return _H3_AVAILABLE


def generate_neighborhood_coordinates(
    neighborhoods: list[str],
    rng: np.random.Generator,
    noise_deg: float = 0.004,
) -> tuple[list[float], list[float]]:
    """Сгенерировать псевдо-реальные координаты на основе названий районов.

    noise_deg ≈ 0.004° ≈ 350–400 м — реалистичный разброс квартир внутри района.
    Квартиры в одном административном районе действительно разбросаны на сотни
    метров от геометрического центра, но остаются в пределах своей H3-ячейки r7.

    Args:
        neighborhoods: Список названий районов
        rng: Инициализированный numpy random generator (для воспроизводимости)
        noise_deg: Стандартное отклонение шума в градусах

    Returns:
        (lats, lngs) — параллельные списки широт и долгот
    """
    lats: list[float] = []
    lngs: list[float] = []
    for neighborhood in neighborhoods:
        base_lat, base_lng = NEIGHBORHOOD_COORDS.get(neighborhood, (55.75, 37.62))
        lats.append(float(base_lat + rng.normal(0.0, noise_deg)))
        lngs.append(float(base_lng + rng.normal(0.0, noise_deg)))
    return lats, lngs


def lat_lng_to_h3(lat: float, lng: float, resolution: int) -> str:
    """Конвертировать координаты в идентификатор H3-ячейки.

    Args:
        lat: Широта в формате WGS84
        lng: Долгота в формате WGS84
        resolution: Разрешение H3 (7=район ~5км², 8=микрорайон ~0.74км²)

    Returns:
        Строковый H3 Cell ID или детерминированная mock-строка без h3.
    """
    if _H3_AVAILABLE:
        return _h3_lib.latlng_to_cell(lat, lng, resolution)
    # Детерминированный fallback без h3: группирует координаты в сетку,
    # где размер ячейки масштабируется примерно как реальный H3.
    # r7 ≈ 0.05°, r8 ≈ 0.025° — соответствует фактическим edge lengths.
    step = 0.05 / (2 ** (resolution - 7))
    grid_lat = round(int(lat / step) * step, 6)
    grid_lng = round(int(lng / step) * step, 6)
    return f"mock_r{resolution}_{grid_lat}_{grid_lng}"


def compute_h3_market_stats(
    df: pl.DataFrame,
    h3_col: str,
    price_col: str = "price",
) -> pl.DataFrame:
    """Добавить рыночную статистику по H3-ячейкам в датафрейм.

    Медиана устойчива к outliers (одна элитная квартира не поднимает
    «среднерыночную» цену всего микрорайона).
    """
    stats = df.group_by(h3_col).agg(
        [
            pl.col(price_col).median().alias(f"{h3_col}_median_price"),
            pl.col(price_col).count().alias(f"{h3_col}_count"),
        ]
    )
    return df.join(stats, on=h3_col, how="left")


def add_h3_features(
    df: pl.DataFrame,
    resolution_district: int = 7,
    resolution_local: int = 8,
    include_market_stats: bool = True,
) -> pl.DataFrame:
    """Добавить H3 геопространственные признаки в датафрейм.

    Входной датафрейм должен содержать столбцы 'latitude' и 'longitude'.
    Если их нет — возвращает датафрейм без изменений (graceful degradation).

    Args:
        df: Датафрейм с 'latitude', 'longitude' и опционально 'price'
        resolution_district: H3 разрешение для района (по умолчанию 7)
        resolution_local: H3 разрешение для микрорайона (по умолчанию 8)
        include_market_stats: Вычислять ли рыночную статистику по hex

    Returns:
        Датафрейм с новыми столбцами:
            h3_r7, h3_r8 — Cell IDs двух масштабов
            h3_r7_median_price, h3_r7_count — если include_market_stats
            price_vs_district — если include_market_stats и price есть
    """
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df

    lats = df["latitude"].to_list()
    lngs = df["longitude"].to_list()

    pairs = zip(lats, lngs, strict=True)
    h3_r7 = [lat_lng_to_h3(lat, lng, resolution_district) for lat, lng in pairs]
    pairs = zip(lats, lngs, strict=True)
    h3_r8 = [lat_lng_to_h3(lat, lng, resolution_local) for lat, lng in pairs]

    df = df.with_columns(
        [
            pl.Series("h3_r7", h3_r7, dtype=pl.String),
            pl.Series("h3_r8", h3_r8, dtype=pl.String),
        ]
    )

    if include_market_stats and "price" in df.columns:
        df = compute_h3_market_stats(df, "h3_r7")
        df = df.with_columns(
            (pl.col("price") / pl.col("h3_r7_median_price")).round(4).alias("price_vs_district")
        )

    return df


def enrich_with_geo(
    df: pl.DataFrame,
    seed: int = 42,
    noise_deg: float = 0.004,
    include_market_stats: bool = True,
) -> pl.DataFrame:
    """Обогатить датафрейм недвижимости геопространственными признаками.

    Удобная обёртка: генерирует координаты из столбца 'neighborhood',
    затем добавляет H3-признаки обоих масштабов и рыночную статистику.

    Требует столбец 'neighborhood' в датафрейме.

    Args:
        df: Датафрейм с колонкой 'neighborhood'
        seed: Seed для воспроизводимости генерации координат
        noise_deg: Разброс координат внутри района (в градусах)
        include_market_stats: Добавлять ли рыночную статистику по hex

    Returns:
        Датафрейм с добавленными geo-признаками
    """
    if "neighborhood" not in df.columns:
        return df

    rng = np.random.default_rng(seed)
    neighborhoods = df["neighborhood"].to_list()
    lats, lngs = generate_neighborhood_coordinates(neighborhoods, rng, noise_deg)

    df = df.with_columns(
        [
            pl.Series("latitude", lats, dtype=pl.Float64),
            pl.Series("longitude", lngs, dtype=pl.Float64),
        ]
    )

    return add_h3_features(df, include_market_stats=include_market_stats)
