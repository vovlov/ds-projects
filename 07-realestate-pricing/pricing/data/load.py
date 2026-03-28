"""
Генерация и загрузка синтетического датасета московской недвижимости.

Почему синтетика: реальные датасеты (Ames Housing, ЦИАН) либо американские,
либо требуют парсинга. Здесь мы генерируем 1000 квартир с реалистичными для
Москвы характеристиками — цены 3-30М руб, площади 25-200 кв.м, 15 районов
с разным уровнем цен.

Признаки спроектированы так, чтобы модель могла уловить реальные закономерности:
- цена растёт с площадью (но нелинейно — маленькие квартиры дороже за кв.м)
- район сильно влияет на цену (Хамовники vs Бирюлёво)
- год постройки влияет U-образно: сталинки и новостройки дороже хрущёвок
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Районы Москвы с ценовыми коэффициентами.
# Коэффициент отражает, во сколько раз район дороже «среднего» — это основной
# драйвер цены после площади.
NEIGHBORHOODS: dict[str, float] = {
    "Хамовники": 1.8,
    "Арбат": 1.9,
    "Тверской": 1.7,
    "Пресненский": 1.6,
    "Басманный": 1.4,
    "Замоскворечье": 1.5,
    "Дорогомилово": 1.3,
    "Раменки": 1.2,
    "Бутово Южное": 0.7,
    "Марьино": 0.8,
    "Люблино": 0.75,
    "Бирюлёво": 0.65,
    "Митино": 0.85,
    "Строгино": 0.95,
    "Куркино": 0.9,
}

# Состояние квартиры: множитель к цене
CONDITION_MAP: dict[str, float] = {
    "отличное": 1.15,
    "хорошее": 1.0,
    "удовлетворительное": 0.85,
    "требует ремонта": 0.7,
}

NUMERICAL_FEATURES = [
    "sqft",
    "bedrooms",
    "bathrooms",
    "year_built",
    "lot_size",
    "age",
    "price_per_sqft",
]

CATEGORICAL_FEATURES = [
    "neighborhood",
    "condition",
    "has_garage",
]

TARGET = "price"

CURRENT_YEAR = 2026


def generate_dataset(n_rows: int = 1000, seed: int = 42) -> pl.DataFrame:
    """Сгенерировать реалистичный датасет московской недвижимости.

    Логика ценообразования:
      base_price = sqft * base_rate * neighborhood_coeff * condition_coeff * age_factor
    где age_factor — U-образная функция (сталинки и новостройки дороже).
    Добавляется шум ~10%, чтобы модель не получала идеальный R^2 = 1.0.
    """
    rng = np.random.default_rng(seed)

    neighborhoods = list(NEIGHBORHOODS.keys())
    conditions = list(CONDITION_MAP.keys())

    # Площадь: 25-200 кв.м, лог-нормальное распределение (большинство 40-80)
    sqft = np.clip(rng.lognormal(mean=4.0, sigma=0.4, size=n_rows), 25, 200).astype(int)

    # Количество комнат зависит от площади
    bedrooms = np.clip(np.round(sqft / 30).astype(int), 1, 6)

    # Санузлы: 1 до 60 кв.м, иначе пропорционально
    bathrooms = np.where(sqft < 60, 1, np.clip(np.round(sqft / 50).astype(int), 1, 4))

    # Год постройки: смесь «эпох» — сталинки, хрущёвки, брежневки, новостройки
    year_probs = [0.08, 0.15, 0.25, 0.52]  # сталин, хрущ, бреж, новые
    epoch = rng.choice(4, size=n_rows, p=year_probs)
    year_built = np.where(
        epoch == 0,
        rng.integers(1935, 1956, size=n_rows),
        np.where(
            epoch == 1,
            rng.integers(1956, 1972, size=n_rows),
            np.where(
                epoch == 2,
                rng.integers(1972, 1995, size=n_rows),
                rng.integers(2005, 2026, size=n_rows),
            ),
        ),
    )

    # Площадь участка (для квартир — условная, для домов больше)
    lot_size = np.where(
        sqft > 120,
        rng.integers(200, 800, size=n_rows),
        rng.integers(0, 50, size=n_rows),
    )

    # Гараж: чаще в новостройках и больших квартирах
    garage_prob = np.where(year_built > 2000, 0.6, 0.2) * np.where(sqft > 80, 1.3, 0.7)
    garage_prob = np.clip(garage_prob, 0, 0.95)
    garage = rng.binomial(1, garage_prob)

    # Район и состояние — случайные, но состояние лучше в новостройках
    neighborhood_idx = rng.integers(0, len(neighborhoods), size=n_rows)
    neighborhood = [neighborhoods[i] for i in neighborhood_idx]
    neighborhood_coeff = np.array([NEIGHBORHOODS[n] for n in neighborhood])

    condition_probs_new = [0.4, 0.4, 0.15, 0.05]
    condition_probs_old = [0.1, 0.3, 0.35, 0.25]
    condition_idx = np.array(
        [
            rng.choice(4, p=condition_probs_new if yb > 2000 else condition_probs_old)
            for yb in year_built
        ]
    )
    condition = [conditions[i] for i in condition_idx]
    condition_coeff = np.array([CONDITION_MAP[c] for c in condition])

    # Ценообразование: базовая ставка ~200K руб/кв.м (средняя по Москве)
    base_rate = 200_000

    # U-образный фактор возраста: сталинки (+10%), хрущёвки (-15%), новостройки (+5%)
    age = CURRENT_YEAR - year_built
    age_factor = np.where(
        age > 70,
        1.10,  # сталинки — историческая ценность
        np.where(
            age > 50,
            0.85,  # хрущёвки — дёшево
            np.where(
                age > 30,
                0.92,  # брежневки — нормально
                1.05,  # новостройки — наценка
            ),
        ),
    )

    # Гараж добавляет ~5% к цене
    garage_factor = np.where(garage == 1, 1.05, 1.0)

    # Итоговая цена с шумом
    price = sqft * base_rate * neighborhood_coeff * condition_coeff * age_factor * garage_factor
    noise = rng.normal(1.0, 0.10, size=n_rows)  # ~10% шум
    price = (price * noise).astype(int)

    # Ограничиваем разумным диапазоном
    price = np.clip(price, 3_000_000, 30_000_000)

    df = pl.DataFrame(
        {
            "price": price,
            "sqft": sqft,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "year_built": year_built,
            "lot_size": lot_size,
            "garage": garage,
            "neighborhood": neighborhood,
            "condition": condition,
        }
    )

    return df


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Добавить инженерные признаки.

    - age: возраст дома — проще интерпретировать, чем year_built
    - price_per_sqft: цена за метр — полезно для EDA и как leak-free фичу не используем
      в модели (она вычисляется из таргета), но нужна для анализа
    - has_garage: бинарный вместо int — CatBoost обработает как категорию
    """
    return df.with_columns(
        (pl.lit(CURRENT_YEAR) - pl.col("year_built")).alias("age"),
        (pl.col("price") / pl.col("sqft")).round(0).cast(pl.Int64).alias("price_per_sqft"),
        pl.when(pl.col("garage") == 1)
        .then(pl.lit("yes"))
        .otherwise(pl.lit("no"))
        .alias("has_garage"),
    )


def load_dataset(n_rows: int = 1000, seed: int = 42) -> pl.DataFrame:
    """Основная точка входа: генерация + feature engineering."""
    df = generate_dataset(n_rows=n_rows, seed=seed)
    return add_features(df)
