"""
Генерация демо-данных / Generate synthetic demo datasets.

Создаёт два CSV-файла с клиентскими транзакциями:
- data/reference.csv — эталонный датасет (январь)
- data/current.csv — текущий датасет (февраль, с дрифтом в amount)

The reference dataset is "normal", the current one has intentional drift
in the `amount` column to demonstrate drift detection capabilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

# Чтобы скрипт работал из корня проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def generate_transactions(
    n_rows: int,
    seed: int,
    amount_mean: float = 150.0,
    amount_std: float = 50.0,
    null_fraction: float = 0.0,
) -> pl.DataFrame:
    """
    Сгенерировать синтетический датасет транзакций.
    Generate a synthetic transactions dataset.
    """
    rng = np.random.default_rng(seed)

    categories = [
        "electronics",
        "groceries",
        "clothing",
        "entertainment",
        "dining",
        "travel",
        "health",
        "other",
    ]

    transaction_ids = list(range(1, n_rows + 1))
    customer_ids = rng.integers(1000, 9999, size=n_rows).tolist()
    amounts = np.abs(rng.normal(amount_mean, amount_std, size=n_rows)).tolist()
    ages = rng.integers(18, 75, size=n_rows).tolist()
    cats = rng.choice(categories, size=n_rows).tolist()

    # Даты: случайные дни в пределах месяца
    days = rng.integers(1, 28, size=n_rows)
    dates = [f"2025-01-{d:02d}" for d in days]

    df = pl.DataFrame(
        {
            "transaction_id": transaction_ids,
            "customer_id": customer_ids,
            "amount": amounts,
            "customer_age": ages,
            "category": cats,
            "transaction_date": dates,
        }
    )

    # Добавляем немного пропусков, если нужно / Add some nulls
    if null_fraction > 0:
        mask = rng.random(n_rows) < null_fraction
        # Polars: создаём новый столбец с null на нужных позициях
        amounts_with_nulls = [None if mask[i] else df["amount"][i] for i in range(n_rows)]
        df = df.with_columns(pl.Series("amount", amounts_with_nulls))

    return df


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    n_rows = 5000
    print(f"Генерируем {n_rows} строк эталонных данных...")
    print(f"Generating {n_rows} rows of reference data...")
    ref_df = generate_transactions(
        n_rows=n_rows,
        seed=42,
        amount_mean=150.0,
        amount_std=50.0,
    )
    ref_path = DATA_DIR / "reference.csv"
    ref_df.write_csv(str(ref_path))
    print(f"  -> {ref_path}")

    # Текущие данные с дрифтом: средняя сумма выросла, появились пропуски
    # Current data with drift: higher mean amount, some nulls
    print(f"\nГенерируем {n_rows} строк текущих данных (с дрифтом)...")
    print(f"Generating {n_rows} rows of current data (with drift)...")
    cur_df = generate_transactions(
        n_rows=n_rows,
        seed=123,
        amount_mean=220.0,  # дрифт: среднее выросло / drift: mean shifted up
        amount_std=80.0,  # дрифт: разброс вырос / drift: std increased
        null_fraction=0.03,  # 3% пропусков / 3% nulls
    )
    cur_path = DATA_DIR / "current.csv"
    cur_df.write_csv(str(cur_path))
    print(f"  -> {cur_path}")

    print("\nГотово! / Done!")
    print(f"Файлы в / Files in: {DATA_DIR}")


if __name__ == "__main__":
    main()
