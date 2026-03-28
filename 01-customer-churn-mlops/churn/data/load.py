"""
Загрузка и предобработка данных Telco Customer Churn.

Датасет IBM содержит 7043 клиента телеком-оператора с 21 признаком.
Особенность: TotalCharges может прийти как строка с пробелом (для новых клиентов
с tenure=0 поле пустое в исходном CSV). Обрабатываем это при загрузке.
"""

from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PATH = DATA_DIR / "raw.csv"

# Признаки, которые кодируются как категории
CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

# Числовые признаки из исходных данных
NUMERICAL_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET = "Churn"


def load_raw(path: Path = RAW_PATH) -> pl.DataFrame:
    """Загрузить CSV и привести типы.

    TotalCharges — коварное поле: в некоторых версиях CSV оно строковое,
    потому что для клиентов с tenure=0 там пробел вместо числа.
    Polars может распарсить его как float или как string — обрабатываем оба случая.
    """
    df = pl.read_csv(path)

    tc = df["TotalCharges"]
    if tc.dtype == pl.String or tc.dtype == pl.Utf8:
        df = df.with_columns(
            pl.col("TotalCharges")
            .str.strip_chars()
            .replace("", None)
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("TotalCharges"),
        )
    else:
        df = df.with_columns(
            pl.col("TotalCharges").fill_null(0.0).alias("TotalCharges"),
        )

    # Целевая переменная: "Yes"/"No" → 1/0
    df = df.with_columns(
        pl.when(pl.col("Churn") == "Yes").then(1).otherwise(0).cast(pl.Int8).alias("Churn"),
    )
    return df


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Инженерные признаки — то, чего нет в исходных данных.

    Идеи:
    - AvgMonthlySpend: сколько клиент платит в среднем в месяц (не то же самое,
      что MonthlyCharges — тот показывает текущий тариф, а этот — факт)
    - ExpectedTotalCharges: сколько бы клиент заплатил, если бы текущий тариф
      не менялся с самого начала. Разница с TotalCharges показывает смену тарифа.
    - TenureGroup: сегментация по «зрелости» клиента (новый / средний / долгий)
    - NumServices: количество подключённых допуслуг (0–6). Гипотеза: чем больше
      услуг — тем сложнее уйти (switching cost).
    """
    return df.with_columns(
        (pl.col("TotalCharges") / (pl.col("tenure") + 1)).alias("AvgMonthlySpend"),
        (pl.col("MonthlyCharges") * pl.col("tenure")).alias("ExpectedTotalCharges"),
        pl.when(pl.col("tenure") <= 12)
        .then(pl.lit("new"))
        .when(pl.col("tenure") <= 36)
        .then(pl.lit("mid"))
        .otherwise(pl.lit("long"))
        .alias("TenureGroup"),
        (
            (pl.col("OnlineSecurity") == "Yes").cast(pl.Int8)
            + (pl.col("OnlineBackup") == "Yes").cast(pl.Int8)
            + (pl.col("DeviceProtection") == "Yes").cast(pl.Int8)
            + (pl.col("TechSupport") == "Yes").cast(pl.Int8)
            + (pl.col("StreamingTV") == "Yes").cast(pl.Int8)
            + (pl.col("StreamingMovies") == "Yes").cast(pl.Int8)
        ).alias("NumServices"),
    )


def prepare_dataset(path: Path = RAW_PATH) -> pl.DataFrame:
    """Полный пайплайн: загрузка → очистка → feature engineering."""
    df = load_raw(path)
    df = add_features(df)
    return df
