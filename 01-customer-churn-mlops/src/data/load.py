"""Data loading and preprocessing for Telco Customer Churn."""

from pathlib import Path

import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_PATH = DATA_DIR / "raw.csv"

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

NUMERICAL_FEATURES = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

TARGET = "Churn"


def load_raw(path: Path = RAW_PATH) -> pl.DataFrame:
    """Load raw CSV and apply minimal cleaning."""
    df = pl.read_csv(path)

    # TotalCharges may be string with " " for new customers, or already float
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

    df = df.with_columns(
        pl.when(pl.col("Churn") == "Yes").then(1).otherwise(0).cast(pl.Int8).alias("Churn"),
    )
    return df


def add_features(df: pl.DataFrame) -> pl.DataFrame:
    """Engineer features from raw data."""
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
    """Full pipeline: load → clean → engineer features."""
    df = load_raw(path)
    df = add_features(df)
    return df
