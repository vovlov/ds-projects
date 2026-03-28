# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Data Quality Platform — EDA Demo
#
# Демонстрация профилирования данных и проверок качества.
# This notebook shows how to use the profiling and quality modules.

# %%
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path / Add project root to path
PROJECT_ROOT = Path.cwd().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# %%
import polars as pl

from quality.data.connectors import CSVConnector
from quality.data.profiler import profile_dataframe
from quality.quality.drift import detect_drift
from quality.quality.expectations import run_suite

# %% [markdown]
# ## 1. Загрузка данных / Load data
#
# Используем CSVConnector для чтения демо-данных.
# Если данных нет — запусти `python scripts/seed_demo_data.py`.

# %%
DATA_DIR = PROJECT_ROOT / "data"

ref_connector = CSVConnector(DATA_DIR / "reference.csv")
cur_connector = CSVConnector(DATA_DIR / "current.csv")

ref_df = ref_connector.read()
cur_df = cur_connector.read()

print(f"Reference: {ref_df.shape}")
print(f"Current:   {cur_df.shape}")

# %%
ref_df.head()

# %% [markdown]
# ## 2. Профилирование / Profiling
#
# Собираем статистику по каждому столбцу.

# %%
ref_profile = profile_dataframe(ref_df)

# Общая информация / Overview
print("=== Overview ===")
for k, v in ref_profile["overview"].items():
    print(f"  {k}: {v}")

# %%
# Статистика по числовым столбцам / Numeric column stats
for col_name, stats in ref_profile["columns"].items():
    if "mean" in stats and stats["mean"] is not None:
        print(f"\n--- {col_name} ---")
        print(f"  Mean:   {stats['mean']}")
        print(f"  Std:    {stats['std']}")
        print(f"  Min:    {stats['min']}")
        print(f"  Max:    {stats['max']}")
        print(f"  Nulls:  {stats['null_count']} ({stats['null_pct']}%)")
        print(f"  Dist:   {stats['distribution']}")

# %% [markdown]
# ## 3. Проверки качества / Quality Checks
#
# Запускаем набор expectations из YAML-конфига.

# %%
SUITE_PATH = PROJECT_ROOT / "configs" / "expectations.yaml"

ref_results = run_suite(ref_df, SUITE_PATH)

passed = sum(1 for r in ref_results if r["passed"])
total = len(ref_results)
print(f"\nReference data: {passed}/{total} checks passed")

for r in ref_results:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"  [{status}] {r['check']} on '{r['column']}'")
    if not r["passed"]:
        print(f"         Details: {r['details']}")

# %%
# Проверим текущие данные (у них есть пропуски в amount)
# Check current data (it has nulls in amount)
cur_results = run_suite(cur_df, SUITE_PATH)

passed = sum(1 for r in cur_results if r["passed"])
total = len(cur_results)
print(f"\nCurrent data: {passed}/{total} checks passed")

for r in cur_results:
    status = "PASS" if r["passed"] else "FAIL"
    print(f"  [{status}] {r['check']} on '{r['column']}'")
    if not r["passed"]:
        print(f"         Details: {r['details']}")

# %% [markdown]
# ## 4. Детекция дрифта / Drift Detection
#
# Сравниваем reference и current для числовых столбцов.

# %%
drift_report = detect_drift(ref_df, cur_df)

print(f"Drift detected: {drift_report['drift_detected']}")
print(f"Columns checked: {drift_report['columns_checked']}")
print(f"Columns with drift: {drift_report['columns_with_drift']}")

for detail in drift_report["details"]:
    col = detail["column"]
    if detail.get("status") == "skipped":
        print(f"\n  {col}: SKIPPED — {detail['reason']}")
    else:
        alert = "DRIFT" if detail["drift_detected"] else "OK"
        print(
            f"\n  {col}: [{alert}] "
            f"PSI={detail['psi']:.4f} ({detail['psi_alert']}), "
            f"KS p-value={detail['ks_p_value']:.4f}"
        )

# %% [markdown]
# ## 5. Визуализация дрифта / Drift Visualization
#
# Гистограммы для столбца `amount` — видно, что среднее выросло.

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ref_amounts = ref_df["amount"].drop_nulls().to_numpy()
cur_amounts = cur_df["amount"].drop_nulls().to_numpy()

ax.hist(ref_amounts, bins=50, alpha=0.5, label="Reference", density=True)
ax.hist(cur_amounts, bins=50, alpha=0.5, label="Current", density=True)
ax.set_xlabel("Amount")
ax.set_ylabel("Density")
ax.set_title("Distribution Drift: amount")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# Как видно, распределение `amount` сместилось вправо — среднее выросло,
# разброс увеличился. PSI и KS-тест корректно это обнаружили.
#
# The `amount` distribution shifted right — higher mean, wider spread.
# Both PSI and the KS test correctly flagged this drift.
