"""
Streamlit-дашборд для Data Quality Platform.

Три вкладки:
1. Профилирование данных — загрузи CSV и смотри статистику по столбцам
2. Проверки качества — запусти набор expectations и посмотри результаты
3. Дрифт — сравни два датасета и увидь, где распределения "уехали"

Three tabs: Profiling, Quality Checks, Drift Detection.
"""

from __future__ import annotations

import io

import polars as pl
import streamlit as st

from src.data.profiler import profile_dataframe
from src.quality.drift import detect_drift
from src.quality.expectations import run_suite

st.set_page_config(
    page_title="Data Quality Platform",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Data Quality Platform")
st.markdown("Платформа мониторинга качества данных / Data quality monitoring dashboard")

tab_profile, tab_quality, tab_drift = st.tabs(
    [
        "Profiling",
        "Quality Checks",
        "Drift Detection",
    ]
)


# -----------------------------------------------------------------------
# Вкладка 1: Профилирование / Tab 1: Profiling
# -----------------------------------------------------------------------
with tab_profile:
    st.header("Data Profiling")
    uploaded = st.file_uploader("Загрузите CSV / Upload CSV", type=["csv"], key="profile_csv")

    if uploaded is not None:
        df = pl.read_csv(io.BytesIO(uploaded.getvalue()))
        st.write(f"**Rows:** {df.height}  |  **Columns:** {df.width}")

        report = profile_dataframe(df)

        # Показываем превью данных / Preview
        with st.expander("Data Preview", expanded=False):
            st.dataframe(df.head(100).to_pandas(), use_container_width=True)

        # Статистика по столбцам / Per-column stats
        for col_name, col_stats in report["columns"].items():
            with st.expander(f"**{col_name}** ({col_stats['dtype']})"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Rows", col_stats["count"])
                    st.metric("Nulls", col_stats["null_count"])
                    st.metric("Null %", f"{col_stats['null_pct']}%")
                with col_b:
                    st.metric("Unique", col_stats["unique_count"])
                    dist = col_stats.get("distribution", "n/a")
                    st.metric("Distribution", dist)

                if "mean" in col_stats and col_stats["mean"] is not None:
                    st.write(
                        f"Mean: {col_stats['mean']} | "
                        f"Std: {col_stats['std']} | "
                        f"Min: {col_stats['min']} | "
                        f"Max: {col_stats['max']}"
                    )
                    # Гистограмма / Histogram
                    pandas_series = df[col_name].drop_nulls().to_pandas()
                    st.bar_chart(pandas_series.value_counts().sort_index().head(50))

                if "top_values" in col_stats:
                    st.write("**Top values:**")
                    st.json(col_stats["top_values"])


# -----------------------------------------------------------------------
# Вкладка 2: Проверки качества / Tab 2: Quality Checks
# -----------------------------------------------------------------------
with tab_quality:
    st.header("Quality Checks")

    uploaded_q = st.file_uploader("Загрузите CSV / Upload CSV", type=["csv"], key="quality_csv")
    suite_text = st.text_area(
        "YAML-конфиг проверок / Suite config (YAML)",
        value=(
            "suite_name: demo\n"
            "expectations:\n"
            "  - check: expect_not_null\n"
            "    column: id\n"
            "  - check: expect_unique\n"
            "    column: id\n"
        ),
        height=200,
    )

    if uploaded_q is not None and st.button("Run Checks"):
        import yaml

        df_q = pl.read_csv(io.BytesIO(uploaded_q.getvalue()))
        suite_config = yaml.safe_load(suite_text)
        results = run_suite(df_q, suite_config)

        passed = sum(1 for r in results if r["passed"])
        total = len(results)
        st.write(f"**{passed}/{total} checks passed**")

        for r in results:
            icon = ":white_check_mark:" if r["passed"] else ":x:"
            label = f"{icon} {r['check']} on `{r['column']}`"
            with st.expander(label, expanded=not r["passed"]):
                st.json(r["details"])


# -----------------------------------------------------------------------
# Вкладка 3: Дрифт / Tab 3: Drift Detection
# -----------------------------------------------------------------------
with tab_drift:
    st.header("Distribution Drift Detection")

    col_ref, col_cur = st.columns(2)
    with col_ref:
        ref_file = st.file_uploader("Reference CSV", type=["csv"], key="drift_ref")
    with col_cur:
        cur_file = st.file_uploader("Current CSV", type=["csv"], key="drift_cur")

    if ref_file is not None and cur_file is not None and st.button("Detect Drift"):
        ref_df = pl.read_csv(io.BytesIO(ref_file.getvalue()))
        cur_df = pl.read_csv(io.BytesIO(cur_file.getvalue()))

        drift_report = detect_drift(ref_df, cur_df)

        status = ":red[DRIFT DETECTED]" if drift_report["drift_detected"] else ":green[No drift]"
        st.subheader(status)
        st.write(
            f"Checked {drift_report['columns_checked']} columns, "
            f"{drift_report['columns_with_drift']} with drift"
        )

        for detail in drift_report["details"]:
            col_name = detail["column"]
            if detail.get("status") == "skipped":
                st.warning(f"{col_name}: skipped — {detail['reason']}")
                continue

            has_drift = detail["drift_detected"]
            icon = ":red_circle:" if has_drift else ":green_circle:"
            label = (
                f"{icon} **{col_name}** — "
                f"PSI={detail['psi']:.4f} ({detail['psi_alert']}), "
                f"KS p={detail['ks_p_value']:.4f}"
            )
            with st.expander(label, expanded=has_drift):
                # Сравнение гистограмм / Histogram comparison
                import pandas as pd

                ref_vals = ref_df[col_name].drop_nulls().to_pandas()
                cur_vals = cur_df[col_name].drop_nulls().to_pandas()

                import numpy as np

                bins = np.linspace(
                    min(ref_vals.min(), cur_vals.min()),
                    max(ref_vals.max(), cur_vals.max()),
                    30,
                )
                ref_hist = np.histogram(ref_vals, bins=bins)[0]
                cur_hist = np.histogram(cur_vals, bins=bins)[0]

                bin_labels = [f"{b:.1f}" for b in bins[:-1]]
                chart_df = pd.DataFrame(
                    {
                        "bin": bin_labels,
                        "reference": ref_hist,
                        "current": cur_hist,
                    }
                ).set_index("bin")

                st.bar_chart(chart_df)
