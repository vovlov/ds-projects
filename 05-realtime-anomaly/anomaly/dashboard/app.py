"""Streamlit-дашборд для Realtime Anomaly Detection.

Три вкладки:
1. Live Monitor — имитация потока метрик с детекцией аномалий (Z-score)
2. Drift Detection — MMD-тест: сравнение reference vs current распределений
3. Architecture — как устроена система, ссылки на API

Дашборд работает автономно без запущенного FastAPI-сервера:
данные генерируются в браузере, детекция запускается локально.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .utils import (
    compute_detection_summary,
    generate_current_data,
    generate_metric_stream,
    generate_reference_data,
)

st.set_page_config(
    page_title="Anomaly Detection — SRE Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _make_metric_traces(
    t: np.ndarray,
    stream: dict[str, np.ndarray],
    predictions: np.ndarray,
    metric: str,
    color: str,
) -> go.Figure:
    """Строит Plotly-график одной метрики с маркерами аномалий."""
    anomaly_idx = np.where(predictions == 1)[0]
    series = stream[metric]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=series,
            mode="lines",
            name=metric,
            line={"color": color, "width": 1.5},
        )
    )
    if len(anomaly_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=t[anomaly_idx],
                y=series[anomaly_idx],
                mode="markers",
                name="Anomaly",
                marker={"color": "#e74c3c", "size": 8, "symbol": "x"},
                showlegend=True,
            )
        )
    fig.update_layout(
        height=200,
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        title={"text": metric.upper(), "font": {"size": 13}},
        yaxis={"title": _metric_unit(metric)},
        showlegend=False,
    )
    return fig


def _metric_unit(metric: str) -> str:
    units = {"cpu": "% CPU", "latency": "ms", "requests": "rps"}
    return units.get(metric, metric)


def _tab_live_monitor() -> None:
    """Вкладка 1: имитация потока метрик + Z-score детекция аномалий."""
    st.header("Live Metrics Monitor")
    st.markdown(
        "Детекция аномалий через **rolling Z-score**: если значение метрики отклоняется "
        "от скользящего среднего более чем на `threshold_sigma` стандартных отклонений — "
        "аномалия. Красные × = обнаруженные аномалии."
    )

    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    with col_ctrl1:
        n_points = st.slider("Stream length", 100, 500, 200, 50)
    with col_ctrl2:
        window_size = st.slider("Window size", 10, 100, 50, 10)
    with col_ctrl3:
        threshold = st.slider("Threshold σ", 1.5, 5.0, 3.0, 0.5)
    with col_ctrl4:
        inject = st.checkbox("Inject incident", value=True, help="Добавить инцидент в конец потока")

    if st.button("Generate & Detect", type="primary"):
        # Импортируем детектор из пакета (не из API — нет зависимости от FastAPI)
        from anomaly.models.detector import MultiMetricDetector

        stream = generate_metric_stream(
            n_points=n_points,
            inject_anomaly=inject,
            anomaly_start=int(n_points * 0.75),
            seed=42,
        )

        detector = MultiMetricDetector(window_size=window_size, threshold_sigma=threshold)
        result = detector.detect(stream)

        t = np.arange(n_points)
        colors = {"cpu": "#3498db", "latency": "#f39c12", "requests": "#2ecc71"}

        for metric in ["cpu", "latency", "requests"]:
            fig = _make_metric_traces(t, stream, result.predictions, metric, colors[metric])
            st.plotly_chart(fig, use_container_width=True)

        # Сводная статистика
        summary = compute_detection_summary(result.predictions, result.scores)
        st.markdown("---")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total points", summary["n_total"])
        c2.metric("Anomalies detected", summary["n_anomalies"])
        c3.metric("Anomaly rate", f"{summary['anomaly_rate']:.1%}")
        c4.metric("Max Z-score", f"{summary['max_score']:.2f}")
        c5.metric("Mean Z-score", f"{summary['mean_score']:.2f}")

        # Z-score trace
        st.markdown("**Z-score over time** (red line = threshold)")
        fig_score = go.Figure()
        fig_score.add_trace(
            go.Scatter(
                x=t, y=result.scores, mode="lines", name="Z-score", line={"color": "#9b59b6"}
            )
        )
        fig_score.add_hline(
            y=threshold, line_dash="dash", line_color="#e74c3c", annotation_text=f"σ={threshold}"
        )
        fig_score.update_layout(height=200, margin={"l": 0, "r": 0, "t": 10, "b": 0})
        st.plotly_chart(fig_score, use_container_width=True)
    else:
        st.info("Нажми **Generate & Detect** для запуска детекции.")


def _tab_drift_detection() -> None:
    """Вкладка 2: MMD drift detection — сравниваем distributions."""
    st.header("Distribution Drift Detection (MMD)")
    st.markdown(
        "**Maximum Mean Discrepancy (MMD)** — kernel-based тест: обнаруживает сдвиг "
        "распределения между *reference* (нормальная работа) и *current* (мониторируемый период). "
        "Threshold вычисляется через bootstrap (permutation test, Gretton et al. 2012 JMLR)."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_ref = st.slider("Reference points", 50, 300, 150, 50)
    with col2:
        n_cur = st.slider("Current points", 30, 200, 80, 10)
    with col3:
        inject_drift = st.checkbox(
            "Inject drift", value=True, help="Сдвинуть distribution в current"
        )
    drift_mag = 3.0
    if inject_drift:
        drift_mag = st.slider("Drift magnitude (σ)", 1.0, 8.0, 4.0, 0.5)

    if st.button("Run MMD Test", type="primary"):
        from anomaly.drift.mmd import MMDDriftDetector

        ref_data = generate_reference_data(n_points=n_ref, seed=0)
        cur_data = generate_current_data(
            n_points=n_cur, inject_drift=inject_drift, drift_magnitude=drift_mag, seed=99
        )

        ref_arr = np.array(ref_data)
        cur_arr = np.array(cur_data)

        # Запускаем MMD detector (numpy-only, без PyTorch)
        detector = MMDDriftDetector(n_bootstrap=200, alpha=0.05)
        detector.fit(ref_arr)
        drift_result = detector.detect(cur_arr)

        # Результат
        status_color = "🔴" if drift_result.is_drift else "🟢"
        status_text = "DRIFT DETECTED" if drift_result.is_drift else "STABLE"
        st.markdown(f"## {status_color} {status_text}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MMD Statistic", f"{drift_result.mmd_statistic:.4f}")
        c2.metric("Threshold (α=5%)", f"{drift_result.threshold:.4f}")
        c3.metric("p-value", f"{drift_result.p_value:.3f}")
        c4.metric("Drift", "Yes" if drift_result.is_drift else "No")

        # Визуализация распределений по каждой метрике
        metric_names = ["CPU (%)", "Latency (ms)", "Requests (rps)"]
        colors_ref = "#3498db"
        colors_cur = "#e74c3c" if inject_drift else "#2ecc71"

        fig = go.Figure()
        for i, mname in enumerate(metric_names):
            fig.add_trace(
                go.Violin(
                    y=ref_arr[:, i],
                    name=f"Ref: {mname}",
                    side="negative",
                    line_color=colors_ref,
                    showlegend=i == 0,
                    legendgroup="ref",
                )
            )
            fig.add_trace(
                go.Violin(
                    y=cur_arr[:, i],
                    name=f"Cur: {mname}",
                    side="positive",
                    line_color=colors_cur,
                    showlegend=i == 0,
                    legendgroup="cur",
                )
            )

        fig.update_layout(
            title="Reference vs Current Distributions (violin plot)",
            height=350,
            violingap=0.3,
            violinmode="overlay",
            margin={"l": 0, "r": 0, "t": 40, "b": 0},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Audit ID: `{drift_result.audit_id}` | {drift_result.timestamp}")
    else:
        st.info("Нажми **Run MMD Test** для запуска теста распределений.")


def _tab_architecture() -> None:
    """Вкладка 3: описание архитектуры системы."""
    st.header("System Architecture")

    st.markdown(
        """
        ### Detection Pipeline

        ```
        Prometheus scrape → Kafka/HTTP stream
                ↓
        [05-realtime-anomaly]
          POST /detect        → Z-score multi-metric ensemble
          POST /drift/check   → MMD bootstrap test (Gretton 2012)
          GET  /metrics       → Prometheus exposition format
                ↓
        Grafana dashboards + AlertManager
                ↓
        [01-customer-churn] POST /retraining/notify
        ```

        ### Компоненты

        | Компонент | Технология | Описание |
        |-----------|-----------|----------|
        | Детектор | Rolling Z-score | <1ms latency, без обучения |
        | Drift тест | MMD RBF kernel | Kernel-based, без предположений о распределении |
        | Метрики | Prometheus client | 7 метрик: Counter, Histogram, Gauge |
        | Оркестрация | FastAPI | REST API с Pydantic v2 |
        | Мониторинг | Grafana | Pre-built dashboard в `grafana/` |

        ### Ссылки

        - **FastAPI docs**: `http://localhost:8005/docs` (локально)
        - **Prometheus metrics**: `http://localhost:8005/metrics`
        - **Источник**: Gretton et al. 2012 JMLR, Evidently AI v0.5, EU AI Act Article 9
        """
    )

    # Простая демо-схема MMD
    st.markdown("### MMD Statistic vs Threshold")
    st.markdown(
        "Ниже — пример: при нормальной работе MMD < threshold (зелёный), "
        "при дрейфе — MMD > threshold (красный)."
    )

    fig = go.Figure()
    scenarios = ["Normal", "Slight drift", "Moderate drift", "Strong drift"]
    mmd_vals = [0.003, 0.018, 0.045, 0.12]
    threshold = 0.025
    colors_bar = ["#2ecc71" if v < threshold else "#e74c3c" for v in mmd_vals]

    fig.add_trace(go.Bar(x=scenarios, y=mmd_vals, marker_color=colors_bar, name="MMD statistic"))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#f39c12", annotation_text="Threshold (α=5%)"
    )
    fig.update_layout(
        height=250,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        yaxis_title="MMD statistic",
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    """Точка входа для Streamlit."""
    st.title("Realtime Anomaly Detection — SRE Dashboard")
    st.markdown(
        "Мониторинг инфраструктурных метрик (CPU, Latency, Requests) "
        "в реальном времени с детекцией аномалий и MMD drift detection."
    )
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Live Monitor", "Drift Detection (MMD)", "Architecture"])

    with tab1:
        _tab_live_monitor()
    with tab2:
        _tab_drift_detection()
    with tab3:
        _tab_architecture()


if __name__ == "__main__":
    main()
