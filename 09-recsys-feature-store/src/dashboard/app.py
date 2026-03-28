"""
Streamlit-дашборд для рекомендательной системы.
Interactive dashboard: explore recommendations, metrics, and data patterns.
"""

from __future__ import annotations

import polars as pl
import streamlit as st

from src.data.load import load_all_data
from src.models.collaborative import CollaborativeRecommender
from src.models.content_based import ContentBasedRecommender, get_popular_items


@st.cache_resource
def load_data():
    """Кешируем загрузку данных / Cache data loading."""
    return load_all_data()


@st.cache_resource
def train_collaborative(_interactions: pl.DataFrame):
    """Кешируем обучение модели / Cache model training."""
    model = CollaborativeRecommender(n_components=30)
    model.fit(_interactions)
    return model


@st.cache_resource
def train_content_based(_products: pl.DataFrame):
    """Кешируем контентную модель / Cache content-based model."""
    model = ContentBasedRecommender()
    model.fit(_products)
    return model


def main() -> None:
    st.set_page_config(
        page_title="RecSys Dashboard",
        page_icon="🛒",
        layout="wide",
    )

    st.title("Recommendation Engine Dashboard")
    st.caption("Дашборд рекомендательной системы для e-commerce")

    # Загрузка данных
    users, products, interactions = load_data()
    collab_model = train_collaborative(interactions)
    content_model = train_content_based(products)

    # --- Боковая панель / Sidebar ---
    st.sidebar.header("Settings / Настройки")

    tab1, tab2, tab3 = st.tabs(
        [
            "Recommendations",
            "Interaction Heatmap",
            "Model Metrics",
        ]
    )

    # --- Вкладка 1: Рекомендации ---
    with tab1:
        st.subheader("User Recommendations / Рекомендации для пользователя")

        user_ids = sorted(interactions["user_id"].unique().to_list())
        selected_user = st.selectbox("Select user / Выберите пользователя", user_ids)
        top_k = st.slider("Number of recommendations / Количество", 5, 20, 10)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Collaborative Filtering (SVD)**")
            collab_recs = collab_model.recommend(selected_user, top_k=top_k)
            if collab_recs:
                rec_df = pl.DataFrame(
                    {
                        "product_id": [r[0] for r in collab_recs],
                        "score": [round(r[1], 3) for r in collab_recs],
                    }
                )
                # Добавляем информацию о товаре
                rec_with_info = rec_df.join(products, on="product_id", how="left")
                st.dataframe(rec_with_info.to_pandas(), use_container_width=True)
            else:
                st.info("No collaborative recommendations for this user")

        with col2:
            st.markdown("**Content-Based Filtering**")
            content_recs = content_model.recommend(selected_user, interactions, top_k=top_k)
            if content_recs:
                rec_df = pl.DataFrame(
                    {
                        "product_id": [r[0] for r in content_recs],
                        "score": [round(r[1], 3) for r in content_recs],
                    }
                )
                rec_with_info = rec_df.join(products, on="product_id", how="left")
                st.dataframe(rec_with_info.to_pandas(), use_container_width=True)
            else:
                st.info("No content-based recommendations for this user")

        # Популярные товары
        st.markdown("---")
        st.subheader("Popular Items (Cold Start) / Популярные товары")
        popular = get_popular_items(interactions, top_k=10)
        pop_df = pl.DataFrame(
            {
                "product_id": [p[0] for p in popular],
                "score": [round(p[1], 3) for p in popular],
            }
        ).join(products, on="product_id", how="left")
        st.dataframe(pop_df.to_pandas(), use_container_width=True)

    # --- Вкладка 2: Тепловая карта ---
    with tab2:
        st.subheader("Interaction Heatmap / Тепловая карта взаимодействий")
        st.caption("Users x Categories: average rating")

        # Строим матрицу пользователь x категория
        inter_with_cat = interactions.join(
            products.select(["product_id", "category"]),
            on="product_id",
            how="left",
        )

        # Агрегируем: средний рейтинг по (возрастная группа, категория)
        heatmap_data = (
            inter_with_cat.join(
                users.select(["user_id", "age_group"]),
                on="user_id",
                how="left",
            )
            .group_by(["age_group", "category"])
            .agg(pl.col("rating").mean().alias("avg_rating"))
            .sort(["age_group", "category"])
        )

        # Pivot для тепловой карты
        pivot = heatmap_data.pivot(
            on="category",
            index="age_group",
            values="avg_rating",
        ).sort("age_group")

        st.dataframe(
            pivot.to_pandas()
            .set_index("age_group")
            .style.background_gradient(cmap="YlOrRd", axis=None),
            use_container_width=True,
        )

        # Распределение оценок
        st.markdown("---")
        st.subheader("Rating Distribution / Распределение оценок")
        rating_counts = interactions["rating"].value_counts().sort("rating")
        st.bar_chart(
            rating_counts.to_pandas().set_index("rating"),
        )

    # --- Вкладка 3: Метрики модели ---
    with tab3:
        st.subheader("Model Metrics / Метрики модели")

        k_values = [5, 10, 20]
        metrics_rows = []

        # Простой train/test split для оценки
        n = interactions.height
        split_idx = int(n * 0.8)
        # Сортируем по времени для реалистичного split
        sorted_inter = interactions.sort("timestamp")
        train = sorted_inter.head(split_idx)
        test = sorted_inter.tail(n - split_idx)

        # Обучаем на train
        eval_model = CollaborativeRecommender(n_components=30)
        eval_model.fit(train)

        for k in k_values:
            metrics = eval_model.evaluate(test, top_k=k)
            metrics_rows.append(
                {
                    "K": k,
                    "Precision@K": round(metrics["precision_at_k"], 4),
                    "Recall@K": round(metrics["recall_at_k"], 4),
                    "NDCG@K": round(metrics["ndcg_at_k"], 4),
                }
            )

        metrics_df = pl.DataFrame(metrics_rows)
        st.dataframe(metrics_df.to_pandas(), use_container_width=True)

        # Описание метрик
        st.markdown("""
        **Metrics explanation / Объяснение метрик:**
        - **Precision@K** — доля релевантных среди рекомендованных
        - **Recall@K** — доля найденных релевантных из всех релевантных
        - **NDCG@K** — учитывает позицию релевантных в списке
        """)


if __name__ == "__main__":
    main()
