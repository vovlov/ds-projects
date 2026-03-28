"""Streamlit demo for NER Service."""

from __future__ import annotations

import streamlit as st

from ..model.ner import predict

st.set_page_config(page_title="NER Service Demo", layout="wide")

LABEL_COLORS = {
    "PER": "#FF6B6B",
    "ORG": "#4ECDC4",
    "LOC": "#45B7D1",
}

SAMPLE_TEXTS = [
    "Владимир Путин посетил Москву и встретился с представителями Газпрома.",
    "Компания Яндекс открыла новый офис в Санкт-Петербурге.",
    "Сбербанк заключил соглашение с ПАО Газпром в Екатеринбурге.",
    "Илон Маск является CEO компании Tesla в США.",
]


def highlight_entities(text: str, entities: list) -> str:
    """Create HTML with highlighted entities."""
    if not entities:
        return text

    # Sort by position (reverse to preserve offsets)
    sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
    result = text

    for entity in sorted_entities:
        color = LABEL_COLORS.get(entity.label, "#999")
        badge = (
            f'<span style="background-color: {color}; padding: 2px 6px; '
            f'border-radius: 4px; color: white; font-weight: bold;">'
            f"{entity.text} <sup>{entity.label}</sup></span>"
        )
        result = result[: entity.start] + badge + result[entity.end :]

    return result


def main():
    st.title("Named Entity Recognition — Russian Text")
    st.markdown(
        "Extracts **persons (PER)**, **organizations (ORG)**, and **locations (LOC)** from text."
    )
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        sample = st.selectbox("Sample texts:", ["Custom"] + SAMPLE_TEXTS)
        if sample == "Custom":
            text = st.text_area("Enter Russian text:", height=150, placeholder="Введите текст...")
        else:
            text = st.text_area("Enter Russian text:", value=sample, height=150)

    if text and st.button("Extract Entities", type="primary"):
        entities = predict(text)

        st.markdown("### Results")
        html = highlight_entities(text, entities)
        st.markdown(html, unsafe_allow_html=True)

        st.markdown("### Entities Found")
        if entities:
            for entity in entities:
                color = LABEL_COLORS.get(entity.label, "#999")
                st.markdown(
                    f"- **{entity.text}** → `{entity.label}` "
                    f'<span style="color: {color};">●</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No entities found.")

    with col2:
        st.markdown("### Legend")
        for label, color in LABEL_COLORS.items():
            st.markdown(
                f'<span style="background-color: {color}; padding: 2px 8px; '
                f'border-radius: 4px; color: white;">{label}</span>',
                unsafe_allow_html=True,
            )
            label_map = {"PER": "Person", "ORG": "Organization", "LOC": "Location"}
            st.caption(label_map.get(label, label))


if __name__ == "__main__":
    main()
