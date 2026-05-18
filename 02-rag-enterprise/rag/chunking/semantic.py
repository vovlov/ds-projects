"""
Семантическая нарезка документов на чанки по смысловым границам.

В отличие от RecursiveCharacterTextSplitter с фиксированным размером,
семантические чанки группируют предложения с похожим содержанием.
Это улучшает recall RAG, сохраняя смысловые единицы (один топик → один чанк).

Алгоритм (Douze et al. 2024 + LangChain SemanticChunker):
  1. Разбить на предложения (граница: .!? или двойной \n)
  2. TF-IDF векторизация каждого предложения
  3. Объединять соседние предложения пока cosine(s_i, s_i+1) > threshold
  4. Принудительно разрезать чанки превышающие max_chunk_chars
  5. Объединять слишком короткие фрагменты с соседом

Graceful degradation: без sklearn → paragraph splitting по двойным \n.

Источники:
- LangChain SemanticChunker (embedding-based boundary detection)
- Anthropic Contextual Retrieval blog 2024
- Douze et al. 2024 FAISS (cosine drift как граница чанка)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class SemanticChunkConfig:
    """Параметры семантического чанкера."""

    similarity_threshold: float = 0.30
    """Порог cosine similarity: ниже порога — начинается новый чанк.
    0.30 — хорошо работает для смешанных корпоративных документов.
    Увеличьте до 0.45 для тематически однородных текстов.
    """

    min_chunk_chars: int = 100
    """Минимальный размер чанка; слишком короткие объединяются с соседом."""

    max_chunk_chars: int = 1200
    """Максимальный размер чанка; предотвращает слишком большие блоки."""


ChunkingStrategy = Literal["fixed", "semantic", "paragraph"]


def is_available() -> bool:
    """Проверить доступность sklearn для TF-IDF векторизации."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401

        return True
    except ImportError:
        return False


def _split_into_sentences(text: str) -> list[str]:
    """Разбить текст на предложения: сначала абзацы, затем по знакам пунктуации."""
    paragraphs = re.split(r"\n\n+", text)
    sentences: list[str] = []
    for para in paragraphs:
        parts = re.split(r"(?<=[.!?])\s+", para.strip())
        sentences.extend(p.strip() for p in parts if p.strip())
    return sentences


def _paragraph_chunks(text: str, max_chars: int = 1200) -> list[str]:
    """Fallback-разбивка по двойным переводам строк без sklearn."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current = paragraphs[0]
    for para in paragraphs[1:]:
        # объединяем абзац с текущим чанком пока не превышен лимит
        if len(current) + len(para) + 2 <= max_chars:
            current = current + "\n\n" + para
        else:
            chunks.append(current)
            current = para
    chunks.append(current)
    return chunks


class SemanticChunker:
    """Чанкер на основе TF-IDF косинусного сходства между соседними предложениями.

    Резкое падение cosine similarity между s_i и s_{i+1} сигнализирует о смене
    темы — это граница чанка. Подход аналогичен LangChain SemanticChunker, но
    использует TF-IDF вместо dense embeddings (не требует API или torch).
    """

    def __init__(self, config: SemanticChunkConfig | None = None) -> None:
        self.config = config or SemanticChunkConfig()
        self._sklearn_available = is_available()

    def chunk(self, text: str) -> list[str]:
        """Разбить текст на семантические чанки.

        Args:
            text: Исходный текст документа.

        Returns:
            Список непустых чанков.
        """
        if not text.strip():
            return []

        if self._sklearn_available:
            return self._tfidf_chunk(text)
        return _paragraph_chunks(text, self.config.max_chunk_chars)

    def _tfidf_chunk(self, text: str) -> list[str]:
        """Семантическая нарезка через TF-IDF cosine similarity (sklearn)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

        sentences = _split_into_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return [sentences[0]]

        vectorizer = TfidfVectorizer(min_df=1)
        try:
            matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            # Пустой словарь (текст только из цифр/символов) → paragraph fallback
            return _paragraph_chunks(text, self.config.max_chunk_chars)

        groups: list[list[str]] = [[sentences[0]]]
        for i in range(1, len(sentences)):
            sim = float(sk_cosine(matrix[i - 1], matrix[i])[0, 0])
            current_len = sum(len(s) for s in groups[-1])

            if (
                sim >= self.config.similarity_threshold
                and current_len < self.config.max_chunk_chars
            ):
                groups[-1].append(sentences[i])
            else:
                groups.append([sentences[i]])

        raw_chunks = [" ".join(g) for g in groups]
        return self._merge_small(raw_chunks)

    def _merge_small(self, chunks: list[str]) -> list[str]:
        """Объединить слишком короткие чанки с предыдущим соседом."""
        if not chunks:
            return chunks
        result = [chunks[0]]
        for chunk in chunks[1:]:
            if len(result[-1]) < self.config.min_chunk_chars:
                result[-1] = result[-1] + " " + chunk
            else:
                result.append(chunk)
        return [c for c in result if c.strip()]

    def chunk_document(self, doc: dict) -> list[dict]:
        """Разбить документ (dict с 'text' и 'metadata') на семантические чанки.

        Args:
            doc: Словарь с ключами 'text' и 'metadata'.

        Returns:
            Список чанков с preserved metadata и полями chunk_index/chunk_total.
        """
        raw_chunks = self.chunk(doc["text"])
        result = []
        for i, text in enumerate(raw_chunks):
            result.append(
                {
                    "text": text,
                    "metadata": {
                        **doc.get("metadata", {}),
                        "chunk_index": i,
                        "chunk_total": len(raw_chunks),
                        "chunking_strategy": "semantic",
                    },
                }
            )
        return result
