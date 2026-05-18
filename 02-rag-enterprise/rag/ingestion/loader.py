"""Document loading and chunking for RAG pipeline."""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(data_dir: Path) -> list[dict]:
    """Load text documents from directory."""
    docs = []
    for ext in ("*.txt", "*.md"):
        for file_path in sorted(data_dir.glob(ext)):
            text = file_path.read_text(encoding="utf-8")
            docs.append({"text": text, "metadata": {"source": file_path.name}})
    return docs


def chunk_documents(
    docs: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    chunking_strategy: str = "fixed",
) -> list[dict]:
    """Split documents into chunks.

    Args:
        docs: List of document dicts with 'text' and 'metadata'.
        chunk_size: Target chunk size in characters (used by 'fixed' strategy).
        chunk_overlap: Overlap between consecutive fixed chunks.
        chunking_strategy: One of:
            - "fixed"     — RecursiveCharacterTextSplitter (default, deterministic)
            - "semantic"  — TF-IDF cosine boundary detection (requires sklearn)
            - "paragraph" — Split on double newlines, merge small paragraphs

    Returns:
        List of chunk dicts with 'text' and 'metadata' (including chunk_index).
    """
    if chunking_strategy == "semantic":
        return _chunk_semantic(docs)
    if chunking_strategy == "paragraph":
        return _chunk_paragraph(docs, max_chars=chunk_size)
    return _chunk_fixed(docs, chunk_size, chunk_overlap)


def _chunk_fixed(docs: list[dict], chunk_size: int, chunk_overlap: int) -> list[dict]:
    """Fixed-size chunking via RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(splits):
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "chunk_total": len(splits),
                        "chunking_strategy": "fixed",
                    },
                }
            )
    return chunks


def _chunk_semantic(docs: list[dict]) -> list[dict]:
    """Semantic chunking via TF-IDF cosine similarity boundaries."""
    from ..chunking.semantic import SemanticChunker

    chunker = SemanticChunker()
    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk_document(doc))
    return chunks


def _chunk_paragraph(docs: list[dict], max_chars: int = 1200) -> list[dict]:
    """Paragraph-based chunking: split on double newlines."""
    from ..chunking.semantic import _paragraph_chunks

    chunks = []
    for doc in docs:
        raw = _paragraph_chunks(doc["text"], max_chars=max_chars)
        for i, text in enumerate(raw):
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": i,
                        "chunk_total": len(raw),
                        "chunking_strategy": "paragraph",
                    },
                }
            )
    return chunks
