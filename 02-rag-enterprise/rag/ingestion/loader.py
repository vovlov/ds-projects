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
) -> list[dict]:
    """Split documents into chunks using recursive character splitting."""
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
                    },
                }
            )
    return chunks
