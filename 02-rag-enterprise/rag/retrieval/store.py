"""
Векторное хранилище на базе ChromaDB.

ChromaDB выбрана за простоту: встраиваемая (без отдельного сервера), persistence
из коробки, cosine similarity по умолчанию. Для production с >100K документов
лучше Qdrant или Pinecone, но для портфолио и малых корпоративных баз — идеально.

Эмбеддинги: ChromaDB использует свою встроенную модель (all-MiniLM-L6-v2).
Для русскоязычных документов в production стоит заменить на multilingual-e5-large.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings

PERSIST_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma_db"


def get_client(persist_dir: Path = PERSIST_DIR) -> chromadb.ClientAPI:
    """Создать клиент ChromaDB с сохранением на диск."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.Client(
        Settings(
            persist_directory=str(persist_dir),
            anonymized_telemetry=False,
        )
    )


def get_or_create_collection(
    client: chromadb.ClientAPI,
    name: str = "documents",
) -> chromadb.Collection:
    """Получить или создать коллекцию с cosine-метрикой."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def index_chunks(
    chunks: list[dict],
    collection: chromadb.Collection,
) -> int:
    """Проиндексировать чанки документов.

    ChromaDB имеет лимит ~5000 объектов за batch,
    поэтому разбиваем на порции по 500.
    """
    if not chunks:
        return 0

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    batch_size = 500
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )

    return len(chunks)


def search(
    query: str,
    collection: chromadb.Collection,
    n_results: int = 5,
) -> list[dict]:
    """Семантический поиск: найти top-k чанков, ближайших к запросу.

    Возвращает список с text, metadata и distance (cosine distance, чем меньше — тем лучше).
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    hits = []
    if results and results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            hits.append(
                {
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else None,
                }
            )
    return hits
