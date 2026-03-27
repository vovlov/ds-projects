"""Tests for RAG Enterprise pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contextlib

from src.generation.chain import build_prompt
from src.ingestion.loader import chunk_documents, load_documents
from src.retrieval.store import get_client, get_or_create_collection, index_chunks, search

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "documents"


class TestIngestion:
    def test_load_documents(self):
        docs = load_documents(DATA_DIR)
        assert len(docs) >= 2
        assert all("text" in d and "metadata" in d for d in docs)

    def test_load_documents_have_source(self):
        docs = load_documents(DATA_DIR)
        for doc in docs:
            assert "source" in doc["metadata"]
            assert doc["metadata"]["source"].endswith(".txt")

    def test_chunk_documents_default(self):
        docs = load_documents(DATA_DIR)
        chunks = chunk_documents(docs)
        assert len(chunks) > len(docs)  # Should produce more chunks than docs
        assert all("text" in c for c in chunks)

    def test_chunk_documents_preserves_metadata(self):
        docs = load_documents(DATA_DIR)
        chunks = chunk_documents(docs)
        for chunk in chunks:
            assert "source" in chunk["metadata"]
            assert "chunk_index" in chunk["metadata"]
            assert "chunk_total" in chunk["metadata"]

    def test_chunk_size_respected(self):
        docs = load_documents(DATA_DIR)
        chunk_size = 256
        chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=32)
        for chunk in chunks:
            # Allow some tolerance for splitting
            assert len(chunk["text"]) <= chunk_size + 50


class TestRetrieval:
    @pytest.fixture
    def indexed_collection(self):
        client = get_client(Path("/tmp/test_chroma_rag"))
        with contextlib.suppress(Exception):
            client.delete_collection("test_docs")
        collection = get_or_create_collection(client, name="test_docs")
        docs = load_documents(DATA_DIR)
        chunks = chunk_documents(docs)
        index_chunks(chunks, collection)
        return collection

    def test_index_chunks(self, indexed_collection):
        assert indexed_collection.count() > 0

    def test_search_returns_results(self, indexed_collection):
        results = search("remote work policy", indexed_collection, n_results=3)
        assert len(results) == 3
        assert all("text" in r for r in results)

    def test_search_relevance(self, indexed_collection):
        results = search("VPN security requirements", indexed_collection, n_results=3)
        # At least one result should mention VPN or security
        texts = " ".join(r["text"].lower() for r in results)
        assert "vpn" in texts or "security" in texts

    def test_search_onboarding(self, indexed_collection):
        results = search("first week onboarding schedule", indexed_collection, n_results=3)
        texts = " ".join(r["text"].lower() for r in results)
        assert "week" in texts or "onboarding" in texts or "day" in texts


class TestGeneration:
    def test_build_prompt_includes_context(self):
        chunks = [
            {"text": "Policy text here", "metadata": {"source": "policy.txt"}},
            {"text": "Another doc", "metadata": {"source": "guide.txt"}},
        ]
        prompt = build_prompt("What is the policy?", chunks)
        assert "Policy text here" in prompt
        assert "Another doc" in prompt
        assert "What is the policy?" in prompt

    def test_build_prompt_includes_sources(self):
        chunks = [{"text": "Some text", "metadata": {"source": "doc.txt"}}]
        prompt = build_prompt("Question?", chunks)
        assert "doc.txt" in prompt
