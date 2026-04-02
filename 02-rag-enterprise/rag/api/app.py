"""FastAPI + Gradio app for RAG Enterprise."""

from __future__ import annotations

import contextlib
from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

from ..generation.chain import generate_answer, generate_answer_with_gate
from ..ingestion.loader import chunk_documents, load_documents
from ..retrieval.store import get_client, get_or_create_collection, index_chunks, search

app = FastAPI(title="RAG Enterprise API", version="1.0.0")

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"

# Global state
_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = get_or_create_collection(client)

        # Auto-index if collection is empty
        if _collection.count() == 0 and DATA_DIR.exists():
            docs = load_documents(DATA_DIR)
            if docs:
                chunks = chunk_documents(docs)
                index_chunks(chunks, _collection)
    return _collection


class QueryRequest(BaseModel):
    """Запрос к RAG-пайплайну."""

    question: str
    n_results: int = 5
    check_faithfulness: bool = True
    faithfulness_threshold: float = 0.5


class QueryResponse(BaseModel):
    """Ответ RAG-пайплайна с оценкой верности источникам."""

    answer: str
    sources: list[str]
    confidence_score: float
    is_faithful: bool
    faithfulness_method: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Agentic RAG: ответ на вопрос + faithfulness gate.

    Возвращает ответ с confidence_score — оценкой того, насколько ответ
    поддержан retrieved документами. Низкий score сигнализирует о potential hallucination.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return QueryResponse(
            answer="No documents indexed. Please add documents to data/documents/ and run /index.",
            sources=[],
            confidence_score=0.0,
            is_faithful=False,
            faithfulness_method="lexical",
        )

    context = search(request.question, collection, n_results=request.n_results)

    if request.check_faithfulness:
        result = generate_answer_with_gate(
            query=request.question,
            context_chunks=context,
            faithfulness_threshold=request.faithfulness_threshold,
        )
    else:
        raw = generate_answer(request.question, context)
        result = {
            **raw,
            "confidence_score": 1.0,
            "is_faithful": True,
            "faithfulness_method": "none",
        }

    return QueryResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence_score=result["confidence_score"],
        is_faithful=result["is_faithful"],
        faithfulness_method=result["faithfulness_method"],
    )


@app.post("/index")
def index_documents():
    """Re-index all documents from data directory."""
    client = get_client()
    # Delete and recreate collection
    with contextlib.suppress(Exception):
        client.delete_collection("documents")

    global _collection
    _collection = get_or_create_collection(client)

    if not DATA_DIR.exists():
        return {"error": f"Data directory not found: {DATA_DIR}"}

    docs = load_documents(DATA_DIR)
    chunks = chunk_documents(docs)
    count = index_chunks(chunks, _collection)
    return {"indexed_chunks": count, "documents": len(docs)}


def ask(question: str, n_results: int = 5) -> str:
    """RAG pipeline: retrieve → generate → faithfulness gate."""
    if not question.strip():
        return "Please enter a question."

    collection = _get_collection()
    if collection.count() == 0:
        return "No documents indexed. Please add documents to data/documents/ and run /index."

    context = search(question, collection, n_results=n_results)
    result = generate_answer_with_gate(question, context)

    sources = ", ".join(result["sources"]) if result["sources"] else "N/A"
    confidence = result["confidence_score"]
    verdict = result["faithfulness_verdict"]
    method = result["faithfulness_method"]

    # Визуальный сигнал о надёжности ответа — низкий confidence предупреждает пользователя
    confidence_badge = "✅" if result["is_faithful"] else "⚠️"
    return (
        f"{result['answer']}\n\n"
        f"---\n"
        f"**Sources:** {sources}\n\n"
        f"**Confidence:** {confidence_badge} {confidence:.2f} ({verdict}, method: {method})"
    )


# Gradio interface
demo = gr.Interface(
    fn=ask,
    inputs=[
        gr.Textbox(label="Question", placeholder="Ask a question about the documents..."),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of context chunks"),
    ],
    outputs=gr.Markdown(label="Answer"),
    title="RAG Enterprise — Document Q&A",
    description="Ask questions about indexed documents. Powered by ChromaDB + Claude API.",
)

app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
