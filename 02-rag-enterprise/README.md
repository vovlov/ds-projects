# 02 — RAG Enterprise: Document Q&A System

> **Evolution from:** [Yandex.Praktikum Project 13 (ML for Text)](https://github.com/vovlov/YandexPraktikum/tree/master/project_13_ML_for_text) — from TF-IDF classification to Retrieval-Augmented Generation

Production-ready RAG pipeline for enterprise document Q&A — ingest documents, chunk and embed them, retrieve relevant context, and generate accurate answers using Claude API.

## Architecture

```
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Documents   │────▶│  Chunking      │────▶│  ChromaDB    │
│  (.txt, .md) │     │  (Recursive)   │     │  (Embeddings)│
└──────────────┘     └────────────────┘     └──────┬───────┘
                                                    │
                                              Query │ Retrieve
                                                    ▼
┌──────────────┐     ┌────────────────┐     ┌──────────────┐
│  Gradio UI   │◀────│  Claude API    │◀────│  Context     │
│  :7860       │     │  (Generation)  │     │  Assembly    │
└──────────────┘     └────────────────┘     └──────────────┘
```

## Quick Start

```bash
# From repo root
make setup-rag

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Add documents to data/documents/ (txt, md files)

# Run
cd 02-rag-enterprise
uv run python -m src.api.app

# Open http://localhost:7860
```

## API

```bash
# Index documents
curl -X POST http://localhost:7860/index

# Health check
curl http://localhost:7860/health
```

## Stack

| Component | Tool |
|-----------|------|
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Vector store | ChromaDB (default embeddings) |
| LLM | Claude API (Anthropic) |
| UI | Gradio |
| API | FastAPI |
| Containerization | Docker |

## How It Works

1. **Ingestion**: Documents are loaded from `data/documents/`, split into chunks (512 chars, 64 overlap)
2. **Embedding**: ChromaDB embeds chunks using its default embedding function
3. **Retrieval**: User query is embedded and matched against stored chunks via cosine similarity
4. **Generation**: Top-k chunks are assembled into a prompt and sent to Claude API
5. **Response**: Answer is generated with source attribution
