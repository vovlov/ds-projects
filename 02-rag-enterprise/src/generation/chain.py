"""RAG generation chain using Anthropic Claude API."""

from __future__ import annotations

import os

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only use information from the provided context to answer questions.
2. If the context doesn't contain enough information, say so clearly.
3. Cite the source document when possible.
4. Be concise and precise.
5. Answer in the same language as the question."""


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """Build a prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")

    context_text = "\n\n---\n\n".join(context_parts)

    return f"""Context:
{context_text}

Question: {query}

Answer based on the context above:"""


def generate_answer(
    query: str,
    context_chunks: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
) -> dict:
    """Generate answer using Claude API with retrieved context."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {
            "answer": "Error: ANTHROPIC_API_KEY not set. Please set it in your environment.",
            "model": model,
            "sources": [],
        }

    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    prompt = build_prompt(query, context_chunks)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    answer = response.content[0].text
    sources = list({c.get("metadata", {}).get("source", "") for c in context_chunks})

    return {
        "answer": answer,
        "model": model,
        "sources": sources,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }
