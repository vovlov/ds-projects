"""RAG generation chain using Anthropic Claude API."""

from __future__ import annotations

import os

from .faithfulness_gate import FaithfulnessResult, check_faithfulness

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


def generate_answer_with_gate(
    query: str,
    context_chunks: list[dict],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    faithfulness_threshold: float = 0.5,
) -> dict:
    """Agentic RAG: генерация ответа + проверка верности источникам.

    Двухэтапный пайплайн:
    1. Основная генерация (generate_answer) — ответ по retrieved-контексту.
    2. Faithfulness gate — проверяет, что ответ поддержан чанками.
       В production используется второй LLM-вызов (Haiku).
       В CI/без API — лексическая эвристика.

    confidence_score = faithfulness gate score (0.0 — 1.0).
    Ниже threshold → is_faithful=False → пользователю нужно показать предупреждение.

    Args:
        query: Вопрос пользователя.
        context_chunks: Retrieved чанки из vector store.
        model: Основная модель для генерации ответа.
        max_tokens: Максимум токенов в ответе.
        faithfulness_threshold: Порог is_faithful (default 0.5).

    Returns:
        Словарь с ключами: answer, model, sources, usage,
        confidence_score, is_faithful, faithfulness_method.
    """
    result = generate_answer(query, context_chunks, model=model, max_tokens=max_tokens)

    contexts = [c["text"] for c in context_chunks]
    gate: FaithfulnessResult = check_faithfulness(
        answer=result["answer"],
        contexts=contexts,
        threshold=faithfulness_threshold,
    )

    return {
        **result,
        "confidence_score": gate.score,
        "is_faithful": gate.is_faithful,
        "faithfulness_method": gate.method,
        "faithfulness_verdict": gate.verdict,
    }
