"""
Streaming RAG generation via SSE (Server-Sent Events).

Паттерн 2026: token-by-token стриминг через Anthropic streaming API.
Faithfulness gate применяется ПОСЛЕ завершения стрима (на полном ответе).

Graceful degradation: без ANTHROPIC_API_KEY симулирует стриминг побуквенно
для демонстрации механики в CI и dev-окружении без ключа.

Источники:
- FastAPI SSE: https://fastapi.tiangolo.com/tutorial/server-sent-events/
- Anthropic streaming API: https://docs.anthropic.com/en/docs/build-with-claude/streaming
- dasroot.net: Streaming RAG Token-Level Citations 2026
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator

from .chain import SYSTEM_PROMPT, build_prompt
from .faithfulness_gate import check_faithfulness


def _sse(event: dict) -> str:
    """Форматирует словарь как SSE-строку: 'data: {...}\\n\\n'."""
    return f"data: {json.dumps(event)}\n\n"


async def stream_answer(
    query: str,
    context_chunks: list[dict],
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
    faithfulness_threshold: float = 0.5,
) -> AsyncGenerator[str, None]:
    """Async SSE generator для streaming RAG ответов.

    Yields SSE события:
    - ``{"type": "token", "text": "..."}`` — фрагмент текста от LLM
    - ``{"type": "sources", "sources": [...]}`` — список источников (после токенов)
    - ``{"type": "done", "confidence": 0.8, "is_faithful": True, ...}`` — финальный ивент
    - ``{"type": "error", "message": "..."}`` — при ошибке API

    Без ANTHROPIC_API_KEY: симулирует стриминг по словам из mock-ответа (CI-friendly).

    Args:
        query: Вопрос пользователя.
        context_chunks: Retrieved чанки из vector store.
        model: Модель для стриминга (Haiku — быстрее для latency-чувствительного UX).
        max_tokens: Максимум токенов ответа.
        faithfulness_threshold: Порог is_faithful для финального события.

    Yields:
        SSE-строки вида ``data: {json}\\n\\n``.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    sources = list({c.get("metadata", {}).get("source", "") for c in context_chunks})
    full_answer = ""

    if not api_key:
        mock_answer = (
            "This is a mock streaming response. "
            "No ANTHROPIC_API_KEY is set — in production, "
            "Claude streams tokens here token-by-token."
        )
        for word in mock_answer.split():
            token = word + " "
            full_answer += token
            yield _sse({"type": "token", "text": token})
    else:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)
        prompt = build_prompt(query, context_chunks)

        try:
            with client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text_chunk in stream.text_stream:
                    full_answer += text_chunk
                    yield _sse({"type": "token", "text": text_chunk})
        except Exception as exc:  # noqa: BLE001
            yield _sse({"type": "error", "message": str(exc)})
            return

    # Источники — отправляем сразу после стрима (до faithfulness, чтоб UI не ждал)
    yield _sse({"type": "sources", "sources": sources})

    # Faithfulness gate на собранном полном ответе
    contexts = [c["text"] for c in context_chunks]
    gate = check_faithfulness(
        answer=full_answer,
        contexts=contexts,
        threshold=faithfulness_threshold,
    )

    yield _sse(
        {
            "type": "done",
            "confidence": gate.score,
            "is_faithful": gate.is_faithful,
            "faithfulness_method": gate.method,
            "faithfulness_verdict": gate.verdict,
        }
    )
