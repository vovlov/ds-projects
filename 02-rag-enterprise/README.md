# 02 — RAG Enterprise: Document Q&A System

**Система вопросов-ответов по корпоративным документам** на базе Retrieval-Augmented Generation. Загружаешь документы — задаёшь вопросы на естественном языке — получаешь ответы с указанием источника.

*Enterprise document Q&A system powered by RAG. Upload documents, ask questions in natural language, get answers with source attribution.*

> **Эволюция:** В Практикуме я работал с [TF-IDF и классификацией текстов](https://github.com/vovlov/YandexPraktikum/tree/master/project_13_ML_for_text). Здесь — полноценный RAG-пайплайн: семантический поиск по векторной базе + генерация через Claude API.

## Бизнес-задача

В любой компании — сотни внутренних документов: политики, регламенты, онбординг. Сотрудники тратят часы на поиск нужной информации. RAG-система позволяет задать вопрос и получить точный ответ за секунды.

## Архитектура

```
  Документы (.txt, .md)
        │
        ▼
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  Chunking   │────▶│  ChromaDB   │────▶│  Retrieval  │
  │  512 chars  │     │  Embeddings │     │  Top-K      │
  │  64 overlap │     │  Cosine     │     │  Semantic   │
  └─────────────┘     └─────────────┘     └──────┬──────┘
                                                  │
                                           Вопрос │ Контекст
                                                  ▼
                                        ┌──────────────────┐
                                        │  Claude API      │
                                        │  + Guardrails    │
                                        │  + Source citing  │
                                        └────────┬─────────┘
                                                 │
                                        ┌────────▼─────────┐
                                        │  Gradio Chat UI  │
                                        │  :7860           │
                                        └──────────────────┘
```

## Быстрый старт

```bash
make setup-rag

# Установить API-ключ
export ANTHROPIC_API_KEY="sk-ant-..."

# Добавить документы в data/documents/ (.txt или .md)

# Запуск
cd 02-rag-enterprise
uv run python -m src.api.app

# Открыть http://localhost:7860
```

## API

```bash
# Индексация документов
curl -X POST http://localhost:7860/index
# → {"indexed_chunks": 12, "documents": 2}

# Health check
curl http://localhost:7860/health
```

## Как это работает

1. **Ingestion:** Документы из `data/documents/` разбиваются на чанки (512 символов, 64 overlap) с помощью LangChain RecursiveCharacterTextSplitter
2. **Embedding:** ChromaDB индексирует чанки, используя встроенную модель эмбеддингов
3. **Retrieval:** Вопрос пользователя эмбеддится и ищется по cosine similarity среди чанков
4. **Generation:** Top-K чанков собираются в промпт и отправляются в Claude API с системным промптом (используй только контекст, цитируй источники)
5. **Response:** Ответ с указанием документов-источников

## Стек

| Компонент | Инструмент | Зачем |
|-----------|-----------|-------|
| Chunking | LangChain Text Splitters | Рекурсивное разбиение с учётом структуры текста |
| Векторная БД | ChromaDB | Лёгкая, встраиваемая, persistence из коробки |
| LLM | Claude API (Anthropic) | Качество генерации, длинный контекст |
| UI | Gradio | Быстрый чат-интерфейс без фронтенда |
| API | FastAPI | REST endpoints для индексации и health check |
| Конфиг | YAML | chunk_size, n_results, модель — не захардкожены |
| Тесты | pytest (11 тестов) | Ingestion, retrieval relevance, prompt building |
