# %% [markdown]
# # RAG Enterprise — Chunking & Retrieval Analysis
#
# Analyze document chunking strategies and retrieval quality.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import contextlib

from src.ingestion.loader import chunk_documents, load_documents
from src.retrieval.store import get_client, get_or_create_collection, index_chunks, search

DATA_DIR = Path.cwd().parent / "data" / "documents"

# %%
docs = load_documents(DATA_DIR)
print(f"Loaded {len(docs)} documents:")
for doc in docs:
    print(f"  - {doc['metadata']['source']}: {len(doc['text'])} chars")

# %% [markdown]
# ## 1. Chunking Analysis — Impact of Chunk Size

# %%
for chunk_size in [128, 256, 512, 1024]:
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=64)
    lengths = [len(c["text"]) for c in chunks]
    avg_len = sum(lengths) / len(lengths)
    print(f"chunk_size={chunk_size:4d} → {len(chunks):3d} chunks, avg length: {avg_len:.0f} chars")

# %% [markdown]
# ## 2. Retrieval Quality Test

# %%
chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=64)
client = get_client(Path("/tmp/eda_chroma"))
with contextlib.suppress(Exception):
    client.delete_collection("eda_test")
collection = get_or_create_collection(client, "eda_test")
index_chunks(chunks, collection)

print(f"Indexed {collection.count()} chunks")

# %%
test_queries = [
    "What are the core working hours for remote employees?",
    "What equipment does the company provide?",
    "How does onboarding work in the first week?",
    "What tools does the company use?",
    "What is the VPN policy?",
]

for query in test_queries:
    results = search(query, collection, n_results=3)
    print(f"\n{'='*60}")
    print(f"Q: {query}")
    for i, r in enumerate(results):
        source = r["metadata"].get("source", "?")
        dist = r.get("distance", "?")
        print(f"  [{i+1}] (dist={dist:.3f}) [{source}] {r['text'][:80]}...")

# %% [markdown]
# ## Key Findings
#
# 1. **512-char chunks** provide good balance between context and precision
# 2. **Cosine distance** < 0.5 indicates strong semantic match
# 3. **Source attribution** works correctly — answers trace back to specific documents
# 4. **Policy vs Onboarding** queries correctly route to respective documents
