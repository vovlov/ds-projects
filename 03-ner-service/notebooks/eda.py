# %% [markdown]
# # NER Service — Entity Analysis
#
# Analyze rule-based NER performance on sample Russian texts.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

from ner.data.dataset import NER_LABELS, get_sample_data
from ner.model.ner import extract_entities_from_bio, predict

# %% [markdown]
# ## 1. Sample Data Statistics

# %%
data = get_sample_data()
print(f"Samples: {len(data)}")

label_counts = {label: 0 for label in NER_LABELS if label != "O"}
for item in data:
    for label in item["labels"]:
        if label != "O":
            label_counts[label] = label_counts.get(label, 0) + 1

print("\nLabel distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count}")

# %% [markdown]
# ## 2. BIO Entity Extraction

# %%
for item in data:
    entities = extract_entities_from_bio(item["tokens"], item["labels"])
    text = " ".join(item["tokens"])
    ents = [(e.text, e.label) for e in entities]
    print(f"{text}")
    print(f"  → {ents}\n")

# %% [markdown]
# ## 3. Rule-Based NER Performance

# %%
test_texts = [
    "Владимир Путин посетил Москву и встретился с представителями Газпрома.",
    "Компания Яндекс открыла новый офис в Санкт-Петербурге.",
    "Сбербанк заключил соглашение с ПАО Газпром в Екатеринбурге.",
    "Илон Маск является CEO компании Tesla в США.",
    "Мария Иванова работает в Amazon в Германии.",
    "МВД России провело операцию в Краснодаре совместно с ФСБ.",
    "Дмитрий Медведев встретился с Ангелой Меркель в Берлине.",
]

for text in test_texts:
    entities = predict(text)
    print(f"\n{text}")
    if entities:
        for e in entities:
            print(f"  [{e.label}] \"{e.text}\" (pos {e.start}-{e.end})")
    else:
        print("  No entities found")

# %% [markdown]
# ## 4. Coverage Analysis

# %%
total = 0
found = 0
for text in test_texts:
    entities = predict(text)
    # Count expected entities manually
    words = text.split()
    total += sum(1 for w in words if w[0].isupper() and len(w) > 2)
    found += len(entities)

print(f"Capitalized words: {total}")
print(f"Entities detected: {found}")
print(f"Detection rate: {found/total:.1%}" if total > 0 else "N/A")

# %% [markdown]
# ## Key Findings
#
# 1. **Rule-based NER** detects well-known entities (Газпром, Яндекс, Москва)
# 2. **Limitations:** Cannot detect unknown names or declined forms
# 3. **PER detection** works for "Имя Фамилия" patterns
# 4. **Transformer model** needed for production-quality NER
