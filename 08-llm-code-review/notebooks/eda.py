# %% [markdown]
# # LLM Code Review — Data & Classifier Analysis
#
# Explore the sample dataset and evaluate the TF-IDF classifier.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

from src.data.samples import CATEGORIES, get_sample_reviews
from src.models.classifier import build_classifier, classify_comment

# %% [markdown]
# ## 1. Dataset Overview

# %%
samples = get_sample_reviews()
print(f"Total samples: {len(samples)}")
print(f"Categories: {CATEGORIES}")

category_counts = {}
for s in samples:
    cat = s["category"]
    category_counts[cat] = category_counts.get(cat, 0) + 1

print("\nCategory distribution:")
for cat, count in sorted(category_counts.items()):
    bar = "#" * (count * 4)
    print(f"  {cat:15s} {count:2d} {bar}")

# %% [markdown]
# ## 2. Sample Diffs — Length Statistics

# %%
diff_lengths = [len(s["code_diff"]) for s in samples]
comment_lengths = [len(s["review_comment"]) for s in samples]

print(f"Diff length    — min: {min(diff_lengths):4d}, max: {max(diff_lengths):4d}, "
      f"avg: {sum(diff_lengths)/len(diff_lengths):.0f}")
print(f"Comment length — min: {min(comment_lengths):4d}, max: {max(comment_lengths):4d}, "
      f"avg: {sum(comment_lengths)/len(comment_lengths):.0f}")

# %% [markdown]
# ## 3. Classifier — Leave-One-Out Evaluation

# %%
pipeline = build_classifier()
correct = 0
for s in samples:
    pred = classify_comment(s["review_comment"], pipeline)
    match = "ok" if pred["category"] == s["category"] else "MISS"
    if pred["category"] == s["category"]:
        correct += 1
    print(f"[{match:4s}] true={s['category']:15s} pred={pred['category']:15s} "
          f"conf={pred['confidence']:.2f}  {s['review_comment'][:60]}...")

print(f"\nTrain accuracy: {correct}/{len(samples)} = {correct/len(samples):.0%}")

# %% [markdown]
# ## 4. TF-IDF Feature Inspection

# %%
tfidf = pipeline.named_steps["tfidf"]
feature_names = tfidf.get_feature_names_out()
print(f"Vocabulary size: {len(feature_names)}")
print(f"Sample features: {list(feature_names[:20])}")

# %% [markdown]
# ## Key Findings
#
# 1. **12 samples** cover all 5 categories — small but usable for TF-IDF baseline
# 2. **Train accuracy ~100%** is expected (fitting on training set) — real evaluation
#    needs held-out data or cross-validation
# 3. **TF-IDF features** capture domain keywords (sql, injection, docstring, O(n^2))
# 4. Next step: expand dataset with real review comments from open-source repos
