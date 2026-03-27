"""Train document classifier baseline (sklearn RandomForest)."""

from __future__ import annotations

import pickle
from pathlib import Path

from src.data.dataset import generate_synthetic_documents, get_feature_matrix
from src.models.classifier import train_classifier

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def main() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("Generating synthetic document features...")
    data = generate_synthetic_documents(n=500)
    X, y, label_encoder = get_feature_matrix(data)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {list(label_encoder.classes_)}")

    print("\nTraining RandomForest classifier...")
    result = train_classifier(X, y)

    print(f"\nAccuracy: {result['accuracy']:.3f}")
    print(f"\n{result['classification_report']}")

    with open(ARTIFACTS_DIR / "model.pkl", "wb") as f:
        pickle.dump(result["model"], f)
    with open(ARTIFACTS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
