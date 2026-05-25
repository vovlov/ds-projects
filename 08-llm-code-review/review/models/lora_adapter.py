"""LoRA Adapter для domain-specific fine-tuning TF-IDF classifier.

Реализует Low-Rank Adaptation (Hu et al. 2021, arxiv 2106.09685) применительно
к sklearn TF-IDF + LogisticRegression пайплайну:

  W_adapted = W_base + alpha/rank * A @ B

Где A ∈ R^(d×r) и B ∈ R^(r×k), r << d (rank << vocab_size).
Базовая модель заморожена — обучаются только A и B.

Обучение: gradient descent с cross-entropy loss (numpy-only, без PyTorch).

Практический эффект:
- security-adapter: усиливает веса для security-терминов в vocab
- performance-adapter: фокусируется на performance vocabulary
- По аналогии с LoRA Land (arxiv 2405.00732): domain-specific адаптация
  без полного переобучения базовой модели.

Sources:
- Hu et al. 2021 LoRA (arxiv 2106.09685)
- LoRA Land: 310 Fine-tuned LLMs (arxiv 2405.00732)
- Serving Heterogeneous LoRA Adapters (arxiv 2511.22880)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class LoRAConfig:
    """Конфигурация LoRA адаптера.

    LoRA adapter configuration.
    """

    rank: int = 4
    """Ранг низкоранговой аппроксимации (r << d). / Low-rank bottleneck dimension."""

    alpha: float = 16.0
    """Масштабирующий коэффициент (effective scale = alpha/rank). / LoRA scaling factor."""

    n_epochs: int = 100
    """Число эпох gradient descent по A и B. / Training epochs for A and B matrices."""

    learning_rate: float = 0.05
    """Шаг обучения для gradient descent. / Learning rate for matrix updates."""

    target_domain: str = "general"
    """Домен адаптации (security/bug/performance/style/documentation/general)."""

    l2_reg: float = 1e-4
    """L2 регуляризация для предотвращения переобучения на малых датасетах."""


@dataclass
class AdapterResult:
    """Результат классификации с активным LoRA адаптером.

    Classification result using the adapted model.
    """

    category: str
    confidence: float
    base_confidence: float
    """Уверенность базовой модели (без адаптера). / Confidence without adapter."""
    adaptation_delta: float
    """Изменение уверенности благодаря адаптеру (+/- = адаптер помог/навредил)."""
    domain: str
    all_probabilities: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "base_confidence": round(self.base_confidence, 3),
            "adaptation_delta": round(self.adaptation_delta, 3),
            "domain": self.domain,
            "all_probabilities": {k: round(v, 3) for k, v in self.all_probabilities.items()},
        }


@dataclass
class AdapterTrainResult:
    """Результат обучения LoRA адаптера.

    Training result with loss history.
    """

    domain: str
    rank: int
    alpha: float
    n_examples: int
    n_epochs: int
    final_loss: float
    initial_loss: float

    @property
    def loss_reduction(self) -> float:
        """Снижение loss за всё обучение (initial - final)."""
        return self.initial_loss - self.final_loss

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "rank": self.rank,
            "alpha": self.alpha,
            "n_examples": self.n_examples,
            "n_epochs": self.n_epochs,
            "final_loss": round(self.final_loss, 4),
            "initial_loss": round(self.initial_loss, 4),
            "loss_reduction": round(self.loss_reduction, 4),
        }


def is_available() -> bool:
    """Проверить наличие зависимостей (numpy + sklearn всегда есть)."""
    try:
        import numpy  # noqa: F401
        from sklearn.pipeline import Pipeline  # noqa: F401

        return True
    except ImportError:
        return False


class LoRAAdapter:
    """Low-Rank Adaptation поверх TF-IDF + LogisticRegression.

    Замораживает базовую модель и обучает малые матрицы A (d×r) и B (r×k),
    добавляя domain-specific смещение весов без изменения базовых параметров.

    Low-Rank Adaptation on top of a frozen sklearn classifier.
    Only A and B are updated; the base TF-IDF + LogReg pipeline is read-only.
    """

    def __init__(self, base_pipeline, config: LoRAConfig | None = None):
        """
        Args:
            base_pipeline: обученный sklearn Pipeline (tfidf + clf).
            config: гиперпараметры адаптера (rank, alpha, epochs, lr).
        """
        self.config = config or LoRAConfig()
        self._pipeline = base_pipeline
        self._A: "Optional[np.ndarray]" = None  # shape: (n_features, rank)
        self._B: "Optional[np.ndarray]" = None  # shape: (rank, n_classes)
        self._fitted = False
        self._train_result: Optional[AdapterTrainResult] = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_base_weights(self) -> tuple:
        """Извлечь W_base.T и b_base из LogisticRegression.

        sklearn хранит coef_ как (n_classes, n_features), транспонируем в (d, k).
        """
        import numpy as np

        clf = self._pipeline.named_steps["clf"]
        W = clf.coef_.T.astype(np.float64)  # (d, k)
        b = clf.intercept_.astype(np.float64)  # (k,)
        return W, b

    def _tfidf_transform(self, texts: list[str]) -> "np.ndarray":
        """Применить только TF-IDF часть пайплайна."""
        tfidf = self._pipeline.named_steps["tfidf"]
        return tfidf.transform(texts).toarray()

    @property
    def _classes(self) -> list[str]:
        return list(self._pipeline.named_steps["clf"].classes_)

    @staticmethod
    def _softmax(logits: "np.ndarray") -> "np.ndarray":
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_x = __import__("numpy").exp(shifted)
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    @staticmethod
    def _cross_entropy(proba: "np.ndarray", y: "np.ndarray", n_classes: int) -> float:
        import numpy as np

        one_hot = np.eye(n_classes)[y]
        return float(-np.mean(np.sum(one_hot * np.log(proba + 1e-10), axis=1)))

    def _forward(
        self, X: "np.ndarray", W_base: "np.ndarray", b_base: "np.ndarray"
    ) -> "np.ndarray":
        """Прямой проход: logits = X @ (W_base + scale*A@B) + b."""
        import numpy as np

        scale = self.config.alpha / self.config.rank
        W_delta = scale * self._A @ self._B  # (d, k)
        logits = X @ (W_base + W_delta) + b_base  # (N, k)
        return logits

    # ── Public API ─────────────────────────────────────────────────────────────

    def fit(self, texts: list[str], labels: list[str]) -> AdapterTrainResult:
        """Обучить матрицы A и B на domain-specific примерах.

        Обучает только A ∈ R^(d×r) и B ∈ R^(r×k) методом gradient descent.
        W_base заморожен — только domain-specific смещение обновляется.

        Train LoRA matrices A and B via gradient descent with cross-entropy loss.
        The base model weights are frozen throughout.

        Args:
            texts: список code review комментариев для fine-tuning.
            labels: соответствующие категории (bug/security/performance/style/documentation).

        Returns:
            AdapterTrainResult с метриками обучения.
        """
        import numpy as np

        if not texts:
            raise ValueError("Need at least 1 example to fit the adapter")

        W_base, b_base = self._get_base_weights()
        n_features, n_classes = W_base.shape
        classes = self._classes

        # Encode labels → integer indices
        label_to_idx = {c: i for i, c in enumerate(classes)}
        y = np.array([label_to_idx.get(lbl, 0) for lbl in labels])

        # TF-IDF features (N, d)
        X = self._tfidf_transform(texts)

        # Initialize A ~ N(0, 0.01), B = 0  (LoRA init: Hu et al. 2021)
        rng = np.random.default_rng(42)
        self._A = rng.normal(0, 0.01, (n_features, self.config.rank))
        self._B = np.zeros((self.config.rank, n_classes))

        scale = self.config.alpha / self.config.rank
        initial_loss: float | None = None
        final_loss = 0.0

        for epoch in range(self.config.n_epochs):
            # Forward pass
            H = X @ self._A  # (N, r)
            logits = H @ self._B * scale + X @ W_base + b_base  # (N, k)
            proba = self._softmax(logits)

            # Loss
            loss = self._cross_entropy(proba, y, n_classes)
            if initial_loss is None:
                initial_loss = loss
            final_loss = loss

            # Gradients (dL/d_logits = (proba - one_hot) / N)
            one_hot = np.eye(n_classes)[y]
            d_logits = (proba - one_hot) / len(texts)  # (N, k)

            # dL/dB = scale * H.T @ d_logits
            d_B = scale * H.T @ d_logits  # (r, k)
            # dL/dA = scale * X.T @ (d_logits @ B.T)
            d_A = scale * X.T @ (d_logits @ self._B.T)  # (d, r)

            # L2 regularisation keeps matrices small on tiny datasets
            d_B += self.config.l2_reg * self._B
            d_A += self.config.l2_reg * self._A

            # Gradient descent step
            self._B -= self.config.learning_rate * d_B
            self._A -= self.config.learning_rate * d_A

        self._fitted = True
        self._train_result = AdapterTrainResult(
            domain=self.config.target_domain,
            rank=self.config.rank,
            alpha=self.config.alpha,
            n_examples=len(texts),
            n_epochs=self.config.n_epochs,
            final_loss=final_loss,
            initial_loss=float(initial_loss or final_loss),
        )
        return self._train_result

    def predict(self, text: str) -> AdapterResult:
        """Классифицировать текст с активным LoRA адаптером.

        Classify a review comment using the adapted model.
        Returns both adapted and base predictions for comparison.

        Args:
            text: review comment text to classify.

        Returns:
            AdapterResult с category, confidence, base_confidence и adaptation_delta.
        """
        import numpy as np

        if not self._fitted:
            raise RuntimeError("Adapter not fitted — call fit() first")

        W_base, b_base = self._get_base_weights()
        X = self._tfidf_transform([text])  # (1, d)
        classes = self._classes

        # Base prediction (no adapter)
        logits_base = X @ W_base + b_base
        proba_base = self._softmax(logits_base)[0]
        base_pred_idx = int(np.argmax(proba_base))
        base_confidence = float(proba_base[base_pred_idx])

        # Adapted prediction
        logits_adapted = self._forward(X, W_base, b_base)
        proba_adapted = self._softmax(logits_adapted)[0]
        pred_idx = int(np.argmax(proba_adapted))
        confidence = float(proba_adapted[pred_idx])

        category = classes[pred_idx]
        delta = confidence - (float(proba_base[pred_idx]) if pred_idx == base_pred_idx else base_confidence)

        return AdapterResult(
            category=category,
            confidence=confidence,
            base_confidence=base_confidence,
            adaptation_delta=round(delta, 4),
            domain=self.config.target_domain,
            all_probabilities={c: round(float(p), 3) for c, p in zip(classes, proba_adapted)},
        )

    def predict_batch(self, texts: list[str]) -> list[AdapterResult]:
        """Классифицировать список текстов. / Classify a batch of texts."""
        return [self.predict(t) for t in texts]

    @property
    def is_fitted(self) -> bool:
        """Проверить, обучен ли адаптер. / Whether the adapter has been trained."""
        return self._fitted

    @property
    def train_result(self) -> Optional[AdapterTrainResult]:
        """Результат последнего обучения. / Last training result."""
        return self._train_result

    def adapter_norm(self) -> float:
        """Фробениусова норма матриц адаптера (мера "силы" адаптации).

        Frobenius norm of adapter matrices — proxy for adaptation magnitude.
        """
        if not self._fitted:
            return 0.0
        import numpy as np

        norm_a = float(np.linalg.norm(self._A, "fro"))
        norm_b = float(np.linalg.norm(self._B, "fro"))
        return round(norm_a + norm_b, 4)

    def save(self, path: str | Path) -> None:
        """Сохранить матрицы A и B в JSON.

        Save adapter matrices to JSON (A and B as nested lists).
        """
        import numpy as np

        if not self._fitted:
            raise RuntimeError("Nothing to save — adapter not fitted")
        data = {
            "A": self._A.tolist(),
            "B": self._B.tolist(),
            "config": {
                "rank": self.config.rank,
                "alpha": self.config.alpha,
                "target_domain": self.config.target_domain,
            },
        }
        Path(path).write_text(json.dumps(data))

    def load(self, path: str | Path) -> None:
        """Загрузить матрицы A и B из JSON.

        Load adapter matrices from a previously saved JSON file.
        """
        import numpy as np

        data = json.loads(Path(path).read_text())
        self._A = np.array(data["A"])
        self._B = np.array(data["B"])
        self._fitted = True
