"""LIME (Local Interpretable Model-Agnostic Explanations) for fraud detection.

Explains individual fraud predictions by fitting a local linear surrogate model
in the neighbourhood of the queried instance. Model-agnostic: works with any
callable that returns predict_proba(X)[:, 1].

Algorithm (Ribeiro et al. 2016 KDD):
1. Sample n_perturbations neighbours by adding Gaussian noise to the instance.
2. Compute kernel weights: exp(-dist² / σ²) — nearby samples matter more.
3. Fit weighted Ridge regression on (perturbation, black-box label) pairs.
4. Return regression coefficients as feature attributions.

References:
- Ribeiro et al. 2016 "Why Should I Trust You?" ACM KDD (arxiv:1602.04938)
- Guidotti et al. 2019 "A Survey of Methods for Explaining Black Box Models"
  ACM CSUR §3.2 (LIME review)
- EU AI Act Article 13 (traceability and transparency for high-risk AI)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class LIMEConfig:
    """Параметры LIME-объяснителя."""

    n_perturbations: int = 500
    kernel_width: float = 0.75  # σ для exp(-d²/σ²)
    n_features_in_explanation: int = 8
    seed: int = 42
    ridge_alpha: float = 1.0  # L2-регуляризация взвешенной Ridge regression


@dataclass
class LIMEFeatureContribution:
    """Вклад одного признака в локальное объяснение."""

    feature_name: str
    value: float  # фактическое значение признака в данной транзакции
    contribution: float  # коэффициент Ridge (знак = направление влияния)
    direction: str  # "increases_fraud_risk" / "decreases_fraud_risk" / "neutral"

    def to_dict(self) -> dict:
        return {
            "feature_name": self.feature_name,
            "value": round(self.value, 4),
            "contribution": round(self.contribution, 4),
            "direction": self.direction,
        }


@dataclass
class LIMEExplanation:
    """Объяснение одного предсказания мошенничества."""

    prediction: float  # P(fraud) оригинальной модели
    local_prediction: float  # P(fraud) локальной суррогатной модели
    local_fidelity: float  # 1 - |prediction - local_prediction| ∈ [0, 1]
    intercept: float  # смещение локальной модели
    top_features: list[LIMEFeatureContribution]  # топ-N по |contribution|
    n_perturbations: int
    method: str = "lime_ridge_regression"

    def to_dict(self) -> dict:
        return {
            "prediction": round(self.prediction, 4),
            "local_prediction": round(self.local_prediction, 4),
            "local_fidelity": round(self.local_fidelity, 4),
            "intercept": round(self.intercept, 4),
            "top_features": [f.to_dict() for f in self.top_features],
            "n_perturbations": self.n_perturbations,
            "method": self.method,
        }


class LIMEExplainer:
    """Model-agnostic local explanations for individual fraud predictions.

    Fits a weighted Ridge regression in the local neighbourhood of each
    transaction to approximate the black-box model behaviour at that point.
    Works with any predict_proba(X) → probabilities[:, 1] interface.

    Usage::

        explainer = LIMEExplainer(feature_names=["avg_amount", ...])
        explanation = explainer.explain(instance, predict_fn=model.predict_proba)
    """

    def __init__(
        self,
        feature_names: list[str],
        config: LIMEConfig | None = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.config = config or LIMEConfig()
        self._rng = np.random.default_rng(self.config.seed)

    # ── private helpers ──────────────────────────────────────────────────────

    def _perturb(self, instance: np.ndarray) -> np.ndarray:
        """Создать соседей вокруг instance через масштабированный Gaussian шум."""
        n = self.config.n_perturbations
        noise = self._rng.standard_normal((n, len(instance)))
        # Масштаб шума = 30% |значения| + 0.1 (защита от нулевых признаков)
        scale = np.abs(instance) * 0.3 + 0.1
        return instance + noise * scale

    def _kernel_weights_normed(
        self, instance_norm: np.ndarray, perturbations_norm: np.ndarray
    ) -> np.ndarray:
        """Экспоненциальное ядро в нормализованном пространстве признаков.

        Расстояния вычисляются ПОСЛЕ стандартизации, чтобы признаки с большим
        масштабом (напр., avg_amount ≈ 800) не доминировали над малыми.
        d = sqrt(Σ diff_norm² / n_features) ∈ [0, 1] при типичных отклонениях.
        """
        n_features = len(instance_norm)
        diffs = perturbations_norm - instance_norm
        distances = np.sqrt(np.sum(diffs**2, axis=1) / max(n_features, 1))
        sigma = self.config.kernel_width
        return np.exp(-(distances**2) / (sigma**2))

    def _ridge_fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Взвешенная Ridge regression с аналитическим решением.

        β = (XᵀWX + αR)⁻¹ XᵀWy,  R = diag(0, 1, ..., 1)  (intercept не штрафуется).
        """
        n_feat = X.shape[1]
        ones = np.ones((len(X), 1))
        X_aug = np.column_stack([ones, X])  # добавляем intercept-столбец

        W = np.diag(weights)
        reg = self.config.ridge_alpha * np.eye(n_feat + 1)
        reg[0, 0] = 0.0  # intercept не регуляризируется

        A = X_aug.T @ W @ X_aug + reg
        b = X_aug.T @ (W @ y)
        try:
            coeffs = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            coeffs = np.zeros(n_feat + 1)

        return coeffs[1:], float(coeffs[0])

    # ── public API ───────────────────────────────────────────────────────────

    def explain(
        self,
        instance: np.ndarray,
        predict_fn: Callable[[np.ndarray], np.ndarray],
    ) -> LIMEExplanation:
        """Объяснить предсказание для одной транзакции.

        Args:
            instance: 1-D массив признаков транзакции.
            predict_fn: функция модели, возвращающая predict_proba(X) (N, 2).

        Returns:
            LIMEExplanation с топ признаками, упорядоченными по |contribution|.
        """
        instance = np.asarray(instance, dtype=float).ravel()
        n_feat = len(instance)

        # Предсказание оригинальной модели для данного примера
        orig_pred = float(predict_fn(instance.reshape(1, -1))[0, 1])

        # Генерация соседей и получение их предсказаний
        perturbations = self._perturb(instance)
        neighbor_preds = predict_fn(perturbations)[:, 1]

        # Нормировать признаки для стабильности Ridge И корректного ядра
        X_all = np.vstack([instance.reshape(1, -1), perturbations])
        y_all = np.concatenate([[orig_pred], neighbor_preds])
        feat_mean = X_all.mean(axis=0)
        feat_std = X_all.std(axis=0)
        feat_std[feat_std < 1e-8] = 1.0
        X_norm = (X_all - feat_mean) / feat_std

        # Веса ядра — вычисляются в нормализованном пространстве.
        # Это критично: без нормализации признаки с большим масштабом (напр.
        # avg_amount ≈ 800) создают огромные расстояния → exp(-d²/σ²) ≈ 0 для
        # всех соседей, и Ridge деградирует в чистую регуляризацию (нулевые коэф.).
        neighbor_weights = self._kernel_weights_normed(X_norm[0], X_norm[1:])

        # Включить оригинальный пример с максимальным весом = 1.0
        w_all = np.concatenate([[1.0], neighbor_weights])

        # Fit взвешенной Ridge regression
        coeffs_norm, intercept = self._ridge_fit(X_norm, y_all, w_all)

        # Денормализовать коэффициенты: β_raw = β_norm / std
        raw_coeffs = coeffs_norm / feat_std

        # Локальное предсказание суррогатной модели для instance
        instance_norm = (instance - feat_mean) / feat_std
        local_pred_raw = intercept + float(np.dot(coeffs_norm, instance_norm))
        local_pred = float(np.clip(local_pred_raw, 0.0, 1.0))
        fidelity = float(np.clip(1.0 - abs(orig_pred - local_pred), 0.0, 1.0))

        # Выбрать топ-N признаков по абсолютному вкладу
        n_top = min(self.config.n_features_in_explanation, n_feat)
        ranked_idx = np.argsort(np.abs(raw_coeffs))[::-1][:n_top]

        top_features: list[LIMEFeatureContribution] = []
        for idx in ranked_idx:
            contrib = float(raw_coeffs[idx])
            if abs(contrib) < 1e-8:
                direction = "neutral"
            elif contrib > 0:
                direction = "increases_fraud_risk"
            else:
                direction = "decreases_fraud_risk"

            top_features.append(
                LIMEFeatureContribution(
                    feature_name=self.feature_names[int(idx)],
                    value=float(instance[idx]),
                    contribution=contrib,
                    direction=direction,
                )
            )

        return LIMEExplanation(
            prediction=round(orig_pred, 4),
            local_prediction=round(local_pred, 4),
            local_fidelity=round(fidelity, 4),
            intercept=round(intercept, 4),
            top_features=top_features,
            n_perturbations=self.config.n_perturbations,
        )

    @staticmethod
    def is_available() -> bool:
        """Всегда True — реализован на чистом numpy, без внешних зависимостей."""
        return True
