"""
VAE-based anomaly detector for fraud detection.
Детектор аномалий на основе вариационного автоэнкодера (VAE).

Ключевая идея: обучаем VAE только на нормальных (licit) транзакциях.
Аномально высокая ошибка реконструкции на тестовых данных → признак мошенничества.

Key idea: train VAE exclusively on normal (licit) transactions.
Anomalously high reconstruction error on unseen data → fraud signal.

Почему VAE, а не обычный автоэнкодер?
Why VAE instead of plain autoencoder?
  - Регуляризованное латентное пространство (KL-дивергенция) не позволяет
    модели «запомнить» нормальные данные — она учится их распределению.
  - KL-regularized latent space prevents memorization of normal data —
    the model learns the underlying distribution.

Requires: PyTorch. Если недоступен — используйте is_available() для проверки.
Requires: PyTorch. If unavailable — check with is_available() first.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def is_available() -> bool:
    """Проверить, доступен ли PyTorch для запуска VAE.

    Check whether PyTorch is available for VAE execution.
    """
    return _TORCH_AVAILABLE


if _TORCH_AVAILABLE:

    class FraudVAE(nn.Module):
        """Вариационный автоэнкодер для обнаружения мошенничества.

        Variational Autoencoder for fraud detection.

        Архитектура:
          Encoder: input → hidden → hidden/2 → (μ, log σ²)
          Latent:  z = μ + ε·σ,  ε ~ N(0, I)   (трюк репараметризации)
          Decoder: z → hidden/2 → hidden → input_reconstruction

        Args:
            input_dim: размерность входных признаков / input feature dimensionality
            hidden_dim: размер скрытого слоя / hidden layer size
            latent_dim: размерность латентного пространства / latent space dimensionality
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int = 64,
            latent_dim: int = 16,
        ) -> None:
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Кодировать входной вектор в параметры распределения (μ, log σ²)."""
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(
            self, mu: torch.Tensor, logvar: torch.Tensor
        ) -> torch.Tensor:
            """Трюк репараметризации: z = μ + ε·σ, ε ~ N(0, I).

            Reparameterization trick: allows backprop through stochastic sampling.
            """
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Прямой проход: encode → sample → decode.

            Returns:
                x_recon: реконструированный вход / reconstructed input
                mu: mean вектор / mean vector
                logvar: log-variance вектор / log-variance vector
            """
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z)
            return x_recon, mu, logvar

    def _vae_loss(
        x_recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """ELBO loss = reconstruction loss (MSE) + KL divergence.

        ELBO (Evidence Lower BOund) — нижняя граница правдоподобия.
        Minimizing this maximizes the likelihood of normal transactions.

        KL( q(z|x) || N(0,I) ) = -0.5 * Σ(1 + log σ² - μ² - σ²)
        """
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

else:

    class FraudVAE:  # type: ignore[no-redef]
        """Заглушка FraudVAE когда PyTorch недоступен."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError(
                "PyTorch is required for FraudVAE. "
                "Install with: pip install torch"
            )


def train_vae(
    X: np.ndarray,
    y: np.ndarray,
    hidden_dim: int = 64,
    latent_dim: int = 16,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict:
    """Обучить VAE на нормальных транзакциях и оценить на тестовой выборке.

    Train VAE on normal transactions only, evaluate on held-out test set.

    Парадигма детекции аномалий:
    Anomaly detection paradigm:
      1. Train: подаём только нормальные образцы (y==0)
      2. Eval: считаем ошибку реконструкции для всех тестовых образцов
      3. Threshold: 95-й перцентиль ошибок нормальных образцов
      4. Predict: reconstruction_error > threshold → мошенничество

    Args:
        X: матрица признаков / feature matrix
        y: метки (0=normal, 1=fraud) / labels (0=normal, 1=fraud)
        hidden_dim: размер скрытого слоя / hidden layer size
        latent_dim: размерность латентного пространства
        epochs: число эпох обучения / training epochs
        batch_size: размер мини-батча / mini-batch size
        lr: learning rate
        test_size: доля тестовой выборки / test set fraction
        seed: random seed для воспроизводимости / for reproducibility

    Returns:
        dict с ключами / dict with keys:
            model: обученный FraudVAE
            scaler: fitted StandardScaler (для инференса)
            threshold: порог ошибки реконструкции / reconstruction error threshold
            f1_score: float
            roc_auc: float
            y_test: ndarray — истинные метки
            y_pred: ndarray — предсказания по порогу
            y_score: ndarray — ошибки реконструкции (аномальность)

    Raises:
        RuntimeError: если PyTorch недоступен
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for VAE training. "
            "Install with: pip install torch"
        )

    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    # VAE чувствителен к масштабу — стандартизация обязательна
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Обучаем только на нормальных примерах — ключевое для anomaly detection
    X_train_normal = X_train[y_train == 0]

    model = FraudVAE(
        input_dim=X_train_normal.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(epochs):
        idx = rng.permutation(len(X_train_normal))
        for start in range(0, len(X_train_normal), batch_size):
            batch_idx = idx[start : start + batch_size]
            batch = torch.tensor(X_train_normal[batch_idx])

            optimizer.zero_grad()
            x_recon, mu, logvar = model(batch)
            loss = _vae_loss(x_recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()

    # Оценка: аномальный score = средняя ошибка реконструкции по признакам
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(X_test)
        x_recon, _, _ = model(x_tensor)
        recon_errors: np.ndarray = (
            ((x_tensor - x_recon) ** 2).mean(dim=1).numpy()
        )

    # Порог: 95-й перцентиль ошибок нормальных тестовых образцов.
    # Мошеннические транзакции реконструируются хуже → их ошибки выше порога.
    normal_errors = recon_errors[y_test == 0]
    threshold = float(
        np.percentile(normal_errors, 95) if len(normal_errors) > 0 else recon_errors.mean()
    )

    y_pred = (recon_errors > threshold).astype(np.int32)

    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    auc = (
        float(roc_auc_score(y_test, recon_errors))
        if len(np.unique(y_test)) > 1
        else 0.5
    )

    return {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "f1_score": f1,
        "roc_auc": auc,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_score": recon_errors,
    }
