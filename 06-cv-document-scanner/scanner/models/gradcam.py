"""
Grad-CAM (Gradient-weighted Class Activation Mapping) для визуализации
решений CNN-классификатора документов.

Grad-CAM позволяет понять, на какие зоны изображения смотрит нейросеть
при классификации — критически важно для объяснимости в страховой сфере
(EU AI Act, ст. 13: требование прозрачности для решений с высоким риском).

На машинах без PyTorch модуль импортируется корректно; функции с реальными
вычислениями поднимают RuntimeError с понятным сообщением.

References:
    Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", arXiv:1610.02391

    Zhou et al. (2016) "Learning Deep Features for Discriminative Localization"
    (Class Activation Mapping, CAM) — предшественник Grad-CAM.
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info(
        "PyTorch не установлен — Grad-CAM визуализация недоступна. "
        "Установите PyTorch или запустите в Docker-контейнере."
    )


def is_available() -> bool:
    """Проверяет, доступен ли Grad-CAM (PyTorch + torchvision).
    Returns True if Grad-CAM computation is possible."""
    return TORCH_AVAILABLE


def _require_torch() -> None:
    """Raises RuntimeError if PyTorch is not installed."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch не установлен. Установите PyTorch или используйте "
            "Docker-образ для Grad-CAM визуализации."
        )


class GradCAM:
    """Вычисляет Grad-CAM тепловую карту для целевого слоя CNN.

    Computes Grad-CAM heatmap by hooking into a target convolutional layer
    and accumulating gradients during a forward-backward pass.

    Типичное использование / Typical usage:
        cam = GradCAM(model, target_layer=model.features[-1])
        heatmap = cam.compute(image_tensor, class_idx=3)
        overlay = cam.overlay(original_image, heatmap)
    """

    def __init__(self, model: Any, target_layer: Any) -> None:
        """
        Args:
            model: обученный EfficientNet или любой другой CNN.
            target_layer: слой, на котором считаем активации.
                          Обычно последний conv-слой (features[-1]).
        """
        _require_torch()
        self.model = model
        self.target_layer = target_layer

        # Хуки для захвата активаций и градиентов
        self._activations: torch.Tensor | None = None  # type: ignore[name-defined]
        self._gradients: torch.Tensor | None = None    # type: ignore[name-defined]

        # Регистрируем forward hook — сохраняет feature map целевого слоя
        self._fwd_hook = target_layer.register_forward_hook(
            self._save_activations
        )
        # Регистрируем backward hook — сохраняет градиенты по активациям
        self._bwd_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(
        self, module: Any, input: Any, output: Any
    ) -> None:
        """Forward hook: сохраняет feature map."""
        self._activations = output.detach()

    def _save_gradients(
        self, module: Any, grad_input: Any, grad_output: Any
    ) -> None:
        """Backward hook: сохраняет градиент по output целевого слоя."""
        self._gradients = grad_output[0].detach()

    def compute(
        self,
        image_tensor: Any,
        class_idx: int | None = None,
    ) -> np.ndarray:
        """Вычисляет нормализованную тепловую карту Grad-CAM.

        Computes normalized Grad-CAM heatmap [0, 1] via:
        1. Forward pass → получаем логиты и feature maps
        2. Backward pass от целевого класса → получаем градиенты
        3. Global average pooling градиентов → веса α_k
        4. Взвешенная сумма feature maps + ReLU → сырая тепловая карта
        5. Нормализация в [0, 1] и апсамплинг до размера входа

        Args:
            image_tensor: тензор формы (1, C, H, W).
            class_idx: индекс класса для визуализации.
                       None → предсказанный класс (argmax).

        Returns:
            Тепловая карта в виде numpy array формы (H, W), значения [0, 1].
        """
        _require_torch()

        self.model.eval()
        image_tensor = image_tensor.requires_grad_(False)

        # Forward pass
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        # Backward pass только от нужного класса
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Проверяем, что хуки сработали
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Активации или градиенты не захвачены. "
                "Убедитесь, что target_layer входит в граф вычислений."
            )

        # α_k = mean по пространственным измерениям градиентов
        # shape: (batch, C, H', W') → (batch, C, 1, 1)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)

        # Взвешенная линейная комбинация feature maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')

        # ReLU: оставляем только позитивное влияние на класс
        cam = F.relu(cam)

        # Апсамплинг до исходного размера входного изображения
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)

        # Нормализация в [0, 1]
        cam_np: np.ndarray = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            # Плоская карта — модель одинаково смотрит на весь документ
            cam_np = np.zeros_like(cam_np)

        return cam_np

    def overlay(
        self,
        original_image: Any,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: str = "jet",
    ) -> np.ndarray:
        """Накладывает тепловую карту на исходное изображение.

        Overlays Grad-CAM heatmap onto the original document image
        using a configurable colormap and transparency.

        Args:
            original_image: PIL Image или numpy array (H, W, 3) uint8.
            heatmap: numpy array (H, W) из compute(), значения [0, 1].
            alpha: прозрачность тепловой карты (0=исходное, 1=только карта).
            colormap: имя matplotlib colormap ('jet', 'hot', 'viridis', ...).

        Returns:
            numpy array (H, W, 3) uint8 — изображение с наложенной картой.
        """
        _require_torch()

        import matplotlib.cm as mpl_cm  # type: ignore

        # Конвертируем исходное изображение в numpy
        if hasattr(original_image, "numpy"):
            img_np = original_image.numpy()
        elif hasattr(original_image, "convert"):
            img_np = np.array(original_image.convert("RGB"))
        else:
            img_np = np.asarray(original_image)

        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        # Применяем colormap к тепловой карте
        cmap = mpl_cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap)[:, :, :3]  # убираем альфа-канал
        heatmap_uint8 = (heatmap_colored * 255).astype(np.uint8)

        # Если размеры не совпадают — ресайзим тепловую карту
        if img_np.shape[:2] != heatmap_uint8.shape[:2]:
            from PIL import Image as PILImage
            h, w = img_np.shape[:2]
            heatmap_pil = PILImage.fromarray(heatmap_uint8).resize(
                (w, h), PILImage.BILINEAR
            )
            heatmap_uint8 = np.array(heatmap_pil)

        # Линейное смешивание: overlay = (1 - α) * image + α * heatmap
        overlay_np = (
            (1 - alpha) * img_np.astype(np.float32)
            + alpha * heatmap_uint8.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        return overlay_np

    def remove_hooks(self) -> None:
        """Удаляет хуки, освобождает ресурсы.

        Call this when done to avoid memory leaks from dangling hooks.
        """
        self._fwd_hook.remove()
        self._bwd_hook.remove()

    def __del__(self) -> None:
        """Автоматически удаляем хуки при сборке мусора."""
        try:
            self.remove_hooks()
        except Exception:
            pass


def preprocess_image(
    image_path: str,
    image_size: tuple[int, int] = (224, 224),
) -> Any:
    """Загружает и препроцессирует изображение документа для CNN.

    Loads and preprocesses a document image for EfficientNet inference.
    Applies ImageNet normalization (the CNN backbone was pretrained on it).

    Args:
        image_path: путь к файлу изображения.
        image_size: (height, width) для ресайза.

    Returns:
        Тензор формы (1, 3, H, W) готовый для инференса.
    """
    _require_torch()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # ImageNet нормализация — обязательна для pretrained EfficientNet
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


def explain_prediction(
    model: Any,
    image_tensor: Any,
    target_layer: Any,
    class_idx: int | None = None,
) -> dict[str, Any]:
    """Объясняет предсказание CNN через Grad-CAM.

    High-level wrapper that runs Grad-CAM and returns a structured result
    with the predicted class, confidence, and the heatmap.

    Возвращает словарь вместо объекта для совместимости с JSON API.

    Args:
        model: обученный CNN (EfficientNet или другой).
        image_tensor: тензор (1, C, H, W).
        target_layer: целевой слой для Grad-CAM.
        class_idx: None → предсказанный класс.

    Returns:
        dict с ключами:
            - predicted_class_idx (int)
            - confidence (float): вероятность предсказанного класса
            - heatmap (np.ndarray): тепловая карта (H, W) в [0, 1]
            - all_probabilities (np.ndarray): softmax по всем классам
    """
    _require_torch()

    cam = GradCAM(model, target_layer)
    try:
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        target_idx = class_idx if class_idx is not None else predicted_idx

        # Grad-CAM требует градиентов — включаем
        heatmap = cam.compute(image_tensor, class_idx=target_idx)

        return {
            "predicted_class_idx": predicted_idx,
            "confidence": float(probs[predicted_idx]),
            "heatmap": heatmap,
            "all_probabilities": probs,
        }
    finally:
        cam.remove_hooks()
