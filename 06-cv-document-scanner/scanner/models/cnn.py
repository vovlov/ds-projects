"""
CNN document classifier -- meant to run inside the Docker container.

On a dev Mac without torch this module still imports fine; every public
function checks TORCH_AVAILABLE and raises a clear error instead of
crashing with a missing-module traceback.

Architecture:
    EfficientNet-V2-S backbone (pretrained on ImageNet) with the
    classifier head replaced by a Linear(1280, 5) layer.  We freeze
    the backbone for the first 3 epochs, then unfreeze and fine-tune
    with a lower LR for 7 more epochs.  Standard augmentation:
    RandomResizedCrop, ColorJitter, RandomRotation, RandomPerspective.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# graceful fallback so the rest of the project works on machines without torch
try:
    import torch
    import torch.nn as nn
    from torchvision import models

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.info(
        "PyTorch not installed -- CNN functionality disabled. "
        "Use the sklearn baseline (src/models/classifier.py) or "
        "run training inside Docker."
    )

NUM_CLASSES = 5  # receipt, id_card, medical_report, invoice, contract


def is_available() -> bool:
    """Check whether the CNN code path can actually execute."""
    return TORCH_AVAILABLE


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed.  Install it or use the Docker image "
            "to run CNN training.  For local testing, use the sklearn baseline."
        )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> Any:
    """EfficientNet-V2-S with a custom classification head.

    Returns an nn.Module (or raises if torch is missing).
    """
    _require_torch()

    weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
    backbone = models.efficientnet_v2_s(weights=weights)

    # replace the classifier (last layer)
    in_features = backbone.classifier[1].in_features
    backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return backbone


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cnn(
    train_loader: Any,
    val_loader: Any,
    epochs_frozen: int = 3,
    epochs_finetune: int = 7,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
    device: str | None = None,
) -> dict[str, Any]:
    """Two-phase training: frozen backbone, then full fine-tune.

    Returns dict with model, history (loss/acc per epoch), best_val_acc.
    """
    _require_torch()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()

    # phase 1 -- freeze backbone, train only the head
    for param in model.features.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr_head)
    history: list[dict[str, float]] = []

    for epoch in range(epochs_frozen):
        _run_epoch(model, train_loader, criterion, optimizer, device, history, epoch)
        _validate(model, val_loader, criterion, device, history, epoch)

    # phase 2 -- unfreeze everything, lower LR for backbone
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {"params": model.features.parameters(), "lr": lr_backbone},
            {"params": model.classifier.parameters(), "lr": lr_head * 0.1},
        ]
    )

    for epoch in range(epochs_frozen, epochs_frozen + epochs_finetune):
        _run_epoch(model, train_loader, criterion, optimizer, device, history, epoch)
        _validate(model, val_loader, criterion, device, history, epoch)

    best_val_acc = max(h["val_acc"] for h in history if "val_acc" in h)
    logger.info("Training complete -- best val accuracy: %.3f", best_val_acc)

    return {"model": model, "history": history, "best_val_acc": best_val_acc}


def _run_epoch(
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: str,
    history: list,
    epoch: int,
) -> None:
    """Single training epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    history.append({"epoch": epoch, "train_loss": avg_loss, "train_acc": acc})
    logger.info("Epoch %d  train_loss=%.4f  train_acc=%.3f", epoch, avg_loss, acc)


def _validate(
    model: Any,
    loader: Any,
    criterion: Any,
    device: str,
    history: list,
    epoch: int,
) -> None:
    """Validation pass after one epoch."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    avg_loss = running_loss / total
    acc = correct / total
    # append val metrics to the last entry (same epoch)
    history[-1].update({"val_loss": avg_loss, "val_acc": acc})
    logger.info("Epoch %d  val_loss=%.4f  val_acc=%.3f", epoch, avg_loss, acc)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model: Any, path: str | Path, input_size: tuple = (1, 3, 224, 224)) -> Path:
    """Export a trained model to ONNX for production inference.

    The ONNX model can be served with ONNX Runtime behind FastAPI
    without needing torch at inference time.
    """
    _require_torch()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(*input_size)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    logger.info("ONNX model exported to %s", path)
    return path
