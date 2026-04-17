# 06 · CV Document Scanner

> **Business domain:** Insurance company — automated document processing  
> **Package:** `scanner/`  
> **Directory:** `06-cv-document-scanner/`

## What it solves

Classifies scanned documents (invoices, forms, letters, scientific papers, etc.) into 16 categories to route them to the correct processing workflow. Replaces ~8 hours of daily manual sorting at an insurance back-office.

## Architecture

```mermaid
graph LR
    A[Document image] --> B[EfficientNet-B0]
    B --> C[16-class softmax]
    C --> D[Top-3 predictions]
    B --> E[GradCAM layer]
    E --> F[Attention heatmap]
    G[RVL-CDIP dataset] --> H[Training pipeline]
    H --> B
```

## Key components

### EfficientNet Classifier (`scanner/models/`)
- EfficientNet-B0 fine-tuned on RVL-CDIP
- ONNX export for CPU inference (no GPU required)
- Albumentations augmentation pipeline

### RVL-CDIP Dataset {#rvl-cdip}

`scanner/data/rvl_cdip.py` — 400K document images across 16 classes:

```
letter · form · email · handwritten · advertisement
scientific_report · scientific_publication · specification
file_folder · news_article · budget · invoice
presentation · questionnaire · resume · memo
```

- `generate_mock_rvl_cdip()` — synthetic 32×32 grayscale images for CI
- `load_rvl_cdip(path)` — real dataset with graceful fallback
- `to_scanner_format()` — converts to internal `DocumentSample` format

### GradCAM Explainability {#gradcam}

`scanner/models/gradcam.py` — CNN decision visualisation (EU AI Act compliance):

- `GradCAM` class with forward + backward hooks
- `compute(image, class_idx)` → attention heatmap
- `overlay(image, heatmap)` → colourised overlay
- `explain_prediction(image)` → top class + GradCAM in one call
- `is_available()` graceful fallback without PyTorch

### API (`scanner/api/app.py`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Classify document image, top-3 classes |
| `/explain` | POST | Classification + GradCAM heatmap (base64) |
| `/health` | GET | Model status + class list |

## Running Tests

```bash
cd 06-cv-document-scanner
../.venv/bin/python -m pytest tests/ -v --tb=short
```
