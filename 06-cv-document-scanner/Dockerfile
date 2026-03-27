# Multi-stage Dockerfile for the document scanner project.
# Stage 1: train the CNN on GPU (large image with torch + CUDA).
# Stage 2: serve the ONNX model with FastAPI (slim image, no torch).

# ---- training stage ----
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime AS trainer

WORKDIR /app
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

COPY src/ src/
COPY configs/ configs/

# train and export ONNX -- expects a mounted volume at /data with images
RUN python -m src.models.cnn || echo "Skipping CNN training (no data mounted)"

# ---- serving stage ----
FROM python:3.11-slim AS server

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
# copy the ONNX artifact from the training stage (if it was produced)
COPY --from=trainer /app/models/*.onnx models/ 2>/dev/null || true

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
