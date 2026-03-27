FROM python:3.12-slim

WORKDIR /workspace

# Install torch CPU + all training dependencies
RUN pip install --no-cache-dir \
    torch==2.5.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    "transformers[torch]>=4.47" \
    accelerate>=1.1.0 \
    datasets>=3.2 \
    seqeval>=1.2 \
    torch-geometric \
    networkx>=3.4 \
    catboost>=1.2 \
    scikit-learn>=1.5 \
    numpy \
    polars

# Copy all project source code
COPY 03-ner-service/src/ /workspace/03-ner-service/src/
COPY 04-graph-fraud-detection/src/ /workspace/04-graph-fraud-detection/src/
COPY 05-realtime-anomaly/src/ /workspace/05-realtime-anomaly/src/

# Copy training scripts
COPY scripts/docker_train.py /workspace/docker_train.py

CMD ["python", "docker_train.py"]
