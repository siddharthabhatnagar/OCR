# Dockerfile for LOCAL Docker / RunPod / Modal deployment.
# HF Spaces Gradio SDK does NOT use this file — it uses app.py directly.
# On HF Spaces, the SDK auto-installs torch/gradio/spaces/uvicorn and runs app.py.
#
# Local build:
#   docker build -t unlimited-ocr-api .
#   docker run --gpus all -p 7860:7860 -e HF_TOKEN=hf_xxx unlimited-ocr-api

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=7860

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1 libglib2.0-0 poppler-utils git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

# Install torch first (large, stable layer)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.10.0 torchvision==0.25.0

# Install other deps (excluding torch, which is already installed)
COPY requirements.txt .
RUN pip install --no-cache-dir gradio==5.9.1 spaces uvicorn[standard] && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py .
COPY service ./service
COPY client ./client

EXPOSE 7860

CMD ["python", "app.py"]
