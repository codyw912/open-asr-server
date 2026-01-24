# syntax=docker/dockerfile:1
# Open ASR Server - CPU/CUDA Container
#
# Build targets:
#   docker build --target cpu -t open-asr-server:cpu .
#   docker build --target cuda -t open-asr-server:cuda .
#
# Run:
#   docker run -p 8000:8000 open-asr-server:cpu
#   docker run --gpus all -p 8000:8000 open-asr-server:cuda

# ==============================================================================
# Base stage - shared dependencies
# ==============================================================================
FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# ==============================================================================
# CPU target - faster-whisper backend
# ==============================================================================
FROM base AS cpu

# Install with faster-whisper (CPU) backend
RUN uv pip install --system -e ".[faster-whisper]"

# Default environment
ENV OPEN_ASR_SERVER_DEFAULT_MODEL="openai/whisper-large-v3"
ENV OPEN_ASR_DEFAULT_BACKEND="faster-whisper"

EXPOSE 8000

CMD ["open-asr-server", "serve", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# CUDA base - with NVIDIA runtime
# ==============================================================================
FROM nvidia/cuda:12.4-runtime-ubuntu22.04 AS cuda-base

WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY pyproject.toml uv.lock ./
COPY src/ ./src/

# ==============================================================================
# CUDA target - faster-whisper with GPU support
# ==============================================================================
FROM cuda-base AS cuda

# Install with faster-whisper backend
# Note: faster-whisper auto-detects CUDA when available
RUN uv pip install --system -e ".[faster-whisper]"

# Install CUDA-specific dependencies for CTranslate2
RUN uv pip install --system nvidia-cublas-cu12 nvidia-cudnn-cu12

# Configure for CUDA
ENV OPEN_ASR_SERVER_DEFAULT_MODEL="openai/whisper-large-v3"
ENV OPEN_ASR_DEFAULT_BACKEND="faster-whisper"
# Tell faster-whisper to use CUDA
ENV CUDA_VISIBLE_DEVICES="0"

EXPOSE 8000

CMD ["open-asr-server", "serve", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# CUDA target with Parakeet (when/if CUDA support is added)
# ==============================================================================
# FROM cuda-base AS cuda-parakeet
# 
# # This would require parakeet to support CUDA/PyTorch
# # Currently parakeet is MLX-only (Apple Silicon)
# # Placeholder for future NeMo Parakeet or similar
# 
# RUN uv pip install --system -e ".[parakeet-cuda]"
# ENV OPEN_ASR_SERVER_DEFAULT_MODEL="nvidia/parakeet-tdt-0.6b"
# ENV OPEN_ASR_DEFAULT_BACKEND="parakeet-cuda"
# 
# EXPOSE 8000
# CMD ["open-asr-server", "serve", "--host", "0.0.0.0", "--port", "8000"]
