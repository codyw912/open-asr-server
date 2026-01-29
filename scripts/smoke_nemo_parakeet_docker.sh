#!/usr/bin/env bash

set -euo pipefail

BASE_IMAGE=${BASE_IMAGE:-open-asr-nemo-base}
BASE_TAG=${BASE_TAG:-torch2.5.1-cu121}
IMAGE=${IMAGE:-open-asr-nemo-dev}
TAG=${TAG:-torch2.5.1-cu121}
CUDA_BASE_IMAGE=${CUDA_BASE_IMAGE:-pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
HF_CACHE_DIR=${HF_CACHE_DIR:-}
INFO=${INFO:-0}
SMOKE_CMD="uv run --active --no-sync scripts/smoke_nemo_parakeet.py samples/jfk_0_5.flac"

echo "Building base image ${BASE_IMAGE}:${BASE_TAG}"
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} docker build -f Dockerfile.nemo.base \
	--build-arg CUDA_BASE_IMAGE="${CUDA_BASE_IMAGE}" \
	--build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
	-t "${BASE_IMAGE}:${BASE_TAG}" .

echo "Building dev image ${IMAGE}:${TAG}"
DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1} docker build -f Dockerfile.nemo \
	--build-arg BASE_IMAGE="${BASE_IMAGE}:${BASE_TAG}" \
	-t "${IMAGE}:${TAG}" .

DOCKER_RUN_ARGS=(--rm -it --device nvidia.com/gpu=all -v "$(pwd)":/workspace -w /workspace)
if [ -n "${HF_CACHE_DIR}" ]; then
	DOCKER_RUN_ARGS+=("-v" "${HF_CACHE_DIR}:/root/.cache/huggingface")
fi

echo "Running smoke test"
if [ "${INFO}" = "1" ]; then
	docker run "${DOCKER_RUN_ARGS[@]}" --entrypoint bash "${IMAGE}:${TAG}" -lc \
		"nvidia-smi || true; \
    python3 -c \"import torch; print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda)\"; \
    ${SMOKE_CMD}"
else
	docker run "${DOCKER_RUN_ARGS[@]}" "${IMAGE}:${TAG}"
fi
