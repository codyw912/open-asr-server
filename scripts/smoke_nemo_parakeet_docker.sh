#!/usr/bin/env bash

set -euo pipefail

IMAGE=${IMAGE:-open-asr-nemo}
TAG=${TAG:-cu121}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}
INFO=${INFO:-0}
SMOKE_CMD="uv run --extra nemo scripts/smoke_nemo_parakeet.py samples/jfk_0_5.flac"

echo "Building Docker image ${IMAGE}:${TAG}"
docker build -f Dockerfile.nemo \
	--build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
	-t "${IMAGE}:${TAG}" .

echo "Running smoke test"
if [ "${INFO}" = "1" ]; then
	docker run --rm -it --gpus all --entrypoint bash "${IMAGE}:${TAG}" -lc \
		"nvidia-smi || true; \
    python3 -c \"import torch; print('cuda_available', torch.cuda.is_available()); print('torch_cuda', torch.version.cuda)\"; \
    ${SMOKE_CMD}"
else
	docker run --rm -it --gpus all "${IMAGE}:${TAG}"
fi
