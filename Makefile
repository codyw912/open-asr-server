BASE_IMAGE ?= open-asr-nemo-base
BASE_TAG ?= torch2.5.1-cu121
DEV_IMAGE ?= open-asr-nemo-dev
DEV_TAG ?= torch2.5.1-cu121
CUDA_BASE_IMAGE ?= pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu121

.PHONY: nemo-base nemo-dev nemo-run nemo-shell

nemo-base:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.nemo.base \
		--build-arg CUDA_BASE_IMAGE=$(CUDA_BASE_IMAGE) \
		--build-arg TORCH_INDEX_URL=$(TORCH_INDEX_URL) \
		-t $(BASE_IMAGE):$(BASE_TAG) .

nemo-dev:
	DOCKER_BUILDKIT=1 docker build -f Dockerfile.nemo \
		--build-arg BASE_IMAGE=$(BASE_IMAGE):$(BASE_TAG) \
		-t $(DEV_IMAGE):$(DEV_TAG) .

nemo-run:
	docker run --rm -it --gpus all \
		-v "$(PWD)":/workspace -w /workspace \
		$(DEV_IMAGE):$(DEV_TAG)

nemo-shell:
	docker run --rm -it --gpus all --entrypoint bash \
		-v "$(PWD)":/workspace -w /workspace \
		$(DEV_IMAGE):$(DEV_TAG)
