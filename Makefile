IMAGE ?= open-asr-nemo
TAG ?= cu121
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu121

.PHONY: nemo-image nemo-run nemo-shell

nemo-image:
	docker build -f Dockerfile.nemo \
		--build-arg TORCH_INDEX_URL=$(TORCH_INDEX_URL) \
		-t $(IMAGE):$(TAG) .

nemo-run:
	docker run --rm -it --gpus all $(IMAGE):$(TAG)

nemo-shell:
	docker run --rm -it --gpus all --entrypoint bash $(IMAGE):$(TAG)
