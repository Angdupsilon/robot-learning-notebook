#!/usr/bin/env bash
# Open a shell on the DGX Spark inside NVIDIA's PyTorch container, with the whole
# robot-learning-notebook checkout mounted at /workspace and this folder as the working directory.
#
#   bash FrameAttention/spark_setup.sh            # from the repo root (or anywhere: paths are resolved)
#   TAG=26.02-py3 bash FrameAttention/spark_setup.sh   # pick a different container release
#
# Why a container: NVIDIA's image ships a PyTorch build for the Spark's GB10 GPU (sm_121).
# Data written to FrameAttention/data stays on the host, so it survives the container.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${TAG:-25.11-py3}"
exec docker run --gpus all -it --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO":/workspace -w /workspace/FrameAttention \
  "nvcr.io/nvidia/pytorch:$TAG" \
  bash -c 'pip install -q -r requirements.txt && python -c "import torch; print(\"torch\", torch.__version__, \"| GPU:\", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))" && exec bash'
