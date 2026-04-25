#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

docker build -t pytorch-unet "$PROJECT_ROOT"

docker rm -f pytorch-unet >/dev/null 2>&1 || true

docker run -d \
  --name pytorch-unet \
  --gpus all \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$PROJECT_ROOT:/workspace/unet" \
  -w /workspace/unet \
  pytorch-unet \
  python train.py --device auto --amp

echo "Container started in background: pytorch-unet"
echo "Follow logs: docker logs -f pytorch-unet"
echo "Stop it: docker stop pytorch-unet"