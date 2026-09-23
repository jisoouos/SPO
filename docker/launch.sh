#!/usr/bin/env bash
set -euo pipefail

image_name="${1:?Usage: bash docker/launch.sh IMAGE_NAME}"

exec sudo docker run -it --rm \
    --gpus all --ipc=host --shm-size 80G \
    --env NCCL_SHM_DISABLE=1 \
    --env NCCL_P2P_DISABLE=1 \
    --env NCCL_ASYNC_ERROR_HANDLING=1 \
    "$image_name" /bin/bash
