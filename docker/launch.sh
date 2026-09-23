#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo 'Usage: bash docker/launch.sh "{YOUR_DATA_ROOT}" "{YOUR_OUTPUT_DIR}"'
    echo 'Both arguments must be existing directories on your Linux Docker host.'
    echo 'Optional environment: IMAGE_NAME, GPU_DEVICES (default: all), SHM_SIZE (default: 80g).'
    echo 'This opens a shell; it does not start training or load a checkpoint.'
}

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    usage
    exit 0
fi
if [[ $# -ne 2 ]]; then
    usage >&2
    exit 2
fi

for directory in "$@"; do
    if [[ "$directory" == *'{YOUR_'* ]]; then
        echo "Replace the path placeholders before launching." >&2
        exit 2
    fi
    if [[ ! -d "$directory" ]]; then
        echo "Both paths must refer to existing directories." >&2
        exit 2
    fi
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
code_dir="$(cd -- "$script_dir/.." && pwd -P)"
data_dir="$(cd -- "$1" && pwd -P)"
output_dir="$(cd -- "$2" && pwd -P)"

# Docker's --mount syntax treats commas as separators, not path characters.
for directory in "$code_dir" "$data_dir" "$output_dir"; do
    if [[ "$directory" == *','* ]]; then
        echo "Mount directory paths must not contain commas." >&2
        exit 2
    fi
done

command -v docker >/dev/null 2>&1 || {
    echo "Docker is required. Run this script on your Docker host." >&2
    exit 1
}

image_name="${IMAGE_NAME:-spo:latest}"
gpu_devices="${GPU_DEVICES:-all}"
shm_size="${SHM_SIZE:-80g}"

# Keep the supplied environment's NCCL settings, but no authentication variables,
# host-wide IPC, home-directory mounts, or machine-specific paths.
exec docker run --interactive --tty --rm \
    --gpus "$gpu_devices" \
    --shm-size "$shm_size" \
    --env NCCL_SHM_DISABLE=1 \
    --env NCCL_P2P_DISABLE=1 \
    --env NCCL_ASYNC_ERROR_HANDLING=1 \
    --mount "type=bind,source=$code_dir,target=/workspace/research,readonly" \
    --mount "type=bind,source=$data_dir,target=/data,readonly" \
    --mount "type=bind,source=$output_dir,target=/output" \
    --workdir /workspace/research \
    "$image_name" /bin/bash
