#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 ]]; then
    echo "Usage: bash docker/build.sh (optional environment: IMAGE_NAME)"
    if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
        exit 0
    fi
    exit 2
fi

command -v docker >/dev/null 2>&1 || {
    echo "Docker is required. Run this script on your Docker host." >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
image_name="${IMAGE_NAME:-spo:latest}"

# Send only the Dockerfile, with no filesystem build context.
# Local credentials, datasets, code, and checkpoints are never sent.
exec docker build --tag "$image_name" - < "$script_dir/Dockerfile"
