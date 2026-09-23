FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Preserve the supplied environment's torchaudio source-build recipe.
RUN (pip uninstall -y torchaudio || true) && \
    cd /tmp && \
    git clone --recursive --branch v2.8.0 https://github.com/pytorch/audio.git && \
    cd audio && \
    USE_CUDA=1 BUILD_SOX=1 BUILD_KALDI=0 \
    pip install . --no-build-isolation --no-deps && \
    rm -rf /tmp/audio

RUN pip install --no-cache-dir \
    huggingface_hub \
    wandb \
    transformers \
    "lightning>=2.2,<3" \
    numpy \
    scipy \
    scikit-learn \
    soundfile \
    librosa

WORKDIR /workspace/research
CMD ["/bin/bash"]
