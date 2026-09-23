FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg sox libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN (pip uninstall -y torchaudio || true) && \
    cd /tmp && \
    git clone --recursive --branch v2.8.0 https://github.com/pytorch/audio.git && \
    cd audio && \
    USE_CUDA=1 BUILD_SOX=1 BUILD_KALDI=0 \
    pip install . --no-build-isolation --no-deps && \
    rm -rf /tmp/audio

RUN pip install --no-cache-dir huggingface_hub
RUN pip install --no-cache-dir wandb
RUN pip install --no-cache-dir transformers
RUN pip install --no-cache-dir "lightning>=2.2,<3"
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir scipy
RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir soundfile
RUN pip install --no-cache-dir librosa

CMD ["/bin/bash"]
