FROM nvcr.io/nvidia/pytorch:25.06-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "pytorch-lightning>=2.2,<3" transformers wandb \
    numpy scipy scikit-learn soundfile librosa

CMD ["/bin/bash"]
