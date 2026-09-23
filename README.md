<h1 align="center">SPO</h1>

<p align="center">
  <strong>Speaker verification with WavLM + ECAPA-TDNN</strong>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&amp;logoColor=white" alt="Framework: PyTorch"></a>
  <a href="https://lightning.ai/docs/pytorch/stable/"><img src="https://img.shields.io/badge/Powered_by-PyTorch_Lightning-792EE5" alt="Powered by PyTorch Lightning"></a>
  <a href="#environment-setting"><img src="https://img.shields.io/badge/Environment-Docker-2496ED?logo=docker&amp;logoColor=white" alt="Environment: Docker"></a>
  <a href="#additional-logger-wb"><img src="https://img.shields.io/badge/Logging-W%26B_optional-FFBE00?logo=weightsandbiases&amp;logoColor=black" alt="Logging: W&amp;B optional"></a>
</p>

<p align="center">
  <a href="#prerequisites">Setup</a> ·
  <a href="#run-experiment">Training</a> ·
  <a href="#additional-logger-wb">W&amp;B</a> ·
  <a href="#pretrained-model">Models</a>
</p>

Training code, the Docker recipe, and a weights-only
[pretrained checkpoint](checkpoints/README.md) are included.

## Prerequisites

### Dataset setting

Prepare your own 16 kHz speech data, noise, RIRs, and the following text lists:

| List | Format of each line |
| --- | --- |
| Training | `speaker_id integer_label {YOUR_AUDIO_PATH}` |
| Validation trials | `binary_label integer_key_1 integer_key_2 {YOUR_AUDIO_1_PATH} {YOUR_AUDIO_2_PATH}` |
| Noise / RIR | `{YOUR_AUDIO_PATH}` |

Training labels must be contiguous integers starting at zero. Trial labels are
`1` for the same speaker and `0` for different speakers; each key must identify
one audio file consistently. Convert official three-column trials to this
five-column format before running. Audio paths cannot contain whitespace.

Replace every `{YOUR_...}` placeholder, including the braces, with your own path.
Inside Docker, use container-visible audio paths such as `/data/...` in the lists.
Relative audio paths are resolved from the working directory, not the list file.

### Environment setting

The Dockerfile is based on the supplied experiment environment:

- NVIDIA PyTorch image: `nvcr.io/nvidia/pytorch:25.06-py3`
- torchaudio: `v2.8.0` source build
- Lightning: `>=2.2,<3`
- Transformers, NumPy, SciPy, scikit-learn, SoundFile, librosa, and W&B

Use a Linux Docker host with NVIDIA GPU support. From the repository root:

**1. Build the environment**

```bash
bash docker/build.sh
```

**2. Open a container shell**

```bash
# Open a shell using GPU 0; prepare both host directories first.
GPU_DEVICES=device=0 bash docker/launch.sh "{YOUR_DATA_ROOT}" "{YOUR_OUTPUT_DIR}"
```

The image is named `spo:latest`. Launching opens a shell, **not a training run**.

| On your machine | Inside the container | Access |
| --- | --- | --- |
| This repository | `/workspace/research` | Read-only |
| `{YOUR_DATA_ROOT}` | `/data` | Read-only |
| `{YOUR_OUTPUT_DIR}` | `/output` | Read/write |

Use an empty output directory for a new run. Results written under `/output`
remain on your machine when the container exits.

<details>
<summary>GPU, shared-memory, and communication options</summary>

Omitting `GPU_DEVICES` exposes all GPUs. Optional `SHM_SIZE` (default `80g`)
and `IMAGE_NAME` override shared memory and the image tag. The supplied NCCL
settings are retained; they disable NCCL P2P and shared-memory transports and may
affect multi-GPU performance. The shared-memory limit is not GPU VRAM.
IPC is private to the container. No host credentials or home directory are
mounted, and only the Docker directory is used for the build.

</details>

This is the single-GPU recipe; select one GPU as shown above. Multi-GPU/DDP
training is not validated. The adapted image and full training recipe still
require end-to-end validation.

## Run experiment

### System arguments

Set training hyperparameters in `arguments.py`.

| Setting | Default |
| --- | --- |
| Training | Step 0 → 18,000 |
| DA probability | Ramp to 0.6 |
| AAM margin | 0.2 at step 10,000 → 0.4 at step 18,000 |
| Global loss multiplier | None |
| Model saving | Best validation EER only |

Provide the five required paths at launch. From the source directory (or the
container shell), run:

```bash
python main.py \
  --train-samples "{YOUR_TRAIN_LIST_PATH}" \
  --vox-trials "{YOUR_TRIAL_LIST_PATH}" \
  --noise-samples "{YOUR_NOISE_LIST_PATH}" \
  --reverb-samples "{YOUR_RIR_LIST_PATH}" \
  --output-dir "{YOUR_OUTPUT_DIR}"
```

For Docker, use list paths under `/data` and `--output-dir "/output"`.
Training starts at step zero; only the lowest validation `vox_eer` model is
saved. There are no periodic/last checkpoints or training-state resume.
Use `python arguments.py --help` to see the path options.

### Additional logger (W&B)

W&B logging is optional and disabled by default. To use it, edit these entries
in the `args` dictionary in `arguments.py`:

```python
'use_wandb'     : True,
'wandb_entity'  : '{YOUR_WANDB_USER_NAME}',  # W&B username or team
'wandb_api_key' : '{YOUR_WANDB_API_KEY}',
'wandb_project' : 'SPO',
'wandb_name'    : None,  # or your own run name
```

`main.py` connects the logger automatically when `use_wandb=True`; no other
code changes are needed. The Dockerfile includes `wandb`. Keep real API keys
local and never commit them.

Logged metrics are total loss, SV loss, W loss, learning rate, actual DA ratio,
AAM margin, and validation `vox_eer`. Training metrics use a 50-step logging
interval. W&B files are stored under the output directory. Only selected numeric
hyperparameters are added to the run config; API keys and data paths are excluded.
Source/Git capture, console capture, and checkpoint uploads are disabled.

The logger uses the [W&B Lightning integration](https://docs.wandb.ai/models/integrations/lightning).
Leave `use_wandb=False` to train without W&B authentication or logging.

## Pretrained model

| Model | Step | Recorded VoxCeleb1-O EER (%) | Download / loading |
| --- | ---: | ---: | --- |
| WavLM + ECAPA-TDNN (SPO) | 18,000 | 0.7550 | [Checkpoint guide](checkpoints/README.md) |

The checkpoint is stored with Git LFS. After cloning, run:

```bash
git lfs install
git lfs pull --include="checkpoints/spo-best.ckpt"
```

The recorded EER comes from the supplied checkpoint's experiment record; it was
not re-evaluated for this release. All model tensors are retained exactly, while
optimizer state, callbacks, paths, and other training metadata are removed.
The current `main.py` trains from step zero, not checkpoint evaluation or resume.
This recipe removes the original global loss multiplier of 50 and is therefore
not a claim to reproduce the original checkpoint's reported score by retraining.

See [third-party notices](THIRD_PARTY_NOTICES.md) for the WavLM attribution and
upstream license. Dataset audio and file lists are not distributed.
