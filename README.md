# SPO

Speaker verification with WavLM and ECAPA-TDNN.

> Initial release: only this README and the Docker environment are currently
> published. Training code and pretrained weights will be added separately;
> the training instructions below apply once the code is available.

## Introduction

This repository contains the PyTorch implementation of SPO. The method separates
extractor and speaker-center updates: the extractor uses the original sample
contributions, while CDF-based weights reduce the influence of hard samples on
speaker centers. Training uses a frozen WavLM frontend and an ECAPA-TDNN backend.

Paper details and the citation will be added when available.

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

```bash
# Build the local image: spo:latest
bash docker/build.sh

# Open a shell using GPU 0; prepare both host directories first.
GPU_DEVICES=device=0 bash docker/launch.sh "{YOUR_DATA_ROOT}" "{YOUR_OUTPUT_DIR}"
```

The launcher mounts this code at `/workspace/research` and data at `/data`
(read-only), and your output directory at `/output` (read/write). It does not
start training. The output directory must be empty for a new run.

Omitting `GPU_DEVICES` exposes all GPUs. Optional `SHM_SIZE` (default `80g`)
and `IMAGE_NAME` override shared memory and the image tag. The supplied NCCL
settings are retained; IPC is private to the container. No host credentials or
home directory are mounted, and only the Docker directory is used for the build.
The adapted image and training recipe still require end-to-end validation.

## Run experiment

### System arguments

Set training hyperparameters in `arguments.py`. Defaults include 18,000 steps,
DA probability ramping to 0.6, and AAM margin increasing from 0.2 at step 10,000
to 0.4 at step 18,000. There is no global loss multiplier.

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

Pretrained weights will be provided separately. The current entry point is for
training from step zero, not checkpoint evaluation or resume. The modified
recipe is not a claim to reproduce the original checkpoint's reported score.

## Citation

Citation details will be added with the paper.
