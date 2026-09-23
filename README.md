<h1 align="center">SPO</h1>

<p align="center">
  <strong>DECOUPLING HARD-SAMPLE GRADIENTS FOR SPEAKER EMBEDDING AND CLASS-CENTER LEARNING</strong>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?logo=pytorch&amp;logoColor=white" alt="Framework: PyTorch"></a>
  <a href="https://lightning.ai/docs/pytorch/stable/"><img src="https://img.shields.io/badge/Powered_by-PyTorch_Lightning-792EE5" alt="Powered by PyTorch Lightning"></a>
  <a href="#setup"><img src="https://img.shields.io/badge/Environment-Docker-2496ED?logo=docker&amp;logoColor=white" alt="Environment: Docker"></a>
  <a href="#wb-optional"><img src="https://img.shields.io/badge/Logging-W%26B_optional-FFBE00?logo=weightsandbiases&amp;logoColor=black" alt="Logging: W&amp;B optional"></a>
</p>

## Abstract

Selective Prototype Optimization (SPO) decouples embedding and prototype updates using complementary stop-gradient operations, preserving full supervision for embedding learning while selectively moderating hard-sample contributions to prototype updates based on class-relative difficulty. On VoxCeleb1-O, SPO reduces EER from 0.914% to 0.755% (17.4% relative reduction) and improves training stability under larger angular margins and stronger data augmentation.

## Setup

Linux, Docker, and NVIDIA GPU support required. Run from the repository root:

```bash
IMAGE_NAME="{YOUR_IMAGE_NAME}" bash docker/build.sh
bash docker/launch.sh "{YOUR_IMAGE_NAME}"
```

Replace `{YOUR_...}` placeholders. Add your own code/data/output mounts to `docker/launch.sh`; no paths are preset. Unmounted files are deleted when the container exits.

## Train

Set hyperparameters in `arguments.py`, then run from the repository root inside the container:

```bash
CUDA_VISIBLE_DEVICES="{YOUR_GPU_ID}" python main.py \
  --train-samples "{YOUR_TRAIN_LIST_PATH}" \
  --vox-trials "{YOUR_TRIAL_LIST_PATH}" \
  --noise-samples "{YOUR_NOISE_LIST_PATH}" \
  --reverb-samples "{YOUR_RIR_LIST_PATH}" \
  --output-dir "{YOUR_OUTPUT_DIR}"
```

Single-GPU recipe. Use an empty output directory. Starts from step 0; saves only the best validation-EER model.

## W&B (optional)

Disabled by default. To enable, set these values in `arguments.py`:

```python
'use_wandb': True,
'wandb_entity': '{YOUR_WANDB_USER_NAME}',
'wandb_api_key': '{YOUR_WANDB_API_KEY}',
'wandb_project': '{YOUR_WANDB_PROJECT}',
```

Keep real API keys local; never commit them.

## Checkpoint

| Model | Step | Recorded VoxCeleb1-O EER (%) |
| --- | ---: | ---: |
| [spo-best.ckpt](checkpoints/spo-best.ckpt) | 18,000 | 0.7550 |

```bash
git lfs install
git lfs pull --include="checkpoints/spo-best.ckpt"
```

Weights only; not for training resume. [Loading example](checkpoints/README.md).
