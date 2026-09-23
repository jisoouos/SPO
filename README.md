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

Speaker embedding extractors are typically trained using classification objectives where each speaker is represented by a learnable class prototype. Speech signals suffer from severe acoustic variability, producing hard samples that destabilize class prototypes despite offering valuable supervision for feature learning. This fundamental tension explains why existing approaches have diverged into two opposing directions: either suppressing or amplifying hard-sample signals. To address this issue, we propose Selective Prototype Optimization (SPO), which decouples the two gradient paths using complementary stop-gradient operations. SPO maintains full supervision for embedding learning while selectively moderating hard-sample contributions to prototype updates based on their class-relative difficulty, introducing no additional parameters or computational overhead. On VoxCeleb1-O, SPO lowers the equal error rate from 0.914% to 0.755% (a 17.4% relative reduction), outperforming existing hard-sample and center-based approaches. Furthermore, by stabilizing the optimization process, SPO prevents performance collapse under aggressive training regimes, enabling larger angular margins and stronger data augmentation to yield continued gains where baselines degrade.

## Setup

Linux, Docker, and NVIDIA GPU support required. Replace all `{YOUR_...}` placeholders.

### 1. Clone the repository

```bash
git clone https://github.com/jiuos/SPO.git
cd SPO
```

### 2. Build the Docker image

```bash
IMAGE_NAME="{YOUR_IMAGE_NAME}" bash docker/build.sh
```

### 3. Start the container

```bash
bash docker/launch.sh "{YOUR_IMAGE_NAME}"
```

## Train

### 4. Configure training

Set hyperparameters in `arguments.py`.

#### W&B (optional)

Disabled by default. To enable, set these values in `arguments.py`:

```python
'use_wandb': True,
'wandb_entity': '{YOUR_WANDB_USER_NAME}',
'wandb_api_key': '{YOUR_WANDB_API_KEY}',
'wandb_project': '{YOUR_WANDB_PROJECT}',
```

Keep real API keys local; never commit them.

### 5. Run training

Run from the repository root in your configured environment, using your own data and output paths:

```bash
CUDA_VISIBLE_DEVICES="{YOUR_GPU_ID}" python main.py \
  --train-samples "{YOUR_TRAIN_LIST_PATH}" \
  --vox-trials "{YOUR_TRIAL_LIST_PATH}" \
  --noise-samples "{YOUR_NOISE_LIST_PATH}" \
  --reverb-samples "{YOUR_RIR_LIST_PATH}" \
  --output-dir "{YOUR_OUTPUT_DIR}"
```

Starts from step 0; saves only the best validation-EER model to the specified output directory.

## Checkpoint

| Model | Step | Recorded VoxCeleb1-O EER (%) |
| --- | ---: | ---: |
| [spo-best.ckpt](https://media.githubusercontent.com/media/jiuos/SPO/main/checkpoints/spo-best.ckpt) | 18,000 | 0.7550 |

Weights only; not for training resume. [Loading example](checkpoints/README.md).
