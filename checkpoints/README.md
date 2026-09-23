# Pretrained checkpoint

| File | Training step | Recorded VoxCeleb1-O EER (%) | Size |
| --- | ---: | ---: | ---: |
| [spo-best.ckpt](spo-best.ckpt) | 18,000 | 0.7550 | 1,354,747,781 bytes |

The EER is from the supplied experiment record, not a new evaluation performed
for this release. The training entry point does not resume from this checkpoint.

## Download

Install [Git LFS](https://git-lfs.com/), then run in the cloned repository:

```bash
git lfs install
git lfs pull --include="checkpoints/spo-best.ckpt"
sha256sum checkpoints/spo-best.ckpt
```

Expected SHA-256:

```text
ecf623a2e668b1d32810842958dd6bfe8244b2d32342ca22fd7d95a3c0b352f9
```

If the file is only a few lines starting with `version https://git-lfs.github.com`,
you have an LFS pointer, not the model. Run the LFS download command above.

## Contents

The file is a plain `{"state_dict": ...}` PyTorch checkpoint with 640 tensors:

- `frontend.*`: frozen WavLM-Large (488 tensors).
- `classifier.*`: ECAPA-TDNN backend (151 tensors).
- `criterion_sv.weight`: speaker-center matrix, shape `[5994, 256]` (1 tensor).

Every tensor is exactly equal to the corresponding tensor in the supplied model.
Optimizer state, scheduler state, callbacks, step/epoch counters, CDF history,
paths, and other training metadata are excluded. Forty unused THOP profiling
counters (`total_ops` / `total_params`) are also removed; no model weight is
removed or changed. This is not a resumable
Lightning training checkpoint. No recordings or speaker-ID lookup table are
included.

## Load for embedding extraction

Run from the repository root in the configured environment. Only the small
WavLM configuration is fetched; its weights are loaded from this checkpoint.
The example uses CPU; choose your device explicitly for real evaluation.

```python
import torch
from transformers import WavLMConfig, WavLMModel
from models import ECAPA_TDNN

checkpoint_path = "{YOUR_CHECKPOINT_PATH}"
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
frontend = WavLMModel(WavLMConfig.from_pretrained("microsoft/wavlm-large"))
backend = ECAPA_TDNN(True, 24, 1024, 1024, 256)

frontend.load_state_dict(
    {key.removeprefix("frontend."): value for key, value in state.items()
     if key.startswith("frontend.")}, strict=True,
)
backend.load_state_dict(
    {key.removeprefix("classifier."): value for key, value in state.items()
     if key.startswith("classifier.")}, strict=True,
)
frontend.eval()
backend.eval()

@torch.inference_mode()
def embed(waveforms):
    # float32 mono audio at 16 kHz, shaped [batch, samples].
    hidden = frontend(waveforms, output_hidden_states=True).hidden_states
    hidden = torch.stack(hidden, dim=1)[:, 1:, :, :]
    return backend(hidden)
```

Speaker verification does not use the training speaker-center matrix. For
matching the supplied validation protocol, use ten 4-second crops per utterance,
normalize each embedding, and average the 10-by-10 cross-crop cosine scores;
the dataset and scoring functions are in `data_module.py` and `model_module.py`.

The public training recipe removes the original global loss multiplier of 50.
That changes retraining, not the numerical contents of these saved weights.
See [third-party notices](../THIRD_PARTY_NOTICES.md) for the WavLM license.
