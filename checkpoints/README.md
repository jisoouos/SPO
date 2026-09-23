# Pretrained checkpoint

## Download

Install [Git LFS](https://git-lfs.com/), then run in the cloned repository:

```bash
git lfs install
git lfs pull --include="checkpoints/spo-best.ckpt"
```

## Load for embedding extraction

Run from the repository root. CPU example:

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
