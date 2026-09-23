import random
import torch

class TorchClipping:
    def __init__(self, clip_range=(0.5, 0.9)):
        self.clip_range = clip_range

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: torch.Tensor (T,) or (C, T)
        Returns clipped waveform
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if self.clip_range[0] == self.clip_range[1]:
            clip_threshold = self.clip_range[0]
        else:
            clip_threshold = random.uniform(*self.clip_range)

        max_val = waveform.abs().max()
        threshold = max_val * clip_threshold

        # Clip with torch.clamp
        waveform = torch.clamp(waveform, -threshold, threshold)

        return waveform
