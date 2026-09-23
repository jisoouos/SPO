import random
import soundfile as sf
import torch

class TorchRIRReverberation:
    def __init__(self, samples, dtype=torch.float32):
        with open(samples, 'r') as f:
            self.rir_files = [line.strip() for line in f if line.strip()]
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T,) or (C, T) on CPU
        returns: same shape as input, length preserved (truncate to T)
        """
        was_1d = (x.dim() == 1)
        if was_1d:
            x = x.unsqueeze(0)  # (1, T)
        x = x.contiguous().to(dtype=self.dtype)
        C, T = x.shape

        # pick & load rir (no cache)
        path = random.choice(self.rir_files)
        rir = self._load_rir_mono_norm(path, device=x.device, dtype=self.dtype)  # (L,)
        L = rir.numel()

        # FFT length (next pow2 of T+L-1)
        n = int(1 << (T + L - 1).bit_length())

        # FFTs
        X = torch.fft.rfft(x, n=n)                        # (C, n//2+1)
        H = torch.fft.rfft(rir.unsqueeze(0), n=n)         # (1, n//2+1)
        Y = X * H                                         # broadcast
        y = torch.fft.irfft(Y, n=n)[..., :T].contiguous() # (C, T), crop to input length

        return y.squeeze(0) if was_1d else y

    def _load_rir_mono_norm(self, path, device, dtype):
        rir_np, _ = sf.read(path, dtype="float32")  # (L,) or (L, C)
        if rir_np.ndim > 1:
            rir_np = rir_np.mean(axis=1)             # mono
        rir = torch.from_numpy(rir_np)

        # energy normalization
        norm = torch.sqrt((rir ** 2).sum()) + 1e-8
        rir = (rir / norm).to(device=device, dtype=dtype).contiguous()
        return rir
