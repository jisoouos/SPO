import math
import random
import soundfile as sf
import torch

class TorchColorNoiseInjection:
    def __init__(self, snr_range=(5, 15)):
        self.snr_range = snr_range
        self.noise_types = ['white', 'pink', 'brown']
        self.eps = 1e-10

    def __call__(self, waveform: torch.Tensor, snr_db=None) -> torch.Tensor:
        """
        Accepts waveform of shape (T,) or (C, T).
        Returns the same shape as input.
        """
        was_1d = False
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # (1, T)
            was_1d = True
        assert waveform.dim() == 2, "Expected (T,) or (C, T)."

        C, T = waveform.shape
        device, dtype = waveform.device, waveform.dtype

        # One SNR for all channels (uncomment the alt line below to randomize per-channel)
        if snr_db is None:
            snr_db = random.uniform(*self.snr_range)
        else:
            snr_db = random.uniform(*snr_db)
        # snr_db = torch.empty(C, device=device).uniform_(*self.snr_range)  # per-channel SNR

        noise_type = random.choice(self.noise_types)

        # Generate independent colored noise per channel
        noises = [self._generate_colored_noise(T, noise_type, device=device, dtype=dtype)
                  for _ in range(C)]
        noise = torch.stack(noises, dim=0)  # (C, T)

        # Compute target scaling per channel for the desired SNR
        signal_power = (waveform.pow(2).mean(dim=1, keepdim=True)).clamp_min(self.eps)  # (C, 1)
        noise_power = (noise.pow(2).mean(dim=1, keepdim=True)).clamp_min(self.eps)      # (C, 1)

        if isinstance(snr_db, torch.Tensor):
            # per-channel SNR case
            target_noise_power = signal_power / (10.0 ** (snr_db.view(-1, 1) / 10.0))
        else:
            # shared SNR for all channels
            target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))

        scale = torch.sqrt(target_noise_power / noise_power)
        noise = noise * scale  # (C, T)

        out = waveform + noise
        return out

    def _generate_colored_noise(self, length: int, noise_type: str, device=None, dtype=None) -> torch.Tensor:
        """
        Generate colored noise of length T on a single channel.
        """
        # rFFT frequency bins (positive frequencies)
        freqs = torch.fft.rfftfreq(length, d=1.0).to(device=device, dtype=dtype)
        freqs[0] = 1e-6  # avoid division by zero

        # Amplitude spectrum
        if noise_type == 'white':
            spectrum = torch.ones_like(freqs)
        elif noise_type == 'pink':
            spectrum = 1.0 / torch.sqrt(freqs)
        elif noise_type == 'brown':
            spectrum = 1.0 / freqs
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Random phase in [0, 2π). Build complex spectrum robustly.
        phases = torch.rand_like(freqs) * 2.0 * math.pi
        real = spectrum * torch.cos(phases)
        imag = spectrum * torch.sin(phases)
        noise_fft = torch.complex(real, imag)  # complex64/complex128 depending on dtype

        # iRFFT to time domain
        noise = torch.fft.irfft(noise_fft, n=length)

        # Peak normalize to stabilize power scaling
        peak = noise.abs().amax().clamp_min(self.eps)
        noise = noise / peak
        return noise.to(dtype=dtype)

class TorchNoise:
    """
    Mixes random noise WAVs listed in a text file.
    Each line in the text file contains a single WAV path.
    """

    def __init__(
        self,
        samples_txt: str,
        snr_range=(5, 15),          # SNR range (dB)
        file_count_range=(1, 1),    # number of noise files to mix
    ):
        self.snr_range = snr_range
        self.file_count_range = file_count_range

        noise_paths = []
        with open(samples_txt, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    noise_paths.append(path)

        if len(noise_paths) == 0:
            raise ValueError(f"No noise files found in {samples_txt}")

        self.noise_paths = noise_paths

    def __call__(self, x: torch.Tensor, snr=None) -> torch.Tensor:
        """Mix random noise into the waveform."""
        was_1d = (x.dim() == 1)
        if was_1d:
            x = x.unsqueeze(0)

        device, dtype = x.device, x.dtype
        C, T = x.shape

        x_dB = self.calculate_decibel(x)

        if snr is None:
            snr = random.uniform(*self.snr_range)
        else:
            snr = random.uniform(*snr)
        num_files = random.randint(*self.file_count_range)
        num_files = min(num_files, len(self.noise_paths))

        if num_files <= 0:
            return x.squeeze(0) if was_1d else x

        noise_paths = random.sample(self.noise_paths, num_files)

        noises = [self.load_noise(p, T, device, dtype) for p in noise_paths]
        if len(noises) == 0:
            return x.squeeze(0) if was_1d else x

        noise_mono = torch.stack(noises, dim=0).mean(dim=0)

        noise_dB = self.calculate_decibel(noise_mono)
        scale = torch.sqrt(
            torch.tensor(10.0, device=device, dtype=dtype)
            ** ((x_dB - noise_dB - snr) / 10.0)
        )

        noise = (scale * noise_mono).unsqueeze(0).expand(C, T)

        x = x + noise
        return x.squeeze(0) if was_1d else x

    def load_noise(self, path: str, target_size: int, device=None, dtype=None) -> torch.Tensor:
        """Load and return mono noise with target length using soundfile."""
        meta = sf.info(path)
        total_frames = int(meta.frames)

        if total_frames <= target_size:
            wav_np, _ = sf.read(path, dtype="float32")
            wav = torch.from_numpy(wav_np)
            if wav.dim() > 1:
                wav = wav.mean(dim=1)
            F_in = wav.numel()
            if F_in < target_size:
                repeat_times = math.ceil(target_size / max(F_in, 1))
                wav = wav.repeat(repeat_times)[:target_size]
        else:
            start = random.randint(0, total_frames - target_size)
            stop = start + target_size
            wav_np, _ = sf.read(path, start=start, stop=stop, dtype="float32")
            wav = torch.from_numpy(wav_np)
            if wav.dim() > 1:
                wav = wav.mean(dim=1)

        wav = wav.to(device=device, dtype=dtype or wav.dtype)
        peak = wav.abs().amax().clamp(min=1e-8)
        return wav / peak

    @staticmethod
    def calculate_decibel(wav: torch.Tensor) -> torch.Tensor:
        """Return power in dB."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        power = (wav ** 2).mean()
        return 10.0 * torch.log10(power + 1e-8)
