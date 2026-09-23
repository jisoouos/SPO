import random

import numpy as np
import soundfile as sf
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Sampler

from data import SVTrainDB

from data import (
    TorchRIRReverberation,
    TorchNoise,
    TorchColorNoiseInjection,
    TorchClipping,
)


# -----------------------------
# DataModules
# -----------------------------
class DataModule(pl.LightningDataModule):
    def __init__(self, args, aug_p_shared):
        super().__init__()
        self.seed = args['seed']
        self.crop_size = args["crop_size"]
        self.test_crop_size = args["test_crop_size"]
        self.TTA_seg_num = args["TTA_seg_num"]
        self.batch_size = args["batch_size"] // args['accumulate_grad_batches']
        self.aug_p_shared = aug_p_shared
        self.aug_p_max = args['p_da']
        self.snr_range = args['snr_range']

        # DA function
        self.reverb = TorchRIRReverberation(args["da_reverb_samples"])
        self.noise = TorchNoise(args["da_noise_samples"])
        self.color_noise = TorchColorNoiseInjection()
        self.samplerate = args['samplerate']
        self.clipping = TorchClipping()

        self.train_db = SVTrainDB(args['train_samples'], args['test_trials'])

    def train_dataloader(self):
        train_set = TrainSet(
            self.train_db.train_set,
            self.reverb,
            self.noise,
            self.color_noise,
            self.clipping,
            self.samplerate,
            self.crop_size,
            self.aug_p_shared,
            self.aug_p_max,
            self.snr_range
        )
        sampler = InfiniteCycleSampler(train_set, shuffle=True, seed=self.seed)
        train_loader = DataLoader(
            train_set,
            num_workers=4,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        enrollment_set = EnrollmentSet(
            self.train_db.enrollment_samples,
            self.test_crop_size,
            self.TTA_seg_num
        )
        loader = DataLoader(
            enrollment_set,
            num_workers=4,
            batch_size=self.batch_size // self.TTA_seg_num,
        )
        return loader


class InfiniteCycleSampler(Sampler[int]):
    def __init__(self, dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.n = len(dataset)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        g = torch.Generator()
        cycle = 0
        indices = []
        pos = 0

        while True:
            if pos >= len(indices):
                g.manual_seed(self.seed + cycle)
                if self.shuffle:
                    indices = torch.randperm(self.n, generator=g).tolist()
                else:
                    indices = list(range(self.n))
                pos = 0
                cycle += 1

            yield indices[pos]
            pos += 1

    def __len__(self):
        return 2**31


# -----------------------------
# Datasets
# -----------------------------
class TrainSet(Dataset):
    def __init__(self, items, da_reverb, da_noise, da_color,
            da_clip, sr, crop_size, aug_p_shared, aug_p_max, snr_range):
        self.items = items
        self.reverb = da_reverb
        self.noise = da_noise
        self.color_noise = da_color
        self.sample_rate = sr
        self.clipping = da_clip
        self.crop_size = crop_size
        self.aug_p_shared = aug_p_shared
        self.aug_p_max = aug_p_max
        self.snr_range = snr_range

    def set_augment_probability(self, p):
        self.aug_p_shared.value = float(p)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = random.choice(self.items[index])
        wav_path = item.path

        # noisy segment (crop 6% longer)
        crop_size = int(self.crop_size * 1.06)
        info = sf.info(wav_path)
        if self.sample_rate is not None:
            assert info.samplerate == self.sample_rate
        total = int(info.frames)

        if total <= crop_size:
            audio, _ = sf.read(wav_path)
            shortage = crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), mode="wrap")
        else:
            max_start = total - crop_size
            start = random.randint(0, max_start)
            stop = start + crop_size

            audio, _ = sf.read(wav_path, start=start, stop=stop)

        audio = torch.from_numpy(audio).to(torch.float32)
        audio, flag_applied = self.augment(audio)
        audio = audio[:self.crop_size]

        return audio, item.label, flag_applied

    def add_noise(self, audio):
        snr_min_final, snr_max_final = self.snr_range

        snr_start = 20

        progress = float(self.aug_p_shared.value) / self.aug_p_max
        progress = min(max(progress, 0.0), 1.0)

        snr_min = snr_start + progress * (snr_min_final - snr_start)
        snr_max = snr_start + progress * (snr_max_final - snr_start)

        snr = (snr_min, snr_max)
        if random.random() < 0.9:
            audio = self.noise(audio, snr)
        else:
            audio = self.color_noise(audio, snr)

        return audio

    def apply_rir(self, audio):
        return self.reverb(audio)

    def apply_bandpass(self, audio):
        low, high = 300, 3400
        L = audio.shape[-1]

        audio_fft = torch.fft.rfft(audio)

        freqs = torch.fft.rfftfreq(L, d=1.0/self.sample_rate).to(audio.device)

        mask = (freqs >= low) & (freqs <= high)

        audio_fft_filtered = audio_fft * mask

        filtered_audio = torch.fft.irfft(audio_fft_filtered, n=L)

        return filtered_audio

    def apply_clipping(self, audio):
        return self.clipping(audio)

    def augment(self, audio):
        """
        Apply random augmentation.
        """
        audio = audio.unsqueeze(0)
        flag_applied = 0

        p = float(self.aug_p_shared.value)
        if random.random() > p:
            return audio.squeeze(0), flag_applied

        augment_ops = [
            ("rir", 0.2, self.apply_rir),
            ("noise", 0.5, self.add_noise),
            ("bp", 0.3, self.apply_bandpass),
            ("clip", 0.1, self.apply_clipping),
        ]

        for _, p, fn in augment_ops:
            if random.random() < p:
                audio = fn(audio)
                flag_applied = 1

        # if nothing applied, pick one op according to normalized probs
        if flag_applied == 0:
            names, probs, fns = zip(*augment_ops)
            total_p = sum(probs)
            probs_norm = [p / total_p for p in probs]

            r = random.random()
            cum = 0.0
            for _, p, fn in zip(names, probs_norm, fns):
                cum += p
                if r < cum:
                    audio = fn(audio)
                    break
            flag_applied = 1

        return audio.squeeze(0), flag_applied


class EnrollmentSet(Dataset):
    def __init__(self, items, crop_size, num_seg):
        self.items = items
        self.crop_size = crop_size
        self.TTA_seg_num = num_seg

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]

        # Read WAV file
        audio, _ = sf.read(item.path)

        # Segment the audio
        if audio.shape[0] <= self.crop_size:
            shortage = self.crop_size - audio.shape[0]
            audio = np.pad(audio, (0, shortage), mode="wrap")
            buffer = np.stack([audio] * self.TTA_seg_num, axis=0)
        else:
            indices = np.linspace(
                0, audio.shape[0] - self.crop_size, self.TTA_seg_num
            )
            buffer = [audio[int(idx) : int(idx) + self.crop_size] for idx in indices]
            buffer = np.stack(buffer, axis=0)

        audio = torch.from_numpy(buffer).to(torch.float32)

        return audio, item.key
