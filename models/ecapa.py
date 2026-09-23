import torch
import torch.nn as nn
import librosa

from .modules import Bottle2neck, FbankAug, PreEmphasis


class LibrosaMelSpectrogram(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=400, hop_length=160, f_min=20, f_max=7600, n_mels=80):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.n_mels = n_mels

    def forward(self, x):
        # x: (batch, time)
        batch_size = x.shape[0]
        specs = []
        for i in range(batch_size):
            audio = x[i].cpu().numpy()
            spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                fmin=self.f_min,
                fmax=self.f_max,
                n_mels=self.n_mels,
                window='hamming'
            )
            specs.append(torch.from_numpy(spec).to(x.device))
        return torch.stack(specs, dim=0)

class ECAPA_TDNN(nn.Module):
    def __init__(
        self,
        ssl_model: bool,
        num_hidden_layers: int,
        hidden_size: int,
        channel: int,
        embedding_size: int,
    ) -> None:
        super().__init__()

        # Feature aggregation: learnable weighted sum over hidden states
        self.torchfbank = nn.Sequential(
            PreEmphasis(),
            LibrosaMelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=20,
                f_max=7600,
                n_mels=80,
            ),
        )
        self.ssl_model = ssl_model
        self.specaug = FbankAug() # Spec augmentation

        # Feature aggregation: learnable weighted sum over hidden states
        self.norm = nn.InstanceNorm1d(hidden_size)
        self.w = nn.Parameter(torch.ones(1, num_hidden_layers, 1, 1))
        self.conv2s = nn.Conv1d(hidden_size, channel, kernel_size=5, stride=1, padding=2)

        # ECAPA module
        self.conv2 = nn.Conv1d(80, channel, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(channel)

        self.layer1 = Bottle2neck(channel, channel, kernel_size=3, dilation=2, scale=4)
        self.layer2 = Bottle2neck(channel, channel, kernel_size=3, dilation=3, scale=4)
        self.layer3 = Bottle2neck(channel, channel, kernel_size=3, dilation=4, scale=4)
        self.layer4 = nn.Conv1d(3 * channel, 1536, kernel_size=1)

        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2)
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, embedding_size)
        self.bn6 = nn.BatchNorm1d(embedding_size)

    def forward(self, x, specaug=False):
        """
        Args:
            x: hidden states from a transformer-based encoder.

        Returns:
            final_embedding: The final embedding tensor.
        """
        if self.ssl_model == True:
            # 1. Stack and weighted aggregation
            x = x * self.w.repeat(x.size(0), 1, 1, 1)
            x = x.sum(dim=1)  # (batch, time, features)

            # 2. Instance normalization + augmentation
            x = self.norm(x.transpose(1, 2))  # (batch, features, time)
            x = self.conv2s(x)
        else:
            # 1. fbanik
            with torch.no_grad():
                x = self.torchfbank(x)+1e-6
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)
                if specaug == True:
                    x = self.specaug(x)
            x = self.conv2(x)

        # 3. ECAPA convolutional front-end

        x = self.relu(x)
        x = self.bn2(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.layer4(x)
        x = self.relu(x)

        # 4. ASP
        time_steps = x.size(-1)
        mean_stat = torch.mean(x, dim=2, keepdim=True).repeat(1, 1, time_steps)
        std_stat = torch.sqrt(torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)).repeat(1, 1, time_steps)
        gx = torch.cat((x, mean_stat, std_stat), dim=1)
        w = self.attention(gx)
        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt(torch.sum((x ** 2) * w, dim=2).sub(mu ** 2).clamp(min=1e-4))

        # 5. Final embedding
        x = torch.cat((mu, sg), dim=1)
        x = self.bn5(x)
        x = self.fc6(x)
        final_embedding = self.bn6(x)

        return final_embedding
