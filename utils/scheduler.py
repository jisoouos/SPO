import math
from torch.optim.lr_scheduler import _LRScheduler

class StepWarmUpAnneal(_LRScheduler):
    """
    Warmup (linear) + step-wise geometric decay (fast early, slow late).
    Decay steps: n_steps
    lr_k = eta_max * gamma^k, where k in [0..n_steps]
    If gamma is None, gamma is computed so that lr at the end matches base_lr (eta_min).
    """
    def __init__(
        self,
        optimizer,
        T_0: int,
        eta_max: float = 0.1,
        T_up: int = 0,
        n_steps: int = 10,
        gamma: float | None = None,
        last_epoch: int = -1,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError(f"Expected non-negative integer T_up, but got {T_up}")
        if n_steps <= 0 or not isinstance(n_steps, int):
            raise ValueError(f"Expected positive integer n_steps, but got {n_steps}")

        self.T_0 = T_0
        self.T_up = T_up
        self.eta_max = float(eta_max)
        self.n_steps = n_steps
        self.gamma = gamma  # if None -> computed per param group

        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs  # treated as eta_min

        # Warmup-only edge case
        if self.T_0 <= self.T_up:
            if self.T_cur < self.T_up:
                return [
                    base_lr + (self.eta_max - base_lr) * (self.T_cur / max(1, self.T_up))
                    for base_lr in self.base_lrs
                ]
            return [self.eta_max for _ in self.base_lrs]

        # 1) Warmup: base_lr -> eta_max
        if self.T_cur < self.T_up:
            return [
                base_lr + (self.eta_max - base_lr) * (self.T_cur / max(1, self.T_up))
                for base_lr in self.base_lrs
            ]

        # 2) Step-wise geometric decay over [T_up, T_0]
        decay_steps_total = self.T_0 - self.T_up
        t = min(self.T_cur - self.T_up, decay_steps_total)
        progress = t / max(1, decay_steps_total)  # 0..1

        step_idx = int(math.floor(progress * self.n_steps))
        step_idx = min(step_idx, self.n_steps)

        lrs = []
        for base_lr in self.base_lrs:
            eta_min = base_lr

            # compute gamma to hit eta_min at the end (recommended)
            if self.gamma is None:
                if self.eta_max <= 0 or eta_min <= 0:
                    raise ValueError("eta_max and eta_min (base_lr) must be > 0 to compute gamma automatically.")
                g = (eta_min / self.eta_max) ** (1.0 / self.n_steps)
            else:
                g = float(self.gamma)

            lr = self.eta_max * (g ** step_idx)

            # clamp to [eta_min, eta_max]
            lr = max(eta_min, min(self.eta_max, lr))
            lrs.append(lr)

        return lrs

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
        else:
            self.T_cur = int(epoch)

        self.last_epoch = int(math.floor(epoch))
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
