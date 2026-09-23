import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    def __init__(self, embedding_size, num_class, margin, scale, buffer_size, buffer_stack_step, loss_alpha_step, class_weight=None, topk_panelty=None):
        super(AAMSoftmax, self).__init__()

        self.scale = scale

        # proposed
        self.buffer_size = buffer_size
        self.num_class = num_class
        self.cos_buffer_values = torch.zeros(num_class, buffer_size, dtype=torch.float32)
        self.cos_buffer_counts = torch.zeros(num_class, dtype=torch.long)
        self.cos_buffer_ptrs = torch.zeros(num_class, dtype=torch.long)
        self.class_modes = torch.zeros(num_class, dtype=torch.float32)
        self.class_sigmas = torch.ones(num_class, dtype=torch.float32)
        self.buffer_stack_step = buffer_stack_step
        self.loss_alpha_step = loss_alpha_step
        self.forward_step = 0

        # weight
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(num_class, embedding_size), requires_grad=True
        )

        # CE
        if class_weight is not None:
            self.ce = nn.CrossEntropyLoss(weight=class_weight, reduction='none')
        else:
            self.ce = nn.CrossEntropyLoss(reduction='none')
        nn.init.xavier_normal_(self.weight, gain=1)

        # positive_margine
        self.margin = margin
        self._set_pos_margin(margin)

        # topk penalty
        self.topk = None
        if topk_panelty is not None:
            self.topk, topk_margin = topk_panelty
            self.topk_margin = topk_margin
            self._set_neg_margin(topk_margin)

    def _compute_output(self, x, weight, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(weight))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))

        # cos(theta + m) -> utilized angle addition and subtraction formulas
        phi = cosine * self.cos_pos_m - sine * self.sin_pos_m

        # cos(theta - m)
        if self.topk is not None:
            penalty_cos = cosine * self.cos_neg_m + sine * self.sin_neg_m

        # one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        if self.topk is not None:
            negative_logits = cosine.clone()
            negative_logits.scatter_(1, label.view(-1, 1), float('-inf'))

            K = min(self.topk, cosine.size(1) - 1)
            topk_idx = negative_logits.topk(K, dim=1).indices

            penalty_mask = torch.zeros_like(cosine)
            penalty_mask.scatter_(1, topk_idx, 1.0)

            remainder = (1.0 - one_hot - penalty_mask).clamp(min=0.0)

            output = (one_hot * phi) + (penalty_mask * penalty_cos) + (remainder * cosine)
            output = output * self.scale
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.scale

        return output, cosine

    def forward(self, x, label, step=None):
        if step is None:
            self.forward_step += 1
        else:
            self.forward_step = int(step)

        if self.forward_step < self.loss_alpha_step:
            output, cosine = self._compute_output(x, self.weight, label)
            target_cosine = cosine.gather(1, label.view(-1, 1)).squeeze(1)
            if self.forward_step >= self.buffer_stack_step:
                self._append_classwise_buffer(target_cosine.detach(), label.detach())

            base_loss = self.ce(output, label).mean()
            return base_loss, base_loss.new_zeros(())

        sv_output, sv_cosine = self._compute_output(x, self.weight.detach(), label)
        sv_loss = self.ce(sv_output, label).mean()

        target_cosine = sv_cosine.gather(1, label.view(-1, 1)).squeeze(1)
        if self.forward_step >= self.buffer_stack_step:
            self._append_classwise_buffer(target_cosine.detach(), label.detach())

        # Only prototype gradients are CDF-weighted; embeddings are fixed on this path.
        w_output, _ = self._compute_output(x.detach(), self.weight, label)
        raw_loss = self.ce(w_output, label)
        alpha = torch.ones_like(raw_loss)

        if self.forward_step >= self.loss_alpha_step:
            alpha = self._compute_alpha(target_cosine.detach(), label, alpha)

        w_loss = (raw_loss * alpha).mean()
        return sv_loss, w_loss

    def _append_classwise_buffer(self, target_cosine: torch.Tensor, labels: torch.Tensor):
        self._ensure_buffer_device(target_cosine.device)
        target_cosine = target_cosine.to(device=self.cos_buffer_values.device, dtype=self.cos_buffer_values.dtype)
        labels = labels.to(device=self.cos_buffer_values.device, dtype=torch.long)

        for class_id in torch.unique(labels):
            values = target_cosine[labels == class_id]
            if values.numel() > self.buffer_size:
                values = values[-self.buffer_size:]

            n = values.numel()
            positions = (self.cos_buffer_ptrs[class_id] + torch.arange(n, device=values.device)) % self.buffer_size
            self.cos_buffer_values[class_id, positions] = values
            self.cos_buffer_ptrs[class_id] = (self.cos_buffer_ptrs[class_id] + n) % self.buffer_size
            self.cos_buffer_counts[class_id] = torch.clamp(self.cos_buffer_counts[class_id] + n, max=self.buffer_size)

    def _compute_alpha(self, target_cosine: torch.Tensor, labels: torch.Tensor, default_alpha: torch.Tensor):
        self._ensure_buffer_device(target_cosine.device)
        labels = labels.to(device=self.cos_buffer_values.device, dtype=torch.long)
        target_cosine = target_cosine.to(device=self.cos_buffer_values.device, dtype=self.cos_buffer_values.dtype)

        valid_mask = self.cos_buffer_counts[labels] >= self.buffer_size
        observed_full = (self.cos_buffer_counts == 0) | (self.cos_buffer_counts >= self.buffer_size)
        valid_mask = valid_mask & observed_full.all()
        current_full_classes = torch.unique(labels[valid_mask])
        if current_full_classes.numel() > 0:
            self._update_class_statistics(current_full_classes)

        modes = self.class_modes[labels].to(dtype=default_alpha.dtype, device=default_alpha.device)
        sigmas = self.class_sigmas[labels].to(dtype=default_alpha.dtype, device=default_alpha.device).clamp_min(1e-6)
        target = target_cosine.to(dtype=default_alpha.dtype, device=default_alpha.device)
        valid_mask = valid_mask.to(device=default_alpha.device)

        z = (target - modes) / (sigmas * math.sqrt(2.0))
        cdf = 0.5 * (1.0 + torch.erf(z))
        hard_weight = cdf * 2.0
        return torch.where(valid_mask & (target < modes-sigmas), hard_weight, default_alpha)

    def _ensure_buffer_device(self, device):
        if self.cos_buffer_values.device == device:
            return
        self.cos_buffer_values = self.cos_buffer_values.to(device=device)
        self.cos_buffer_counts = self.cos_buffer_counts.to(device=device)
        self.cos_buffer_ptrs = self.cos_buffer_ptrs.to(device=device)
        self.class_modes = self.class_modes.to(device=device)
        self.class_sigmas = self.class_sigmas.to(device=device)

    def _update_class_statistics(self, class_ids: torch.Tensor):
        values = self.cos_buffer_values[class_ids]
        full_values = self.cos_buffer_values[self.cos_buffer_counts >= self.buffer_size]
        global_min = full_values.min()
        global_max = full_values.max()
        width = (global_max - global_min).clamp_min(1e-6)

        bin_idx = ((values - global_min) / width * 100).long().clamp(0, 99)
        hist = torch.zeros((values.size(0), 100), device=values.device, dtype=torch.float32)
        hist.scatter_add_(1, bin_idx, torch.ones_like(values, dtype=torch.float32))

        mode_idx = hist.argmax(dim=1).to(values.dtype)
        modes = global_min + (mode_idx + 0.5) * (width / 100.0)
        sigmas = torch.sqrt(((values - modes.unsqueeze(1)) ** 2).mean(dim=1)).clamp_min(1e-6)

        self.class_modes[class_ids] = modes.detach()
        self.class_sigmas[class_ids] = sigmas.detach()

    def _set_pos_margin(self, m: float):
        self.margin = float(m)
        self.cos_pos_m = math.cos(self.margin)
        self.sin_pos_m = math.sin(self.margin)

    def _set_neg_margin(self, m: float):
        self.topk_margin = float(m)
        self.cos_neg_m = math.cos(self.topk_margin)
        self.sin_neg_m = math.sin(self.topk_margin)

    def set_margin(self, m: float):
        self._set_pos_margin(m)
