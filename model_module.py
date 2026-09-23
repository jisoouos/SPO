import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils import compute_eer, StepWarmUpAnneal


class ModelModule(pl.LightningModule):
    """
    Single-phase training module. Trials are provided as a dict[name -> list of TestTrial].
    """

    def __init__(
        self,
        args,
        classifier,
        criterion_sv,
        trials,
        frontend=None,
    ):
        super().__init__()
        self.classifier = classifier
        self.criterion_sv = criterion_sv
        self.frontend = frontend
        self.have_ssl_model = args['have_ssl_model']
        self.use_wandb = args["use_wandb"]

        if not isinstance(trials, dict) or len(trials) == 0:
            raise ValueError("trials must be a non-empty dict of name -> trial list")
        self.trials = trials
        self.primary_trial_name = next(iter(trials.keys()))

        # LR / schedule
        self.lr_min = args['lr_min']
        self.lr_max = args['lr_max']
        self.train_steps = args['train_steps']

        self.TTA_seg_num = args["TTA_seg_num"]
        self.weight_decay = args["weight_decay"]

        self.current_EER = None
        self._val_outputs = []

    # -------------------------
    # Optimizer / scheduler
    # -------------------------
    def configure_optimizers(self):
        params = list(self.classifier.parameters())

        if isinstance(self.criterion_sv, torch.nn.Module):
            params += list(self.criterion_sv.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr_min,
            weight_decay=self.weight_decay,
        )

        warmup_steps = int(0.03 * self.train_steps)
        scheduler = {
            "scheduler": StepWarmUpAnneal(
                optimizer,
                T_0=self.train_steps,
                eta_max=self.lr_max,
                T_up=warmup_steps,
                n_steps=self.train_steps,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]

    # -------------------------
    # Forward / train
    # -------------------------
    def forward(self, x, specaug=False):
        if self.have_ssl_model:
            with torch.no_grad():
                hidden_states = self.frontend(x, output_hidden_states=True).hidden_states
                hidden_states = torch.stack(hidden_states, dim=1)[:, 1:, :, :]
            embeddings = self.classifier(hidden_states)
        else:
            embeddings = self.classifier(x, specaug=specaug)
        return embeddings

    def train(self, mode: bool=True):
        if self.have_ssl_model:
            super().train(mode)
            self.frontend.eval()
            return self
        else:
            return super().train(mode)

    def training_step(self, batch, batch_idx):
        x, labels, flag_da = batch
        x = x.to(dtype=torch.float32, device=self.device)
        labels = labels.to(self.device)

        # forward student
        if self.have_ssl_model:
            embeddings = self(x)
        else:
            embeddings = self(x, specaug=True)

        # ----- losses -----
        sv_loss, w_loss = self.criterion_sv(embeddings, labels, step=int(self.global_step))

        loss = sv_loss + w_loss

        if self.use_wandb:
            self.log_dict(
                {
                    "train/loss": loss,
                    "train/loss_sv": sv_loss,
                    "train/loss_w": w_loss,
                    "train/lr": self.trainer.optimizers[0].param_groups[0]["lr"],
                    "train/p_da": flag_da.to(self.device).float().mean(),
                    "train/margin": self.criterion_sv.margin,
                },
                on_step=True,
                on_epoch=False,
                sync_dist=True,
                logger=True,
                batch_size=labels.size(0),
            )

        return loss

    # -------------------------
    # Validation
    # -------------------------
    def on_validation_epoch_start(self):
        self._val_outputs = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        x, keys = batch
        x = x.to(dtype=torch.float32, device=self.device, non_blocking=True)

        batch_size = x.size(0)
        x = x.view(batch_size * self.TTA_seg_num, -1)

        embeddings = self(x)
        embeddings = embeddings.view(batch_size, self.TTA_seg_num, -1)

        self._val_outputs.append({"keys": keys.cpu(), "embeddings": embeddings.detach().cpu()})

    @torch.inference_mode()
    def on_validation_epoch_end(self):
        # 1) collect cpu keys/embeddings from validation_step
        keys_list = []
        embeddings_list = []

        for out in self._val_outputs:
            # out["keys"]: (B,) on CPU
            # out["embeddings"]: (B, TTA, D) on CPU
            keys_list.append(out["keys"])
            embeddings_list.append(out["embeddings"])

        if len(keys_list) == 0:
            return

        all_keys = torch.cat(keys_list, dim=0)            # (N,) CPU
        all_embeddings = torch.cat(embeddings_list, dim=0) # (N, TTA, D) CPU

        # 2) (optional) gather across ranks
        if self.trainer.world_size > 1:
            gathered_keys = self.all_gather(all_keys)              # may be on device
            gathered_embeddings = self.all_gather(all_embeddings)  # may be on device
        else:
            gathered_keys = all_keys
            gathered_embeddings = all_embeddings

        # 3) flatten + ensure CPU tensors
        gathered_keys = gathered_keys.reshape(-1).detach().cpu()

        # gathered_embeddings could be (world, N, TTA, D) or (N, TTA, D)
        gathered_embeddings = gathered_embeddings.detach().cpu()
        if gathered_embeddings.dim() == 4:
            gathered_embeddings = gathered_embeddings.reshape(
                -1, gathered_embeddings.size(-2), gathered_embeddings.size(-1)
            )
        else:
            # already (N, TTA, D)
            gathered_embeddings = gathered_embeddings.reshape(
                -1, gathered_embeddings.size(-2), gathered_embeddings.size(-1)
            )

        # 4) build key -> embedding map (NO huge list pre-allocation)
        embedding_map = {}
        for k, emb in zip(gathered_keys.tolist(), gathered_embeddings):
            embedding_map[int(k)] = emb  # emb: (TTA, D) CPU tensor

        # free intermediate buffers early
        self._val_outputs.clear()
        del keys_list, embeddings_list, all_keys, all_embeddings, gathered_keys, gathered_embeddings

        # 5) Evaluate all trial sets
        metrics = {}
        for name, trials in self.trials.items():
            # NOTE: compute_trial_scores must accept embedding_map and use embedding_map[key]
            scores, labels = self.compute_trial_scores(embedding_map, trials)

            eer = compute_eer(scores, labels)
            metrics[name] = {"eer": eer}

        # 6) choose monitor value
        vox_names = [n for n in self.trials if n.lower().startswith("vox")]
        monitor_name = vox_names[0] if vox_names else self.primary_trial_name

        # Keep the best-model metric even when optional W&B logging is disabled.
        self.log_dict(
            {"vox_eer": metrics[monitor_name]["eer"]},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            logger=self.use_wandb,
        )
        self.current_EER = metrics[monitor_name]["eer"]

    # -------------------------
    # Trial scoring
    # -------------------------
    @torch.inference_mode()
    def process_trial_chunk(self, trial_chunk, embedding_map):
        labels = []
        emb1_list, emb2_list = [], []

        for key1, key2, label in trial_chunk:
            emb1_list.append(embedding_map[key1])  # (T, D) on CPU
            emb2_list.append(embedding_map[key2])
            labels.append(label)

        # (B,T,D) CPU float32/float16
        a = torch.stack(emb1_list, dim=0)
        b = torch.stack(emb2_list, dim=0)

        # normalize on last dim
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)

        # cosine matrix per sample: (B,T,T) = (B,T,D) x (B,D,T)
        cos_mat = torch.matmul(a, b.transpose(1, 2))

        # score: (B,)
        scores = cos_mat.mean(dim=(1, 2))
        return scores, labels

    def compute_trial_scores(self, embedding_list, trials, chunk_size=10000):
        all_scores = []
        all_labels = []
        trial_chunk = []

        for trial in trials:
            trial_chunk.append((trial.key1, trial.key2, trial.label))

            if len(trial_chunk) >= chunk_size:
                scores_chunk, labels_chunk = self.process_trial_chunk(
                    trial_chunk, embedding_list
                )
                all_scores.append(scores_chunk)
                all_labels.extend(labels_chunk)
                trial_chunk = []

        if trial_chunk:
            scores_chunk, labels_chunk = self.process_trial_chunk(
                trial_chunk, embedding_list
            )
            all_scores.append(scores_chunk)
            all_labels.extend(labels_chunk)

        scores = torch.cat(all_scores, dim=0)
        return scores, all_labels
