import multiprocessing as mp
import os
import warnings

import pytorch_lightning as pl
import torch
from transformers import WavLMModel

from arguments import get_args
from data_module import DataModule
from loss import AAMSoftmax
from model_module import ModelModule
from models import ECAPA_TDNN


def create_wandb_logger(args):
    """Create an optional metric logger without uploading credentials or source."""
    if not args["use_wandb"]:
        return False

    for field in ("wandb_entity", "wandb_api_key", "wandb_project"):
        value = args[field]
        if not isinstance(value, str) or not value.strip() or "{YOUR_" in value:
            raise ValueError(f"Fill in {field} in arguments.py before enabling W&B.")

    import wandb
    from pytorch_lightning.loggers import WandbLogger

    # Authenticate through the process environment, not the run configuration.
    os.environ["WANDB_API_KEY"] = args["wandb_api_key"].strip()
    config_keys = (
        "seed", "train_steps", "eval_interval_steps", "batch_size",
        "accumulate_grad_batches", "lr_max", "lr_min", "weight_decay",
        "gradient_clip_val", "samplerate", "crop_size", "test_crop_size",
        "TTA_seg_num", "p_da", "snr_range", "num_hidden_layers", "hidden_size",
        "embedding_size", "ecapa_channel", "aam_margin", "aam_margin_final",
        "aam_scale", "topk_panalty", "buffer_size", "buffer_stack_step",
        "loss_alpha_step",
    )
    return WandbLogger(
        project=args["wandb_project"].strip(),
        entity=args["wandb_entity"].strip(),
        name=args["wandb_name"],
        save_dir=args["output_dir"],
        log_model=False,
        config={key: args[key] for key in config_keys},
        settings=wandb.Settings(
            disable_git=True,
            disable_code=True,
            save_code=False,
            console="off",
        ),
    )


class RampAugmentProbability(pl.Callback):
    def __init__(self, train_steps: int, p_max: float, ramp_portion: float):
        self.train_steps = train_steps
        self.p_max = p_max
        self.ramp_portion = ramp_portion

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        t = min(trainer.global_step / (self.train_steps * self.ramp_portion), 1.0)
        trainer.train_dataloader.dataset.set_augment_probability(self.p_max * t)


class RampAAMMargin(pl.Callback):
    def __init__(self, m_start: float, m_end: float, start_step: int, end_step: int):
        if end_step <= start_step:
            raise ValueError(f"end_step must be greater than start_step: {start_step} -> {end_step}")
        self.m_start = m_start
        self.m_end = m_end
        self.start_step = start_step
        self.end_step = end_step

    def _margin_at_step(self, step: int) -> float:
        if step <= self.start_step:
            return self.m_start
        if step >= self.end_step:
            return self.m_end
        t = (step - self.start_step) / (self.end_step - self.start_step)
        return self.m_start + (self.m_end - self.m_start) * t

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        pl_module.criterion_sv.set_margin(self._margin_at_step(int(trainer.global_step)))


def main():
    warnings.filterwarnings(
        "ignore",
        message=".*apply_effects_tensor has been deprecated.*",
        category=UserWarning,
    )
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    args = get_args()
    pl.seed_everything(args["seed"], workers=True)

    # Never delete an earlier run or mix its best model with this run.
    output_dir = args["output_dir"]
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Output directory is not empty; choose a new --output-dir: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    # The logger is lazy; no run is initialized when this module is imported.
    experiment_logger = create_wandb_logger(args)

    datamodule = DataModule(args, aug_p_shared=mp.Value("d", 0.0))

    # Keep the pretrained frontend; training itself starts at step zero.
    frontend = None
    if args["have_ssl_model"]:
        frontend = WavLMModel.from_pretrained(
            args["ssl_model"],
            revision="main",
            ignore_mismatched_sizes=False,
        )
        for parameter in frontend.parameters():
            parameter.requires_grad = False

    classifier = ECAPA_TDNN(
        args["have_ssl_model"],
        args["num_hidden_layers"],
        args["hidden_size"],
        args["ecapa_channel"],
        args["embedding_size"],
    )
    criterion_sv = AAMSoftmax(
        embedding_size=args["embedding_size"],
        num_class=len(datamodule.train_db.train_set),
        margin=args["aam_margin"],
        scale=args["aam_scale"],
        buffer_size=args["buffer_size"],
        buffer_stack_step=args["buffer_stack_step"],
        loss_alpha_step=args["loss_alpha_step"],
        topk_panelty=args["topk_panalty"],
    )

    best_model = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir,
        filename="best-{step:06d}-{vox_eer:.4f}",
        monitor="vox_eer",
        mode="min",
        save_top_k=1,
        save_last=False,
        save_weights_only=True,
        every_n_epochs=1,
        save_on_train_epoch_end=False,
    )
    callbacks = [
        best_model,
        RampAugmentProbability(args["train_steps"], args["p_da"], 0.2),
        RampAAMMargin(
            m_start=args["aam_margin"],
            m_end=args["aam_margin_final"],
            start_step=args["loss_alpha_step"],
            end_step=args["train_steps"],
        ),
    ]

    model_module = ModelModule(
        args,
        classifier,
        criterion_sv,
        trials=datamodule.train_db.trials,
        frontend=frontend,
    )
    trainer = pl.Trainer(
        max_steps=args["train_steps"],
        accelerator="gpu",
        devices=-1,
        benchmark=False,
        logger=experiment_logger,
        log_every_n_steps=50,
        callbacks=callbacks,
        strategy="auto",
        gradient_clip_val=args["gradient_clip_val"],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        val_check_interval=args["eval_interval_steps"] * args["accumulate_grad_batches"],
        accumulate_grad_batches=args["accumulate_grad_batches"],
    )
    trainer.fit(model_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
