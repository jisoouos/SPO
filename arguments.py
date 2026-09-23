import argparse
from pathlib import Path


def _local_path(value):
    if not value.strip():
        raise argparse.ArgumentTypeError("Path must not be empty")
    return str(Path(value).expanduser())


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train WavLM + ECAPA from step zero and retain the best validation model.",
    )
    parser.add_argument("--train-samples", required=True, type=_local_path, metavar="TRAIN_LIST",
                        help="Training list: speaker_id integer_label audio_path")
    parser.add_argument("--vox-trials", required=True, type=_local_path, metavar="TRIAL_LIST",
                        help="Validation list: binary_label key1 key2 audio_path1 audio_path2")
    parser.add_argument("--noise-samples", required=True, type=_local_path, metavar="NOISE_LIST",
                        help="Noise list with one audio path per line")
    parser.add_argument("--reverb-samples", required=True, type=_local_path, metavar="RIR_LIST",
                        help="RIR list with one audio path per line")
    parser.add_argument("--output-dir", required=True, type=_local_path, metavar="OUTPUT_DIR",
                        help="New or empty directory for the single best model")
    options = parser.parse_args(argv)

    args = {
        # User-supplied locations; no fixed experiment name or machine paths.
        'output_dir'            : options.output_dir,
        'seed'                  : 4221,

        # Optional W&B logger: fill in your own values, then set use_wandb=True.
        # Never commit a real API key. None lets W&B generate the run name.
        'use_wandb'             : False,
        'wandb_entity'          : '{YOUR_WANDB_USER_NAME}',
        'wandb_api_key'         : '{YOUR_WANDB_API_KEY}',
        'wandb_project'         : 'SPO',
        'wandb_name'            : None,

        # data
        'train_samples'         : options.train_samples,
        'test_trials'           : {'vox': options.vox_trials},
        'da_noise_samples'      : options.noise_samples,
        'da_reverb_samples'     : options.reverb_samples,

        # hyper parameters
        'train_steps'               : 18000,
        'eval_interval_steps'       : 300,

        'batch_size'                : 1536,
        'accumulate_grad_batches'   : 8,
        'lr_max'                    : 5e-2,
        'lr_min'                    : 3e-4,
        'weight_decay'              : 5e-4,
        'gradient_clip_val'         : 1000,

        # data processing
        'samplerate'                : 16000,
        'crop_size'                 : 16000 * 3, # 3sec
        'test_crop_size'            : 16000 * 4, # 4sec
        'TTA_seg_num'               : 10,
        'p_da'                      : 0.6,
        'snr_range'                 : (5, 15),

        # frontend
        'ssl_model'                 : 'microsoft/wavlm-large',
        'num_hidden_layers'         : 24,
        'hidden_size'               : 1024,
        'have_ssl_model'              : True, # True: SSL + ECAPA, False: fbank + ECAPA

        # backend

        'embedding_size'            : 256,
        'ecapa_channel'             : 1024,

        # loss
        'aam_margin'                : 0.2,
        'aam_margin_final'          : 0.4,
        'aam_scale'                 : 30,
        'topk_panalty'              : (5, 0.1),

        #proposed
        'buffer_size'              : 400, # cosine scores retained per class
        'buffer_stack_step'        : 7000, # start collecting cosine scores
        'loss_alpha_step'          : 10000, # start two-path CDF weighting
    }

    return args


if __name__ == "__main__":
    get_args()
