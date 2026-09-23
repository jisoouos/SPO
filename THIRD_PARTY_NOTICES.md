# Third-party notices

## Microsoft WavLM-Large

This project uses **WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
Speech Processing**, by Sanyuan Chen and colleagues at Microsoft Research.

- Model source: https://huggingface.co/microsoft/wavlm-large
- Project: https://github.com/microsoft/unilm/tree/master/wavlm
- Paper: https://arxiv.org/abs/2110.13900
- License linked by the official model card:
  https://github.com/microsoft/UniSpeech/blob/main/LICENSE
- License URI: https://creativecommons.org/licenses/by-sa/3.0/legalcode

The official model card links to **Creative Commons Attribution-ShareAlike 3.0
Unported**. The bundled checkpoint includes the WavLM frontend tensors from the
supplied trained model; this frontend was frozen during training. Those tensors
are redistributed without numerical changes in a new weights-only container,
together with the trained ECAPA backend and speaker-center tensors. The upstream
license and its warranty disclaimer continue to apply to the WavLM component.
No endorsement by Microsoft or the original authors is implied.

## Dependencies and data

PyTorch, Lightning, Transformers, librosa, and the other installed packages retain
their respective licenses. The Dockerfile references NVIDIA's PyTorch container;
the base image is not redistributed in this repository and its terms still apply.
No speech recordings, dataset file lists, or augmentation datasets are included.

This notice records third-party terms; it does not assign a new license to the
project's original code or independently trained components.
