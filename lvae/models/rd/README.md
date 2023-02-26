# An Improved Upper Bound on the Rate-Distortion Function of Images

VAEs can be used to upper-bound the **information rate-distortion function**, R(D), of random variables.
We develop a model based on the ResNet VAE architecture and use it to upper-bound the R(D) of images.
The resulting upper bound shows that more than 30% BD-rate reduction w.r.t. VTM 18.0 is theoretically achievable.
This means that existing codecs are far from optimum.

**Paper:** TBD \
**Arxiv:** TBD

### Note: an upper bound for R(D), measured by bpp-MSE, is a *lower bound* in the PSNR-bpp representation.


## Features
- **A continuous rate-distortion curve.** We use modern variable-rate compression techniques to achieve a continuous upper-bound function on R(D).
- **A better bound.** Our theoretical PSNR-bpp curve is more than 30% better than VTM 18.0.


## Pre-trained models

|      Name       | `lmb_range` | Param  | Bpp range (Kodak) | Kodak BD-rate | Tecknick BD-rate | CLIC BD-rate |
| :-------------: | :---------: | :----: | :---------------: | :-----------: | :--------------: | :----------: |
| `rd_model_base` | `[4, 2048]` | 186.7M |   0.079 - 1.957   |     -31.4     |      -34.5       |    -31.9     |

*BD-rate is w.r.t. VTM 18.0. Lower BD-rate is better.


<!-- ## Usage
TBD -->


## Evaluate the Upper Bound for R(D) (i.e., a lower bound in the PSNR-bpp plane)
```
python eval-var-rate.py --model rd_model_base --dataset_name kodak --device cuda:0
```
- `kodak` can be replaced by any other dataset name in `lvae.paths.known_datasets`

Note: due to the stochastic nature of VAEs, the evaluation results may vary slightly from run to run.


## Training

The following commands reproduce the training of our pre-trained model. \
Training progress is tracked using Weights & Biases.
By default, the run locates at https://wandb.ai/home > *default* project > *var-rate-exp* group.

### Single GPU training
```
python train-var-rate.py --model rd_model_base --batch_size 16 --iterations 200_000 --workers 4 --wbmode online
```

### Single GPU training, using GPU id=2
```
CUDA_VISIBLE_DEVICES=2 python train-var-rate.py --model rd_model_base --batch_size 16 --iterations 200_000 --workers 4 --wbmode online
```

### Multi-GPU training, using two GPUs id=4,5
```
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 train-var-rate.py --model rd_model_base --batch_size 16 --iterations 200_000 --workers 4 --wbmode online
```


## Citation
**TBD**
