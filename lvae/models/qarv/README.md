# QARV: Quantization-Aware ResNet VAE for Lossy Image Compression

**Paper:** TBD, published at TBD \
**Arxiv:** TBD


## Features
TBD


## Pre-trained models
Note: BD-rate is w.r.t. VTM 18.0

|   Name    | Bpp range (Kodak) | BD-rate (Kodak) |
|:---------:|:-----------------:|:---------------:|
| qarv_base |  `0.217 - 2.219`  |     -6.537      |


## Usage
TBD


## Evaluation
```
python eval-var-rate.py --model qarv_base --dataset_name kodak --device cuda:0
```


## Training

### Single GPU training
```
python train-var-rate.py --model qarv_base --batch_size 32 --iterations 2_000_000 --workers 8 --wbmode online
```
Training progress is tracked using `wandb`.
By default, the run in in the https://wandb.ai/home > `default` project > `var-rate-exp` group.

### Single GPU training, using the GPU with id=2
```
CUDA_VISIBLE_DEVICES=2 python train-var-rate.py --model qarv_base --batch_size 32 --iterations 2_000_000 --workers 8 --wbmode online
```

### Multi-GPU training, using two GPUs id=4,5
```
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 train-var-rate.py --model qarv_base --batch_size 16 --iterations 2_000_000 --workers 8 --wbmode online
```


## Citation
**TBD**
