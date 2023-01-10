# Empirical Upper Bound on the Rate-Distortion Function

**Paper:** TBD, published at TBD \
**Arxiv:** TBD


## Features
TBD


## Pre-trained models
TBD


## Usage
TBD


## Evaluation
TBD


## Training

### Single GPU training
```
python train-var-rate.py --model rd_model_a --batch_size 16 --iterations 1_000_000 --workers 4 --wbmode online
```
Training progress is tracked using `wandb`:  \
By default, the run in in the https://wandb.ai/home > `default` project > `var-rate-exp` group.

### Single GPU training, using GPU id=2
```
CUDA_VISIBLE_DEVICES=2 python train-var-rate.py --model rd_model_a --batch_size 16 --iterations 1_000_000 --workers 4 --wbmode online
```

### Multi-GPU training, using two GPUs id=4,5
```
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 train-var-rate.py --model rd_model_a --batch_size 16 --iterations 1_000_000 --workers 4 --wbmode online
```


## Citation
**TBD**
