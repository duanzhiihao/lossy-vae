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
### Data preparation
**COCO**
1. Download the COCO dataset `2017 Train images [118K/18GB]` from https://cocodataset.org/#download
2. Unzip the images anywhere, e.g., to `/path/to/coco/train2017`
3. Edit `lossy-vae/lvae/paths.py` such that `known_datasets['coco-train2017'] = '/path/to/coco/train2017'`

**Kodak**
1. Download the 24 Kodak images from http://r0k.us/graphics/kodak
2. Put them anywhere, e.g., at `/path/to/kodak`
3. Edit `lossy-vae/lvae/paths.py` such that `known_datasets['kodak'] = '/path/to/kodak'`


### Single GPU training
```
python train-var-rate.py --model rd_model_a --batch_size 16 --iterations 1_000_000 --workers 4 --wbmode online
```
Training progress is tracked using `wandb`: https://wandb.ai/home

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
