# Lossy Image Compression using Hierarchical VAEs


## Implemented Models
TBD


## Install
**Requirements**:
- Python, `pytorch>=1.9`, `tqdm`, `compressai` ([link](https://github.com/InterDigitalInc/CompressAI)), `timm>=0.5.4` ([link](https://github.com/rwightman/pytorch-image-models)).

**Download and Install**:
1. Download the repository;
2. Modify the dataset paths in `lossy-vae/lvae/paths.py`.

## Datasets Download
**COCO**
1. Download the COCO dataset `2017 Train images [118K/18GB]` from https://cocodataset.org/#download
2. Unzip the images anywhere, e.g., to `/path/to/coco/train2017`
3. Edit `lossy-vae/lvae/paths.py` such that `known_datasets['coco-train2017'] = '/path/to/coco/train2017'`

**Kodak**
1. Download the 24 Kodak images from http://r0k.us/graphics/kodak
2. Put them anywhere, e.g., at `/path/to/kodak`
3. Edit `lossy-vae/lvae/paths.py` such that `known_datasets['kodak'] = '/path/to/kodak'`


## Usage
Detailed usage is provided in each model's folder



<!-- ## Evaluation
TBD

## Training
Training is done by minimizing the `stats['loss']` term returned by the model's `forward()` function.

### Data preparation
We used the COCO dataset for training, and the Kodak images for periodic evaluation.
- COCO: https://cocodataset.org
- Kodak: http://r0k.us/graphics/kodak

### Single GPU training
TBD

### Multi-GPU training
```
torchrun --nproc_per_node 2 train-var-rate.py --model qarv_base --model_args lmb_range=[16,2048] --batch_size 16 --iterations 2_000_000 --workers 6 --wbproject topic --wbgroup exp-lmb16-1024 --wbmode online
``` -->


## License
TBD
