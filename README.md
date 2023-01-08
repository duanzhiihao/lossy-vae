# Lossy Image Compression using Hierarchical VAEs


## Install
**Requirements**:
- Python, `pytorch>=1.9`, `tqdm`, `compressai` ([link](https://github.com/InterDigitalInc/CompressAI)), `timm>=0.5.4` ([link](https://github.com/rwightman/pytorch-image-models)).


**Download and Install**:
1. Download the repository;
2. Modify the dataset paths in `lossy-vae/lvae/paths.py`.


## Implemented Models
TBD


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
