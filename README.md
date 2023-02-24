# Lossy Image Compression using Hierarchical VAEs

This repository is still under construction.

## Implemented Methods
- Lossy Image Compression with Quantized Hierarchical VAEs
    - **WACV 2023, Best Algorithms Paper Award** [[arXiv](https://arxiv.org/abs/2208.13056)]
    - Code coming soon
- An Improved Upper Bound on the Rate-Distortion Function of Images
    - Code at [lossy-vae/lvae/models/rd](lvae/models/rd)
- QARV: Quantization-Aware ResNet VAE for Lossy Image Compression
    - Technical report [[arXiv](https://arxiv.org/abs/2302.08899)]
    - Code at [lossy-vae/lvae/models/qarv](lvae/models/qarv)


## Install
**Requirements**:
- Python
- PyTorch >= 1.9 : https://pytorch.org/get-started/locally
- tqdm : `conda install tqdm`
- CompressAI : https://github.com/InterDigitalInc/CompressAI
- **timm >= 0.8.0** : https://github.com/huggingface/pytorch-image-models

**Download and Install**:
1. Download the repository;
2. Modify the dataset paths in `lossy-vae/lvae/paths.py`.
3. [Optional] pip install the repository in development mode:
```
cd /pasth/to/lossy-vae
python -m pip install -e .
```


## Usage
Detailed usage is provided in each model's folder


## Prepare Datasets for Training and Evaluation
**COCO**
1. Download the COCO dataset "2017 Train images [118K/18GB]" from https://cocodataset.org/#download
2. Unzip the images anywhere, e.g., at `/path/to/datasets/coco/train2017`
3. Edit `lossy-vae/lvae/paths.py` such that
```
known_datasets['coco-train2017'] = '/path/to/datasets/coco/train2017'
```

**Kodak** ([link](http://r0k.us/graphics/kodak)),
**Tecnick TESTIMAGES** ([link](https://testimages.org/)),
and **CLIC** ([link](http://compression.cc/))
```
python scripts/download-dataset.py --name kodak         --datasets_root /path/to/datasets
                                          clic2022-test
                                          tecnick
```
Then, edit `lossy-vae/lvae/paths.py` such that `known_datasets['kodak'] = '/path/to/datasets/kodak'`, and similarly for other datasets.



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
