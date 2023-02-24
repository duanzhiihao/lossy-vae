# Lossy Image Compression using Hierarchical VAEs

This repository contains authors' implementation of several deep hierarchical VAE-based methods related to lossy image compression. \
Code is under active development, and the API is subject to change.

### Features
**Progressive coding:** our models learn *a deep hierarchy of* latent variables and compress/decompress images in a *coarse-to-fine* fashion. This feature comes from the hierarchical nature of ResNet VAEs.
<p align="center">
  <img src="https://user-images.githubusercontent.com/24869582/187014268-405851e8-b8a5-47e3-b28d-7b5d4ac20316.png" width="756" height="300">
</p>

**Compression efficiency**: our models are powerful in terms of both rate-distortion (bpp-PSNR) and decoding speed.

<div align="center">

|  Model Name | CPU* Enc. | CPU* Dec. | 1080 ti Enc. | 1080 ti Dec. | BD-rate* |
|:-----------:|:---------:|:---------:|:------------:|:------------:|:--------:|
|  `qres34m`  |   0.899s  |   0.441s  |    0.213s    |    0.120s    |   -3.95  |
| `qarv_base` |   0.880s  |   0.295s  |    0.211s    |    0.096s    |   -6.54  |

</div>

*Time is the latency to encode/decode a 512x768 image, averaged over 24 Kodak images. Tested in plain PyTorch (v1.13 + CUDA 11.7) code, ie, no mixed-precision, torchscript, ONNX/TensorRT, etc. \
*CPU is Intel 10700k. \
*BD-rate is w.r.t. [VTM 18.0](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/tree/VTM-18.0), averaged on three common test sets (Kodak, Tecnick TESTIMAGES, and CLIC 2022 test set).


## Implemented Methods - Pre-Trained Models and Training Scripts Available
- **Lossy Image Compression with Quantized Hierarchical VAEs** [[arXiv](https://arxiv.org/abs/2208.13056)]
    - Published at WACV 2023, ***Best Algorithms Paper Award***
    - [Abstract]: a 12-layer VAE model named QRes-VAE. Good compression performance.
    - [Code]\: [lossy-vae/lvae/models/qres](lvae/models/qres)
- **An Improved Upper Bound on the Rate-Distortion Function of Images**
    - [Abstract]: a 15-layer VAE model used to estimate the information R(D) function. Shows that -30% BD-rate w.r.t. VTM is theoretically achievable.
    - [Code]\: [lossy-vae/lvae/models/rd](lvae/models/rd)
- **QARV: Quantization-Aware ResNet VAE for Lossy Image Compression** [[arXiv](https://arxiv.org/abs/2302.08899)]
    - Technical report
    - [Abstract]: improved version of QRes-VAE. Variable-rate, faster decoding, better performance.
    - [Code]\: [lossy-vae/lvae/models/qarv](lvae/models/qarv)


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
