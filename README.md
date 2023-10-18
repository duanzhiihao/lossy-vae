# Lossy Image Compression using Hierarchical VAEs

This repository contains authors' implementation of several deep learning-based methods related to lossy image compression. \
This project is under active development.

- [Models](#models)
- [Results](#results)
- [Install](#install)
- [Usage (compress, decompress, train, evaluation)](#usage)
- [Licenses](#license)

## Models
### Implemented Methods (Pre-Trained Models Available)
- **Lossy Image Compression with Quantized Hierarchical VAEs** [[arxiv](https://arxiv.org/abs/2208.13056)]
    - Published at WACV 2023,[ ***Best Algorithms Paper Award***](https://wacv2023.thecvf.com/node/174)
    - Abstract: a 12-layer VAE model named QRes-VAE. Good compression performance.
    - \[Code & pre-trained models\]: [lossy-vae/lvae/models/qres](lvae/models/qresvae)
- **QARV: Quantization-Aware ResNet VAE for Lossy Image Compression** [[arxiv](https://arxiv.org/abs/2302.08899)] [[ieee](https://ieeexplore.ieee.org/document/10274142)]
    - Published at TPAMI 2023
    - Abstract: an improved version of the previous model; **Variable-rate, faster decoding, better performance.**
    - \[Code & pre-trained models\]: [lossy-vae/lvae/models/qarv](lvae/models/qarv)
- **An Improved Upper Bound on the Rate-Distortion Function of Images** [[arxiv](https://arxiv.org/abs/2309.02574)]
    - Published at ICIP 2023
    - Abstract: a 15-layer VAE model used to estimate the information R(D) function. This model proves that -30% BD-rate w.r.t. VTM is theoretically achievable.
    - \[Code & pre-trained models\]: [lossy-vae/lvae/models/rd](lvae/models/rd)

### Features
**Progressive coding:** our models learn *a deep hierarchy of* latent variables and compress/decompress images in a *coarse-to-fine* fashion. This feature comes from the hierarchical nature of ResNet VAEs.
<p align="center">
  <img src="https://user-images.githubusercontent.com/24869582/187014268-405851e8-b8a5-47e3-b28d-7b5d4ac20316.png" width="756" height="300">
</p>

**Compression performance**: our models are powerful in terms of both rate-distortion and decoding speed. Please see the results section below.


## Results
### Bpp-PSNR results in JSON format
- Kodak images: [lossy-vae/results/kodak](results/kodak)
- Tecknick TESTIMAGES RGB 1200x1200: [lossy-vae/results/tecnick-rgb-1200](results/tecnick-rgb-1200)
- CLIC 2022 test set: [lossy-vae/results/clic2022-test](results/clic2022-test)

Notes on metric computation:
- Bpp and PSNR are first compute for each image and then averaged over all images in a dataset.
- Bpp is the saved file size (in bits) divided by # of image pixels.
- PSNR is computed in RGB space (not YUV).

### Encoding/decoding latency on CPU/GPU, and BD-rate
<div align="center">

| Model Name  | CPU* Enc. | CPU* Dec. | 3080 ti Enc. | 3080 ti Dec. | BD-rate* (lower is better) |
| :---------: | :-------: | :-------: | :----------: | :----------: | :------------------------: |
|  `qres34m`  |  0.899s   |  0.441s   |    0.116s    |    0.083s    |          -3.95 %           |
| `qarv_base` |  0.757s   |  0.295s   |    0.096s    |    0.063s    |          -7.26 %           |

</div>

*Time is the latency to encode/decode a 512x768 image, averaged over 24 Kodak images. Tested in plain PyTorch (v1.13 + CUDA 11.7) code, ie, no mixed-precision, torchscript, ONNX/TensorRT, etc. \
*CPU is Intel 10700k. \
*BD-rate is w.r.t. [VTM 18.0](https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/tree/VTM-18.0), averaged on three common test sets (Kodak, Tecnick TESTIMAGES, and CLIC 2022 test set).



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
### Get pre-trained weights
```python
from lvae import get_model
model = get_model('qarv_base', pretrained=True) # weights are downloaded automatically
model.eval()
model.compress_mode(True) # initialize entropy coding
```

### Compress images
Encode an image:
```python
model.compress_file('/path/to/image.png', '/path/to/compressed.bits')
```

Decode an image:
```python
im = model.decompress_file('/path/to/compressed.bits')
# im is a torch.Tensor of shape (1, 3, H, W). RGB. pixel values in [0, 1].
```


### Datasets
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

**Custom Dataset**
1. Prepare a folder containing images. The folder should contain only images (may contain subfolders).
2. Edit `lossy-vae/lvae/paths.py` such that `known_datasets['custom-name'] = '/path/to/my-dataset'`, where `custom-name` is the name of your dataset, and `/path/to/my-dataset` is the path to the folder containing images.
3. Then, you can use `custom-name` as the dataset name in the training/evaluation scripts.

### Training and evaluation scripts
Training and evaluation scripts vary from model to model. For example, `qres34m` uses fixed-rate train/eval scheme, while `qarv_base` uses variable-rate train/eval scheme. \
Detailed training/evaluation instructions are provided in each model's subfolder (see the section [Models](#models)).


## License
Code in this repository is freely available for non-commercial use.
