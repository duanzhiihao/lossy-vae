# Lossy Image Compression with Quantized Hierarchical VAEs
QRes-VAE (Quantized ResNet VAE) is a neural network model for lossy image compression.
It is based on the ResNet VAE architecture.

**Paper:** Lossy Image Compression with Quantized Hierarchical VAEs, [ ***WACV 2023 Best Algorithms Paper Award***](https://wacv2023.thecvf.com/node/174) \
**Arxiv:** https://arxiv.org/abs/2208.13056


## Features

- **Progressive coding:** the QRes-VAE model learns a hierarchy of features. It compresses/decompresses images in a coarse-to-fine fashion. \
Note: images below are from the CelebA dataset and COCO dataset, respectively.
<p align="center">
  <img src="https://user-images.githubusercontent.com/24869582/187014268-405851e8-b8a5-47e3-b28d-7b5d4ac20316.png" width="756" height="300">
</p>

- **Lossy compression efficiency:** the QRes-VAE model has a competetive rate-distortion performance, especially at higher bit rates.
<p align="center">
  <img src="https://user-images.githubusercontent.com/24869582/187009894-f2897f2e-be5a-4ba5-b1aa-2b8c4269f43e.png" width="774" height="300">
</p>


## Pre-trained models
**QRes-VAE (34M):** our main model for natural image compression.
```
model = lvae.get_model('qres34m', lmb=16, pretrained=True)
# pre-trained models are provided for lmb in {16, 32, 64, 128, 256, 512, 1024, 2048}
```
Note: `lmb` is the multiplier for MSE during training. I.e., `loss = rate + lmb * mse`.
A larger `lmb` produces a higher bit rate and lower distortion (better reconsturction quality).

**QRes-VAE (17M):** a smaller model trained on the CelebA dataset for ablation study.
```
model = lvae.get_model('qres17m', lmb=1, pretrained=True)
# pre-trained models are provided for lmb in {1, 2, 4, 8, 16, 32, 64}
```

**QRes-VAE (34M, lossless):** a lossless compression model. Better than PNG but not as good as WebP.
```
model = lvae.get_model('qres34m_lossless', pretrained=True)
```


## Usage
### Image compression
```
import lvae

model = lvae.get_model('qres34m', lmb=16, pretrained=True)
model.eval()
model.compress_mode(True) # initialize entropy encoder

# compress
model.compress_file('path/to/image.png', 'path/to/compressed.bin')

# decompress
im = model.decompress_file('path/to/compressed.bin')

# im is a torch.Tensor of shape (1, 3, H, W), RGB, pixel values in [0, 1]
```

### As a VAE generative model
- **Progressive decoding**: [scripts/qresvae/progressive-decoding.ipynb](../../../scripts/qresvae/progressive-decoding.ipynb)
- **Sampling**: [scripts/qresvae/uncond-sampling.ipynb](../../../scripts/qresvae/uncond-sampling.ipynb)
- **Latent space interpolation**: [scripts/qresvae/latent-interpolation.ipynb](../../../scripts/qresvae/latent-interpolation.ipynb)
- **Inpainting**: [scripts/qresvae/inpainting.ipynb](../../../scripts/qresvae/inpainting.ipynb)


## Evaluate lossy compression
- Rate-distortion curve: `python eval-fix-rate.py --model qres34m --dataset kodak --device cuda:0`
    - Supported models: `qres34m`, `qres17m`
    - `kodak` can be replaced by any other dataset name in `lvae.paths.known_datasets`
- Estimate end-to-end flops: [scripts/qresvae/estimate-flops.ipynb](../../../scripts/qresvae/estimate-flops.ipynb)


## Evaluate lossless compression
- Demo: how to compress/decompress a single image: [scripts/qresvae/demo-lossless.ipynb](../../../scripts/qresvae/demo-lossless.ipynb)
- Compute lossless compression bpp: `python scripts/qresvae/evaluate-lossless.py --root /path/to/dataset`. For Kodak images, the bpp is 10.369.


## Training
Training is done by minimizing the `stats['loss']` term returned by the model's `forward()` function.

We provide the training script in `train-fix-rate.py`.
Training progress is tracked using Weights & Biases. By default, the run locates at https://wandb.ai/home > *default* project > *fix-rate-exp* group.

### Single GPU training using default settings
```
python train-fix-rate.py --model qres34m
```

### Specify `lmb`
```
python train-fix-rate.py --model qres34m --model_args 'lmb=2048'
```
The training loss function is loss = Rate + `lmb` * distortion.
- A large `lmb` results in high PSNR but high bpp
- A small `lmb` results in low PSNR but low bpp

### Specify GPU id=2, batch size, iterations, number of CPU workers, and use online W&B logging
```
CUDA_VISIBLE_DEVICES=2 python train-fix-rate.py --model qres34m --model_args 'lmb=2048' --batch_size 32 --iterations 600_000 --workers 8 --wbmode online
```

### Multi-GPU training, using two GPUs (id=2,3)
```
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 train-fix-rate.py --model qres34m --model_args 'lmb=2048' --batch_size 16 --iterations 600_000 --workers 8 --wbmode online
```
Batch size is per-GPU. I.e., the total batch size is 16 x 2 = 32 in this example.

### Reproduce our paper's results
Our training code is slightly updated from the one we used in our paper. To approximately reproduce our paper's results, please use the following command:
```
torchrun --nproc_per_node 4 train-fix-rate.py --model qres34m --model_args 'lmb=2048' --batch_size 16 --iterations 600_000 --workers 8 --wbmode online
```
This requires 4 GPUs, each having >16GB memory.


## Citation
```
@article{duan2023qres,
    title={Lossy Image Compression with Quantized Hierarchical VAEs},
    author={Duan, Zhihao and Lu, Ming and Ma, Zhan and Zhu, Fengqing},
    journal={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    pages={198--207},
    year={2023},
    month=Jan
}
```
