# Lossy Image Compression with Quantized Hierarchical VAEs
QRes-VAE (Quantized ResNet VAE) is a neural network model for lossy image compression.
It is based on the ResNet VAE architecture.

**Paper:** Lossy Image Compression with Quantized Hierarchical VAEs, **WACV 2023 Best Paper Award (Algorithms track)** \
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

**QRes-VAE (17M):** a smaller model trained on the CelebA dataset for ablation study.
```
model = lvae.get_model('qres17m', lmb=1, pretrained=True)
# pre-trained models are provided for lmb in {1, 2, 4, 8, 16, 32, 64}
```

**QRes-VAE (34M, lossless):** a lossless compression model. Better than PNG but not as good as WebP.
```
model = lvae.get_model('qres34m_lossless', pretrained=True)
```

Note: the `lmb` discussed above is the multiplier for MSE during training. I.e., `loss = rate + lmb * mse`.
A larger `lmb` produces a higher bit rate and lower distortion (ie, better reconsturction quality).


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


## Evaluate lossy compression efficiency
- Rate-distortion curve: `python eval-fix-rate.py --model qres34m --dataset kodak --device cuda:0`
    - Supported models: `qres34m`, `qres17m`
    - `kodak` can be replaced by any other dataset name in `lvae.paths.known_datasets`
- Estimate end-to-end flops: [scripts\qresvae\estimate-flops.ipynb](../../../scripts/qresvae/estimate-flops.ipynb)


## Training
Training is done by minimizing the `stats['loss']` term returned by the model's `forward()` function.

Training scripts comming soon...

<!-- ### Single GPU training
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
``` -->


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
