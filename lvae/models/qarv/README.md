# QARV: Quantization-Aware ResNet VAE for Lossy Image Compression

QARV (Quantization-Aware ResNet VAE) is an improved version of QRes-VAE.
- **Technical report**
- **Arxiv:** https://arxiv.org/abs/2302.08899


## Features
- ***Continuously* variable-rate:** QARV can compress images at any target bitrates, from around 0.2 bpp to 2.2 bpp, with a single model.
- **Faster decoding:** QARV is faster than QRes-VAE in terms of decoding speed.
- **Strong R-D performance:** QARV achieves -6.537 BD-rate w.r.t. VTM 18.0.


## Pre-trained models

|     Name    |  `lmb_range` | Param | Bpp range (Kodak) | Kodak BD-rate | Tecknick BD-rate | CLIC BD-rate |
|:-----------:|:------------:|:-----:|:-----------------:|:-------------:|:----------------:|:------------:|
| `qarv_base` | `[16, 2048]` | 93.4M |   0.208 - 2.210   |     -5.9 %    |      -8.9 %      |    -6.9 %    |

*BD-rate is w.r.t. VTM 18.0, lower is better.

**Load pre-trained models by**
```
import lvae

model = lvae.get_model('qarv_base', pretrained=True)
```

## Usage
### Image compression
```
model = lvae.get_model('qarv_base', pretrained=True)
model.eval()
model.compress_mode(True) # initialize entropy coding

# compress
model.compress_file('path/to/image.png', 'path/to/compressed.bin')

# decompress
im = model.decompress_file('path/to/compressed.bin')

# im is a torch.Tensor of shape (1, 3, H, W), RGB, pixel values in [0, 1]
```


## Evaluation
The following command evaluates the pre-trained `qarv_base` model on the `kodak` dataset and produces a rate-distortion curve.
```
python eval-var-rate.py --model qarv_base --dataset_name kodak --device cuda:0
```
- `kodak` can be replaced by any other dataset name in `lvae.paths.known_datasets`


## Training
The following commands **reproduce the training of our model used in the paper**. \
Training progress is tracked using Weights & Biases.
By default, the run locates at https://wandb.ai/home > *default* project > *var-rate-exp* group.

### Single GPU training
```
python train-var-rate.py --model qarv_base --batch_size 32 --iterations 2_000_000 --workers 8 --wbmode online
```

### Single GPU training, using the GPU with id=2
```
CUDA_VISIBLE_DEVICES=2 python train-var-rate.py --model qarv_base --batch_size 32 --iterations 2_000_000 --workers 8 --wbmode online
```

### Multi-GPU training, using two GPUs id=4,5
```
CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node 2 train-var-rate.py --model qarv_base --batch_size 16 --iterations 2_000_000 --workers 8 --wbmode online
```


## Citation
```
@article{duan2023qarv,
    title={QARV: Quantization-Aware ResNet VAE for Lossy Image Compression},
    author={Duan, Zhihao and Lu, Ming and Ma, Jack and Ma, Zhan and Zhu, Fengqing},
    journal={arXiv preprint arXiv:2302.08899},
    year={2023},
    month=Feb
}
```
