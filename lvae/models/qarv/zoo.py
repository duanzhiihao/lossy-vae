import torch
from torch.hub import load_state_dict_from_url

from lvae.models.registry import register_model
import lvae.models.common as common
import lvae.models.qarv.model as qarv


@register_model
def qarv_base(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(6)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(3)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    model = qarv.VariableRateLossyVAE(cfg)

    if pretrained is True:
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_base-2022-dec-12.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


from pytorch_msssim import ms_ssim
def msssim_loss(fake, real):
    real = real * 0.5 + 0.5
    assert 0.0 <= real.min() and real.max() <= 1.0
    ms = ms_ssim(fake * 0.5 + 0.5, real, data_range=1.0, size_average=False)
    return 1 - ms

@register_model
def qarv_base_ms(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(6)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(3)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    model = qarv.VariableRateLossyVAE(cfg)

    model.distortion_name = 'msim'
    model.distortion_func = msssim_loss

    if pretrained is True:
        raise NotImplementedError()
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


def log_msssim_loss(fake, real):
    real = real * 0.5 + 0.5
    assert 0.0 <= real.min() and real.max() <= 1.0
    ms = ms_ssim(fake * 0.5 + 0.5, real, data_range=1.0, size_average=False)
    assert ms.min() > 0.0 and ms.max() <= 1.0, f'{ms.min()=}, {ms.max()=}'
    return -1.0 * torch.log(ms)

@register_model
def qarv_base_logms(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(6)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(3)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    model = qarv.VariableRateLossyVAE(cfg)

    model.distortion_name = 'log-ms'
    model.distortion_func = log_msssim_loss

    if pretrained is True:
        raise NotImplementedError()
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


import torch.nn.functional as tnf
def l1_loss(fake, real):
    mae = tnf.l1_loss(fake, real, reduction='none').mean(dim=(1,2,3))
    return mae

@register_model
def qarv_base_l1(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(6)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(3)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    model = qarv.VariableRateLossyVAE(cfg)

    model.distortion_name = 'mae'
    model.distortion_func = l1_loss

    if pretrained is True:
        raise NotImplementedError()
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


@register_model
def qarv_3z(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(6)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, None, 288, 24]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        # *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(1)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(1)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    model = qarv.VariableRateLossyVAE(cfg)

    if pretrained is True:
        raise NotImplementedError()
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_base-2022-dec-12.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


@register_model
def qarv_5z(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(6)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 64, 288, 12]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(1)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(1)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(2)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    model = qarv.VariableRateLossyVAE(cfg)

    if pretrained is True:
        raise NotImplementedError()
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_base-2022-dec-12.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


@register_model
def qarv_7z(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(6)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 144, 12]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(2)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    model = qarv.VariableRateLossyVAE(cfg)

    if pretrained is True:
        raise NotImplementedError()
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_base-2022-dec-12.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


@register_model
def qarv_11z(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(6)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(3)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(3)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(4)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(8)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    model = qarv.VariableRateLossyVAE(cfg)

    if pretrained is True:
        raise NotImplementedError()
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_base-2022-dec-12.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


@torch.inference_mode()
def main():
    model = qarv_3z()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params/1e6:.2f} M')
    debug = 1

if __name__ == '__main__':
    main()
