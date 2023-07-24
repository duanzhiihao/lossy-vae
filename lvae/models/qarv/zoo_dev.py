import torch
from torch.hub import load_state_dict_from_url

from lvae.models.registry import register_model
import lvae.models.common as common
import lvae.models.qarv.model_dev as qarv


@register_model
def qarvb_sfpl(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # maximum downsampling factor
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    res_block = qarv.ConvNeXtBlockAdaLNSoftPlus
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
        *[qarv.VRLVBlockBaseSoftplus(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBaseSoftplus(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBaseSoftplus(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(3)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBaseSoftplus(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
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
def qarvb_mlpp(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # maximum downsampling factor
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    # _emb_dim = cfg['lmb_embed_dim'][1]

    # model configuration
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*3]

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

    dec_dims = [ch*3, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBaseMLPprior(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=3) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBaseMLPprior(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(2)],
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        *[qarv.VRLVBlockBaseMLPprior(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(3)],
        res_block(dec_dims[2], kernel_size=5, mlp_ratio=2),
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        *[qarv.VRLVBlockBaseMLPprior(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
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


@register_model
def qarv2_b3(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [192, ch*3, ch*4, ch*4, ch*4]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(7)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(6)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], kernel_size=7),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(6)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], kernel_size=5),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], kernel_size=3),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*4, ch*4, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=2) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=2),
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
        raise NotImplementedError()
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_base-2022-dec-12.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    elif pretrained: # str or Path
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    return model


@register_model
def qarv2_s6(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*3, ch*3]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], kernel_size=7),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], kernel_size=5),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], kernel_size=3),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*3, ch*3, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 12]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=2) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=2),
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
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(2)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(4)],
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
def qarv2_s8(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*3, ch*3]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(3)],
        common.SetKey('enc_s8'),
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(2)],
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(3)],
        common.SetKey('enc_s16'),
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(2)],
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(3)],
        common.SetKey('enc_s32'),
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(2)],
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*3, ch*3, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 12]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=2) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=2),
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
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(2)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(4)],
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
def qarv2_s6_11input(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    cfg['im_shift'] = -0.5
    cfg['im_scale'] = 2.0
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*3, ch*3]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], kernel_size=7),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], kernel_size=5),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], kernel_size=3),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*3, ch*3, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 12]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=2) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=2),
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
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(2)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(4)],
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
def qarv2_s7(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*3, ch*3]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], kernel_size=7),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], kernel_size=5),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], kernel_size=3),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*3, ch*3, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 96, 12]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=2) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=2),
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
def qarv2_tiny(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*3, ch*3]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], kernel_size=7),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], kernel_size=5),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], kernel_size=3),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*3, ch*3, ch*3, ch*2, ch*1]
    z_dims = [32, 64, 128, 16]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=2) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=2),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        res_block(dec_dims[1], kernel_size=3, mlp_ratio=3),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], kernel_size=3, mlp_ratio=3) for _ in range(1)],
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
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(4)],
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
def qarv2_t2(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*3, ch*3]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], kernel_size=7),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], kernel_size=5),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], kernel_size=3),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*3, ch*3, ch*3, ch*2, ch*1]
    z_dims = [32, 32, 128, 32]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=2) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=2),
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
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(1)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75),
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(4)],
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
def qarv2_z6sim(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*3, ch*3]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0]) for _ in range(4)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1]) for _ in range(4)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2]) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3]) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4]) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*3, ch*3, ch*3, ch*2, ch*1]
    z_dims = [32, 64, 128, 16]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1]) for _ in range(1)],
        res_block(dec_dims[0]),
        common.patch_upsample(dec_dims[0], dec_dims[1]),
        # 2x2
        res_block(dec_dims[1]),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2]) for _ in range(1)],
        res_block(dec_dims[1]),
        common.patch_upsample(dec_dims[1], dec_dims[2]),
        # 4x4
        res_block(dec_dims[2]),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3]) for _ in range(2)],
        res_block(dec_dims[2]),
        common.patch_upsample(dec_dims[2], dec_dims[3]),
        # 8x8
        res_block(dec_dims[3]),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4]) for _ in range(2)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3]),
        common.patch_upsample(dec_dims[3], dec_dims[4]),
        # 16x16
        *[res_block(dec_dims[4]) for _ in range(4)],
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
def qarv2_z6mlp4(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*2, ch*2]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], mlp_ratio=4) for _ in range(3)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], mlp_ratio=4) for _ in range(3)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], mlp_ratio=4),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], mlp_ratio=4),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], mlp_ratio=4),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [32, 64, 128, 16]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1]),
        # 2x2
        res_block(dec_dims[1], mlp_ratio=4),
        *[qarv.VRLVBlockBase(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[1], mlp_ratio=4),
        common.patch_upsample(dec_dims[1], dec_dims[2]),
        # 4x4
        res_block(dec_dims[2], mlp_ratio=4),
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], mlp_ratio=4) for _ in range(2)],
        res_block(dec_dims[2], mlp_ratio=4),
        common.patch_upsample(dec_dims[2], dec_dims[3]),
        # 8x8
        res_block(dec_dims[3], mlp_ratio=4),
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], mlp_ratio=4) for _ in range(2)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], mlp_ratio=4),
        common.patch_upsample(dec_dims[3], dec_dims[4]),
        # 16x16
        *[res_block(dec_dims[4], mlp_ratio=4) for _ in range(3)],
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
def qv2_t6(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*2, ch*2]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], mlp_ratio=4) for _ in range(3)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], mlp_ratio=4) for _ in range(3)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], mlp_ratio=4),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], mlp_ratio=4),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], mlp_ratio=4),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [32, 64, 128, 32]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockSmall(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1]),
        # 2x2
        *[qarv.VRLVBlockSmall(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[1], mlp_ratio=4),
        common.patch_upsample(dec_dims[1], dec_dims[2]),
        # 4x4
        *[qarv.VRLVBlockSmall(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], mlp_ratio=4) for _ in range(2)],
        res_block(dec_dims[2], mlp_ratio=4),
        common.patch_upsample(dec_dims[2], dec_dims[3]),
        # 8x8
        *[qarv.VRLVBlockSmall(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], mlp_ratio=4) for _ in range(2)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], mlp_ratio=4),
        common.patch_upsample(dec_dims[3], dec_dims[4]),
        # 16x16
        *[res_block(dec_dims[4], mlp_ratio=4) for _ in range(3)],
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
def qv2_t6z(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*2, ch*2]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], mlp_ratio=4) for _ in range(3)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], mlp_ratio=4) for _ in range(3)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], mlp_ratio=4),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], mlp_ratio=4),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], mlp_ratio=4),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [48, 96, 192, 48]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockSmall(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1]),
        # 2x2
        *[qarv.VRLVBlockSmall(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[1], mlp_ratio=4),
        common.patch_upsample(dec_dims[1], dec_dims[2]),
        # 4x4
        *[qarv.VRLVBlockSmall(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], mlp_ratio=4) for _ in range(2)],
        res_block(dec_dims[2], mlp_ratio=4),
        common.patch_upsample(dec_dims[2], dec_dims[3]),
        # 8x8
        *[qarv.VRLVBlockSmall(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], mlp_ratio=4) for _ in range(2)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], mlp_ratio=4),
        common.patch_upsample(dec_dims[3], dec_dims[4]),
        # 16x16
        *[res_block(dec_dims[4], mlp_ratio=4) for _ in range(3)],
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
def qv2_t6z1221(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*2, ch*2]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], mlp_ratio=4) for _ in range(3)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], mlp_ratio=4) for _ in range(3)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], mlp_ratio=4),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], mlp_ratio=4),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], mlp_ratio=4),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [64, 64, 192, 64]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockSmall(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1]),
        # 2x2
        *[qarv.VRLVBlockSmall(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], mlp_ratio=4) for _ in range(2)],
        res_block(dec_dims[1], mlp_ratio=4),
        common.patch_upsample(dec_dims[1], dec_dims[2]),
        # 4x4
        *[qarv.VRLVBlockSmall(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], mlp_ratio=4) for _ in range(2)],
        res_block(dec_dims[2], mlp_ratio=4),
        common.patch_upsample(dec_dims[2], dec_dims[3]),
        # 8x8
        *[qarv.VRLVBlockSmall(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], mlp_ratio=4) for _ in range(1)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], mlp_ratio=4),
        common.patch_upsample(dec_dims[3], dec_dims[4]),
        # 16x16
        *[res_block(dec_dims[4], mlp_ratio=4) for _ in range(3)],
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
def qv2t6_z32(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]
    ch = 128
    enc_dims = [ch, ch*2, ch*3, ch*2, ch*2]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], mlp_ratio=4) for _ in range(3)],
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], mlp_ratio=4) for _ in range(3)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1], mlp_ratio=4),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2], mlp_ratio=4),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s32'),
        res_block(enc_dims[3], mlp_ratio=4),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], mlp_ratio=4) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [64, 64, 128, 32]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockSmall(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1]),
        # 2x2
        *[qarv.VRLVBlockSmall(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-2], mlp_ratio=4) for _ in range(2)],
        res_block(dec_dims[1], mlp_ratio=4),
        common.patch_upsample(dec_dims[1], dec_dims[2]),
        # 4x4
        *[qarv.VRLVBlockSmall(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3], mlp_ratio=4) for _ in range(2)],
        res_block(dec_dims[2], mlp_ratio=4),
        common.patch_upsample(dec_dims[2], dec_dims[3]),
        # 8x8
        *[qarv.VRLVBlockSmall(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4], mlp_ratio=4) for _ in range(1)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        res_block(dec_dims[3], mlp_ratio=4),
        common.patch_upsample(dec_dims[3], dec_dims[4]),
        # 16x16
        *[res_block(dec_dims[4], mlp_ratio=4) for _ in range(3)],
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


def main():
    model = qv2_t6()
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'Number of parameters: {num_params} ({num_params/1e6:.2f}M)')

if __name__ == '__main__':
    main()
