import torch
from torch.hub import load_state_dict_from_url

from lvae.models.registry import register_model
import lvae.models.common as cm
import lvae.models.qarv.model_v2 as qarv


@register_model
def qv2_3z(lmb_range=(64,8192), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    ch = 128
    enc_dims = [ch*1, ch*2, ch*3, ch*2, ch*2]

    res_block = cm.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        cm.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0]) for _ in range(4)],
        res_block(enc_dims[0]),
        cm.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1]) for _ in range(4)],
        cm.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        cm.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2]) for _ in range(4)],
        cm.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        cm.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=5) for _ in range(4)],
        # cm.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        cm.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=3) for _ in range(4)],
        cm.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [256, 384, 256]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockV2(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=3) for _ in range(1)],
        *[res_block(dec_dims[0], kernel_size=3, mlp_ratio=4) for _ in range(2)],
        cm.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[res_block(dec_dims[1], kernel_size=5, mlp_ratio=3) for _ in range(4)],
        cm.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[res_block(dec_dims[2], mlp_ratio=2) for _ in range(3)],
        *[qarv.VRLVBlockV2(dec_dims[2], z_dims[1], enc_key='enc_s16', enc_width=enc_dims[-3]) for _ in range(1)],
        *[res_block(dec_dims[2], mlp_ratio=2) for _ in range(3)],
        cm.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3], mlp_ratio=1.75) for _ in range(4)],
        *[qarv.VRLVBlockV2(dec_dims[3], z_dims[2], enc_key='enc_s8', enc_width=enc_dims[-4]) for _ in range(1)],
        cm.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        *[res_block(dec_dims[3], mlp_ratio=1.75) for _ in range(4)],
        cm.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], mlp_ratio=1.5) for _ in range(6)],
        cm.patch_upsample(dec_dims[4], im_channels, rate=4)
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
def qv2_3z_no_enc_res(lmb_range=(64,8192), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    ch = 128
    enc_dims = [ch*1, ch*2, ch*3, ch*2, ch*2]

    res_block = cm.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        cm.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0]) for _ in range(4)],
        cm.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1]) for _ in range(4)],
        cm.SetKey('enc_s8'),
        cm.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2]) for _ in range(4)],
        cm.SetKey('enc_s16'),
        cm.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=5) for _ in range(4)],
        # cm.SetKey('enc_s32'),
        cm.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=3) for _ in range(4)],
        cm.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [256, 384, 256]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockV2(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=3) for _ in range(1)],
        *[res_block(dec_dims[0], kernel_size=3, mlp_ratio=4) for _ in range(2)],
        cm.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[res_block(dec_dims[1], kernel_size=5, mlp_ratio=3) for _ in range(4)],
        cm.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[res_block(dec_dims[2], mlp_ratio=2) for _ in range(3)],
        *[qarv.VRLVBlockV2(dec_dims[2], z_dims[1], enc_key='enc_s16', enc_width=enc_dims[-3]) for _ in range(1)],
        *[res_block(dec_dims[2], mlp_ratio=2) for _ in range(3)],
        cm.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3], mlp_ratio=1.75) for _ in range(4)],
        *[qarv.VRLVBlockV2(dec_dims[3], z_dims[2], enc_key='enc_s8', enc_width=enc_dims[-4]) for _ in range(1)],
        cm.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        *[res_block(dec_dims[3], mlp_ratio=1.75) for _ in range(4)],
        cm.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], mlp_ratio=1.5) for _ in range(6)],
        cm.patch_upsample(dec_dims[4], im_channels, rate=4)
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
def qv2_4z(lmb_range=(64,8192), pretrained=False):
    cfg = dict()

    # maximum downsampling factor
    cfg['max_stride'] = 64
    # images used during training for logging
    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    # variable-rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64

    # model configuration
    ch = 128
    enc_dims = [ch*1, ch*2, ch*3, ch*2, ch*2]

    res_block = cm.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        cm.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0]) for _ in range(4)],
        cm.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1]) for _ in range(4)],
        cm.SetKey('enc_s8'),
        cm.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2]) for _ in range(4)],
        cm.SetKey('enc_s16'),
        cm.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=5) for _ in range(4)],
        cm.SetKey('enc_s32'),
        cm.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=3) for _ in range(4)],
        cm.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [256, 256, 384, 256]
    cfg['dec_blocks'] = [
        # 1x1
        qarv.VRLVBlockV2(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=3),
        *[res_block(dec_dims[0], kernel_size=3) for _ in range(2)],
        cm.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[res_block(dec_dims[1], kernel_size=5) for _ in range(2)],
        qarv.VRLVBlockV2(dec_dims[1], z_dims[1], enc_key='enc_s32', enc_width=enc_dims[-3], kernel_size=5),
        *[res_block(dec_dims[1], kernel_size=5) for _ in range(2)],
        cm.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[res_block(dec_dims[2]) for _ in range(3)],
        qarv.VRLVBlockV2(dec_dims[2], z_dims[2], enc_key='enc_s16', enc_width=enc_dims[-3]),
        *[res_block(dec_dims[2]) for _ in range(3)],
        cm.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3]) for _ in range(4)],
        qarv.VRLVBlockV2(dec_dims[3], z_dims[3], enc_key='enc_s8', enc_width=enc_dims[-4]),
        cm.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        *[res_block(dec_dims[3]) for _ in range(4)],
        cm.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4]) for _ in range(6)],
        cm.patch_upsample(dec_dims[4], im_channels, rate=4)
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
