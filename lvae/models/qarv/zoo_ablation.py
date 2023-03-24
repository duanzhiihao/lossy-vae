import torch
from torch.hub import load_state_dict_from_url

from lvae.models.registry import register_model
import lvae.models.common as common
import lvae.models.qarv.model as qarv


@register_model
def qarv_2z(lmb_range=(16,2048), pretrained=False):
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
    enc_dims = [ch*1, ch*2, ch*3, ch*2, ch*2]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        # common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        # common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [256, 384]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[res_block(dec_dims[1], kernel_size=3, mlp_ratio=3) for _ in range(4)],
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[1], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(1)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75) for _ in range(6)],
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(6)],
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
    enc_dims = [ch*1, ch*2, ch*3, ch*2, ch*2]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        # common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [256, 384, 256]
    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLVBlockBase(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[res_block(dec_dims[1], kernel_size=3, mlp_ratio=3) for _ in range(4)],
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        *[qarv.VRLVBlockBase(dec_dims[2], z_dims[1], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(1)],
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        *[qarv.VRLVBlockBase(dec_dims[3], z_dims[2], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(1)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        *[res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(6)],
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


class HyperPriorBidInfLatentBlock(qarv.VRLVBlockBase):
    def fuse_feature_and_z(self, feature, z):
        # add the new information carried by z to the feature
        feature = self.z_proj(z)
        return feature


@register_model
def mshyp_2z(lmb_range=(16,2048), pretrained=False):
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
    enc_dims = [ch*1, ch*2, ch*3, ch*2, ch*2]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        # common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        # common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [256, 384]
    cfg['dec_blocks'] = [
        # 1x1
        *[HyperPriorBidInfLatentBlock(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[res_block(dec_dims[1], kernel_size=3, mlp_ratio=3) for _ in range(4)],
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        *[HyperPriorBidInfLatentBlock(dec_dims[2], z_dims[1], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(1)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75) for _ in range(6)],
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(6)],
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


class HyperPriorLatentBlock(qarv.VRLVBlockBase):
    def __init__(self, width, zdim, enc_key, enc_width, embed_dim=None, kernel_size=7, mlp_ratio=2):
        super(qarv.VRLVBlockBase, self).__init__()
        self.in_channels  = width
        self.out_channels = width
        self.enc_key = enc_key

        block = common.ConvNeXtBlockAdaLN
        embed_dim = embed_dim or self.default_embedding_dim
        self.resnet_front = block(width,   embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.resnet_end   = block(width,   embed_dim, kernel_size=kernel_size, mlp_ratio=mlp_ratio)
        self.posterior0 = block(enc_width, embed_dim, kernel_size=kernel_size)
        # self.posterior1 = block(width,     embed_dim, kernel_size=kernel_size)
        self.posterior2 = block(width,     embed_dim, kernel_size=kernel_size)
        self.post_merge = common.conv_k1s1(enc_width, width)
        self.posterior  = common.conv_k3s1(width, zdim)
        self.z_proj     = common.conv_k1s1(zdim, width)
        self.prior      = common.conv_k1s1(width, zdim*2)

        import lvae.models.entropy_coding as entropy_coding
        self.discrete_gaussian = entropy_coding.DiscretizedGaussian()
        self.is_latent_block = True

    def transform_posterior(self, feature, enc_feature, lmb_embedding):
        """ posterior q(z_i | z_<i, x)

        Args:
            feature     (torch.Tensor): feature map
            enc_feature (torch.Tensor): feature map
        """
        enc_feature = self.posterior0(enc_feature, lmb_embedding)
        # feature = self.posterior1(feature, lmb_embedding)
        # merged = torch.cat([feature, enc_feature], dim=1)
        merged = self.post_merge(enc_feature)
        merged = self.posterior2(merged, lmb_embedding)
        qm = self.posterior(merged)
        return qm

    def fuse_feature_and_z(self, feature, z):
        # add the new information carried by z to the feature
        feature = self.z_proj(z)
        return feature


@register_model
def msh_2z(lmb_range=(16,2048), pretrained=False):
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
    enc_dims = [ch*1, ch*2, ch*3, ch*2, ch*2]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        # common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        # common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [256, 384]
    cfg['dec_blocks'] = [
        # 1x1
        *[HyperPriorLatentBlock(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[res_block(dec_dims[1], kernel_size=3, mlp_ratio=3) for _ in range(4)],
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        *[HyperPriorLatentBlock(dec_dims[2], z_dims[1], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(1)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75) for _ in range(6)],
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(6)],
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
def msh_3z(lmb_range=(16,2048), pretrained=False):
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
    enc_dims = [ch*1, ch*2, ch*3, ch*2, ch*2]

    res_block = common.ConvNeXtBlockAdaLN
    res_block.default_embedding_dim = cfg['lmb_embed_dim'][1]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[res_block(enc_dims[0], kernel_size=7) for _ in range(4)],
        res_block(enc_dims[0]),
        common.patch_downsample(enc_dims[0], enc_dims[1]),
        # 8x8
        *[res_block(enc_dims[1], kernel_size=7) for _ in range(4)],
        common.SetKey('enc_s8'),
        res_block(enc_dims[1]),
        common.patch_downsample(enc_dims[1], enc_dims[2]),
        # 4x4
        *[res_block(enc_dims[2], kernel_size=5) for _ in range(4)],
        common.SetKey('enc_s16'),
        res_block(enc_dims[2]),
        common.patch_downsample(enc_dims[2], enc_dims[3]),
        # 2x2
        *[res_block(enc_dims[3], kernel_size=3) for _ in range(4)],
        # common.SetKey('enc_s32'),
        res_block(enc_dims[3]),
        common.patch_downsample(enc_dims[3], enc_dims[4]),
        # 1x1
        *[res_block(enc_dims[4], kernel_size=1) for _ in range(4)],
        common.SetKey('enc_s64'),
    ]

    dec_dims = [ch*2, ch*2, ch*3, ch*2, ch*1]
    z_dims = [256, 384, 256]
    cfg['dec_blocks'] = [
        # 1x1
        *[HyperPriorLatentBlock(dec_dims[0], z_dims[0], enc_key='enc_s64', enc_width=enc_dims[-1], kernel_size=1, mlp_ratio=4) for _ in range(1)],
        res_block(dec_dims[0], kernel_size=1, mlp_ratio=4),
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[res_block(dec_dims[1], kernel_size=3, mlp_ratio=3) for _ in range(4)],
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        *[HyperPriorLatentBlock(dec_dims[2], z_dims[1], enc_key='enc_s16', enc_width=enc_dims[-3], kernel_size=5, mlp_ratio=2) for _ in range(1)],
        *[res_block(dec_dims[2], kernel_size=5, mlp_ratio=2) for _ in range(2)],
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        *[HyperPriorLatentBlock(dec_dims[3], z_dims[2], enc_key='enc_s8', enc_width=enc_dims[-4], kernel_size=7, mlp_ratio=1.75) for _ in range(1)],
        common.CompresionStopFlag(), # no need to execute remaining blocks when compressing
        *[res_block(dec_dims[3], kernel_size=7, mlp_ratio=1.75) for _ in range(3)],
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[res_block(dec_dims[4], kernel_size=7, mlp_ratio=1.5) for _ in range(6)],
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


@torch.no_grad()
def main():
    # model = qarv_2z()
    model = mshyp_2z()
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'Number of parameters: {num_params/1e6:.3f} M')
    debug = 1

if __name__ == '__main__':
    main()
