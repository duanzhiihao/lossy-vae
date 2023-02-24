import lvae.models.qresvae.model as qres

from lvae.models.registry import register_model


@register_model
def qres34m(lmb=32, pretrained=False):
    cfg = dict()

    enc_nums = [6, 6, 6, 4, 2]
    dec_nums = [1, 2, 3, 3, 3]
    z_dims = [16, 14, 12, 10, 8]

    im_channels = 3
    ch = 96 # 128
    cfg['enc_blocks'] = [
        qres.get_conv(im_channels, ch*2, kernel_size=4, stride=4, padding=0),
        *[qres.MyConvNeXtBlock(ch*2, kernel_size=7) for _ in range(enc_nums[0])], # 16x16
        qres.MyConvNeXtPatchDown(ch*2, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=7) for _ in range(enc_nums[1])], # 8x8
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=5) for _ in range(enc_nums[2])], # 4x4
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=3) for _ in range(enc_nums[3])], # 2x2
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=1) for _ in range(enc_nums[4])], # 1x1
    ]
    cfg['dec_blocks'] = [
        *[qres.QLatentBlockX(ch*4, z_dims[0], kernel_size=1) for _ in range(dec_nums[0])], # 1x1
        qres.subpix_conv(ch*4, ch*4, up_rate=2),
        *[qres.QLatentBlockX(ch*4, z_dims[1], kernel_size=3) for _ in range(dec_nums[1])], # 2x2
        qres.subpix_conv(ch*4, ch*4, up_rate=2),
        *[qres.QLatentBlockX(ch*4, z_dims[2], kernel_size=5) for _ in range(dec_nums[2])], # 4x4
        qres.subpix_conv(ch*4, ch*4, up_rate=2),
        *[qres.QLatentBlockX(ch*4, z_dims[3], kernel_size=7) for _ in range(dec_nums[3])], # 8x8
        qres.subpix_conv(ch*4, ch*2, up_rate=2),
        *[qres.QLatentBlockX(ch*2, z_dims[4], kernel_size=7) for _ in range(dec_nums[4])], # 16x16
        qres.subpix_conv(ch*2, im_channels, up_rate=4)
    ]
    cfg['out_net'] = qres.MSEOutputNet(mse_lmb=lmb)

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    cfg['max_stride'] = 64

    model = qres.HierarchicalVAE(cfg)
    return model


@register_model
def qres34m_lossless():
    cfg = SimpleConfig()

    enc_nums = [6, 6, 6, 4, 2]
    dec_nums = [1, 2, 3, 3, 3]
    z_dims = [16, 14, 12, 10, 8]

    im_channels = 3
    ch = 96 # 128
    cfg['enc_blocks'] = [
        qres.get_conv(im_channels, ch*2, kernel_size=4, stride=4, padding=0),
        *[qres.MyConvNeXtBlock(ch*2, kernel_size=7) for _ in range(enc_nums[0])], # 16x16
        qres.MyConvNeXtPatchDown(ch*2, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=7) for _ in range(enc_nums[1])], # 8x8
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=5) for _ in range(enc_nums[2])], # 4x4
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=3) for _ in range(enc_nums[3])], # 2x2
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=1) for _ in range(enc_nums[4])], # 1x1
    ]
    cfg['dec_blocks'] = [
        *[qres.QLatentBlockX(ch*4, z_dims[0], kernel_size=1) for _ in range(dec_nums[0])], # 1x1
        qres.subpix_conv(ch*4, ch*4, up_rate=2),
        *[qres.QLatentBlockX(ch*4, z_dims[1], kernel_size=3) for _ in range(dec_nums[1])], # 2x2
        qres.subpix_conv(ch*4, ch*4, up_rate=2),
        *[qres.QLatentBlockX(ch*4, z_dims[2], kernel_size=5) for _ in range(dec_nums[2])], # 4x4
        qres.subpix_conv(ch*4, ch*4, up_rate=2),
        *[qres.QLatentBlockX(ch*4, z_dims[3], kernel_size=7) for _ in range(dec_nums[3])], # 8x8
        qres.subpix_conv(ch*4, ch*2, up_rate=2),
        *[qres.QLatentBlockX(ch*2, z_dims[4], kernel_size=7) for _ in range(dec_nums[4])], # 16x16
    ]
    cfg['out_net'] = qres.GaussianNLLOutputNet(
        conv_mean=qres.subpix_conv(ch*2, im_channels, up_rate=4),
        conv_scale=qres.subpix_conv(ch*2, im_channels, up_rate=4)
    )

    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    cfg['max_stride'] = 64

    model = qres.HierarchicalVAE(cfg)
    return model


@register_model
def qres17m(lmb=None):
    cfg = SimpleConfig()

    enc_nums = [6,6,4,2]
    dec_nums = [1,2,4,5]
    z_dims = [16, 8, 6, 4]

    im_channels = 3
    ch = 72 # 128
    cfg['enc_blocks'] = [
        qres.get_conv(im_channels, ch*2, kernel_size=4, stride=4, padding=0),
        *[qres.MyConvNeXtBlock(ch*2, kernel_size=7) for _ in range(enc_nums[0])], # 16x16
        qres.MyConvNeXtPatchDown(ch*2, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=5) for _ in range(enc_nums[1])], # 8x8
        qres.MyConvNeXtPatchDown(ch*4, ch*4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=3) for _ in range(enc_nums[2])], # 4x4
        qres.MyConvNeXtPatchDown(ch*4, ch*4, down_rate=4),
        *[qres.MyConvNeXtBlock(ch*4, kernel_size=1) for _ in range(enc_nums[3])], # 1x1
    ]
    from torch.nn import Upsample
    cfg['dec_blocks'] = [
        *[qres.QLatentBlockX(ch*4, z_dims[0], kernel_size=1) for _ in range(dec_nums[0])], # 1x1
        Upsample(scale_factor=4),
        *[qres.QLatentBlockX(ch*4, z_dims[1], kernel_size=3) for _ in range(dec_nums[1])], # 4x4
        qres.deconv(ch*4, ch*4, kernel_size=3),
        *[qres.QLatentBlockX(ch*4, z_dims[2], kernel_size=5) for _ in range(dec_nums[2])], # 8x8
        qres.deconv(ch*4, ch*2),
        *[qres.QLatentBlockX(ch*2, z_dims[3], kernel_size=7) for _ in range(dec_nums[3])], # 16x16
        qres.subpix_conv(ch*2, im_channels, up_rate=4)
    ]
    cfg['out_net'] = qres.MSEOutputNet(mse_lmb=lmb)

    # mean and std computed on CelebA
    cfg['im_shift'] = -0.4356
    cfg['im_scale'] = 3.397893306150187
    cfg['max_stride'] = 64

    model = qres.HierarchicalVAE(cfg)
    return model
