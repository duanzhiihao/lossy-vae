import torch
from torch.hub import load_state_dict_from_url

from lvae.models.registry import register_model
import lvae.models.common as common
import lvae.models.qarv.model as qarv
import lvae.models.qarv_cls.model as qarv_cls


@register_model
def qarv_cls_test(lmb_range=(16,2048), pretrained=False):
    cfg = dict()

    # variable rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    _emb_dim = cfg['lmb_embed_dim'][1]

    ch = 128
    enc_dims = [ch*1, ch*2, ch*4, ch*6, ch*6]
    dec_dims = [ch*6, ch*6, ch*4, ch*2, ch*1]
    z_dims = [32, 32, 96, 8]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[qarv.ConvNeXtBlockAdaLN(enc_dims[0], _emb_dim) for _ in range(6)],
        qarv.ConvNeXtAdaLNPatchDown(enc_dims[0], enc_dims[1], embed_dim=_emb_dim),
        # 8x8
        *[qarv.ConvNeXtBlockAdaLN(enc_dims[1], _emb_dim) for _ in range(6)],
        qarv.ConvNeXtAdaLNPatchDown(enc_dims[1], enc_dims[2], embed_dim=_emb_dim),
        # 4x4
        *[qarv.ConvNeXtBlockAdaLN(enc_dims[2], _emb_dim) for _ in range(6)],
        qarv.ConvNeXtAdaLNPatchDown(enc_dims[2], enc_dims[3], embed_dim=_emb_dim),
        # 2x2
        *[qarv.ConvNeXtBlockAdaLN(enc_dims[3], _emb_dim) for _ in range(6)],
        qarv.ConvNeXtAdaLNPatchDown(enc_dims[3], enc_dims[4], embed_dim=_emb_dim),
        # 1x1
        *[qarv.ConvNeXtBlockAdaLN(enc_dims[4], _emb_dim) for _ in range(6)],
    ]

    cfg['dec_blocks'] = [
        # 1x1
        *[qarv.VRLatentBlockBase(dec_dims[0], z_dims[0], _emb_dim, enc_width=enc_dims[-1]) for _ in range(1)],
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[qarv.VRLatentBlockBase(dec_dims[1], z_dims[1], _emb_dim, enc_width=enc_dims[-2]) for _ in range(2)],
        qarv_cls.ClassificationCutPoint(),
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[qarv.VRLatentBlockBase(dec_dims[2], z_dims[2], _emb_dim, enc_width=enc_dims[-3]) for _ in range(3)],
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[qarv.VRLatentBlockBase(dec_dims[3], z_dims[3], _emb_dim, enc_width=enc_dims[-4]) for _ in range(3)],
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[qarv.ConvNeXtBlockAdaLN(dec_dims[4], _emb_dim) for _ in range(4)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    cfg['classification_channel'] = dec_dims[1]

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    cfg['max_stride'] = 64

    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']
    cfg['cls_only'] = False

    model = qarv_cls.ClassificationLossyVAE(cfg)
    if pretrained:
        raise NotImplementedError()
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_z96-p3.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    return model
