import torch
from torch.hub import load_state_dict_from_url

from lvae.models.registry import register_model
import lvae.models.common as common
import lvae.models.rd.library as lib


@register_model
def rd_model_base(lmb_range=(4,2048), pretrained=False):
    cfg = dict()

    # variable rate
    cfg['lmb_range'] = (float(lmb_range[0]), float(lmb_range[1]))
    cfg['lmb_embed_dim'] = (256, 256)
    cfg['sin_period'] = 64
    _emb_dim = cfg['lmb_embed_dim'][1]

    enc_dims = [256, 512, 640, 768, 768]
    dec_dims = [768, 768, 640, 512, 256]
    z_dims = [32, 32, 32, 32, 32]

    im_channels = 3
    cfg['enc_blocks'] = [
        # 64x64
        common.patch_downsample(im_channels, enc_dims[0], rate=4),
        # 16x16
        *[lib.ConvNeXtBlockAdaLN(enc_dims[0], _emb_dim) for _ in range(6)],
        lib.ConvNeXtAdaLNPatchDown(enc_dims[0], enc_dims[1], embed_dim=_emb_dim),
        # 8x8
        *[lib.ConvNeXtBlockAdaLN(enc_dims[1], _emb_dim) for _ in range(6)],
        lib.ConvNeXtAdaLNPatchDown(enc_dims[1], enc_dims[2], embed_dim=_emb_dim),
        # 4x4
        *[lib.ConvNeXtBlockAdaLN(enc_dims[2], _emb_dim) for _ in range(6)],
        lib.ConvNeXtAdaLNPatchDown(enc_dims[2], enc_dims[3], embed_dim=_emb_dim),
        # 2x2
        *[lib.ConvNeXtBlockAdaLN(enc_dims[3], _emb_dim) for _ in range(4)],
        lib.ConvNeXtAdaLNPatchDown(enc_dims[3], enc_dims[3], embed_dim=_emb_dim),
        # 1x1
        *[lib.ConvNeXtBlockAdaLN(enc_dims[3], _emb_dim) for _ in range(4)],
    ]

    cfg['dec_blocks'] = [
        # 1x1
        *[lib.VRLatentBlock3PosV2(dec_dims[0], z_dims[0], _emb_dim, enc_width=enc_dims[-1]) for _ in range(1)],
        common.patch_upsample(dec_dims[0], dec_dims[1], rate=2),
        # 2x2
        *[lib.VRLatentBlock3PosV2(dec_dims[1], z_dims[1], _emb_dim, enc_width=enc_dims[-2]) for _ in range(2)],
        common.patch_upsample(dec_dims[1], dec_dims[2], rate=2),
        # 4x4
        *[lib.VRLatentBlock3PosV2(dec_dims[2], z_dims[2], _emb_dim, enc_width=enc_dims[-3]) for _ in range(3)],
        common.patch_upsample(dec_dims[2], dec_dims[3], rate=2),
        # 8x8
        *[lib.VRLatentBlock3PosV2(dec_dims[3], z_dims[3], _emb_dim, enc_width=enc_dims[-4]) for _ in range(4)],
        common.patch_upsample(dec_dims[3], dec_dims[4], rate=2),
        # 16x16
        *[lib.VRLatentBlock3PosV2(dec_dims[4], z_dims[4], _emb_dim, enc_width=enc_dims[-5]) for _ in range(5)],
        common.patch_upsample(dec_dims[4], im_channels, rate=4)
    ]

    # mean and std computed on imagenet
    cfg['im_shift'] = -0.4546259594901961
    cfg['im_scale'] = 3.67572653978347
    cfg['max_stride'] = 64

    cfg['log_images'] = ['collie64.png', 'gun128.png', 'motor256.png']

    model = lib.VariableRateLossyVAE(cfg)
    if isinstance(pretrained, str):
        msd = torch.load(pretrained)['model']
        model.load_state_dict(msd)
    elif pretrained:
        raise NotImplementedError()
        url = 'https://huggingface.co/duanzh0/my-model-weights/resolve/main/qarv_base-dec12-2022.pt'
        msd = load_state_dict_from_url(url)['model']
        model.load_state_dict(msd)
    return model


@torch.no_grad()
def main():
    model = rd_model_base()
    num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'Number of parameters = {num/1e6:.2f} M')
    debug = 1

if __name__ == '__main__':
    main()
