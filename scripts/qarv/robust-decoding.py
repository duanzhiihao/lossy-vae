from pathlib import Path
from PIL import Image
import math
import torch
import torchvision as tv
import torchvision.transforms.functional as tvf

from lvae.models.qarv.zoo import qarv_base


def get_latents(masked_stats):
    return [None if (st is None) else st['z'] for st in masked_stats]


def get_bpp(masked_stats, img_hw):
    kl_divs = [st['kl'] for st in masked_stats if (st is not None)]
    imH, imW = img_hw
    bpp = sum([kl.sum(dim=(1,2,3)) for kl in kl_divs]) / (imH * imW) * math.log2(math.e)
    return bpp


@torch.no_grad()
def main():
    device = torch.device('cuda:0')

    # initialize model
    model = qarv_base(pretrained=True)

    model = model.to(device=device)
    model.eval()

    impath = Path('images/zebra256.png')
    # impath = Path('images/house256.png')
    im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=device)
    nB, imC, imH, imW = im.shape

    lmb = 16
    _, stats_all = model.forward_end2end(im, lmb=lmb, get_latent=True)

    progressive_decodings = []
    bpps = []
    L = len(stats_all)
    for anchor in range(L):
        masked_stats = [stat if (i <= anchor) else None for (i,stat) in enumerate(stats_all)]
        name = 'progressive'
        # masked_stats = [None if (i == anchor) else stat for (i,stat) in enumerate(stats_all)]
        # name = 'exclude'
        # masked_stats = [None if (i < anchor) else stat for (i,stat) in enumerate(stats_all)]
        # name = 'reverse'
        # masked_stats = [stat if (i == anchor) else None for (i,stat) in enumerate(stats_all)]
        # name = 'single'
        # conditional sampling
        latents = get_latents(masked_stats)
        _bhw = (nB, imH//64, imW//64)
        sample = model.conditional_sample(lmb=lmb, latents=latents, bhw_repeat=_bhw, t=0)
        progressive_decodings.append(sample.squeeze(0))
        # compute bpp
        bpp = get_bpp(masked_stats, img_hw=(imH, imW))
        bpps.append(bpp.item())
        print(f'{name}={anchor}, bpp={bpp.item()}')
    progressive_decodings = torch.stack(progressive_decodings, dim=0)
    pad_size = 4
    im = tv.utils.make_grid(progressive_decodings, nrow=12, padding=pad_size, pad_value=1)
    img = tvf.to_pil_image(im)

    # plot and save
    print(', '.join([f'{b:.3f} bpp' for b in bpps]))
    im = im[:, pad_size:-pad_size-1, pad_size:-pad_size-1]
    img = tvf.to_pil_image(im)
    fpath = f'runs/qarv-{name}-lmb{lmb}-{impath.stem}.png'
    print(fpath)
    img.save(fpath)
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
