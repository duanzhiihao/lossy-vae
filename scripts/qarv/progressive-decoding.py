import json
import pickle
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from PIL import Image
import math
import torch
import torchvision as tv
import torchvision.transforms.functional as tvf

from lvae.models.qarv.zoo import qarv_base


@torch.no_grad()
def main():
    device = torch.device('cuda:0')

    # initialize model
    model = qarv_base(pretrained=True)

    model = model.to(device=device)
    model.eval()

    impath = Path('images/house256.png')
    im = tvf.to_tensor(Image.open(impath)).unsqueeze_(0).to(device=device)
    nB, imC, imH, imW = im.shape

    lmb = 16
    _, stats_all = model.forward_end2end(im, lmb=lmb, get_latents=True)

    progressive_decodings = []
    bpps = []
    L = len(stats_all)
    for keep in range(1, L+1):
        if keep == 0:
            latents = None
            kl = torch.zeros(1)
            sample = model.conditional_sample(lmb=lmb, latents=latents, t=0, bhw_repeat=(1,4,4))
        else:
            latents = [stat['z'] if (i < keep) else None for (i,stat) in enumerate(stats_all)]
            kl_divs = [stat['kl'] for (i,stat) in enumerate(stats_all) if (i < keep)]
            bpp = sum([kl.sum(dim=(1,2,3)) for kl in kl_divs]) / (imH * imW) * math.log2(math.e)
            sample = model.conditional_sample(lmb=lmb, latents=latents, t=0)
        progressive_decodings.append(sample.squeeze(0))
        bpps.append(bpp.item())
        print(f'Keep={keep}, bpp={bpp.item()}')
    progressive_decodings = torch.stack(progressive_decodings, dim=0)
    pad_size = 4
    im = tv.utils.make_grid(progressive_decodings, nrow=12, padding=pad_size, pad_value=1)
    img = tvf.to_pil_image(im)

    # plot and save
    print(', '.join([f'{round(b,3)} bpp' for b in bpps]))
    im = im[:, pad_size:-pad_size-1, pad_size:-pad_size-1]
    img = tvf.to_pil_image(im)
    img.save(f'runs/qarv-progressive-{impath.stem}.png')
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()
