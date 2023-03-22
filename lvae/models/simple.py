from collections import OrderedDict
import math
import torch.nn as nn
import torch.nn.functional as tnf

from compressai.models.utils import conv, deconv
from compressai.layers import GDN


class FactorizedPrior(nn.Module):
    def __init__(self, N=128, M=64):
        super().__init__()

        self.g_a = nn.Sequential(
            conv(3, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, 3),
        )

    def forward(self, x):
        y = self.g_a(x)
        x_hat = self.g_s(y)

        loss = tnf.mse_loss(x_hat, x, reduction='mean')

        stats = OrderedDict()
        stats['loss'] = loss

        # ================ Logging ================
        stats['bppix'] = math.prod(y.shape[1:4]) * 32 / math.prod(x.shape[2:4])
        psnr = -10 * math.log10(loss.item())
        stats['psnr'] = psnr

        return stats
