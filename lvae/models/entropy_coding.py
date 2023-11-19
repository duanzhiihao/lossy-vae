import math
import scipy.stats
import torch
import torch.distributions as td

from compressai.ops import LowerBound
from compressai.entropy_models import GaussianConditional


def _to_float32(*args):
    if len(args) == 1:
        return args[0].to(dtype=torch.float)
    else:
        return tuple([t.to(dtype=torch.float) for t in args])


# @torch.autocast('cuda', enabled=False) # disable mixed precision
def _safe_log_prob_mass(distribution, x, bin_size, prob_clamp):
    prob_mass = distribution.cdf(x + 0.5*bin_size) - distribution.cdf(x - 0.5*bin_size)
    log_prob = torch.where(
        prob_mass > prob_clamp,
        torch.log(prob_mass.clamp(min=1e-8)),
        distribution.log_prob(x) + math.log(bin_size)
    )
    return log_prob


def _sanity_check_scale_table(scale_table):
    assert isinstance(scale_table, torch.Tensor)
    assert (scale_table.dim() == 1) and (scale_table.shape[0] >= 1) and (scale_table.min() > 0)
    assert torch.equal(scale_table, torch.sort(scale_table)[0])


def gaussian_log_prob_mass(mean, scale, x, bin_size=1.0, prob_clamp=1e-6):
    """ Compute log(P) of a "quantized" Normal(`mean`, `scale`) distribution evaluated at `x`,
    where P = cdf(`x` + 0.5*bin_size) - cdf(`x` - 0.5*bin_size).

    Args:
        mean        (Tensor): mean of the Gaussian
        scale       (Tensor): scale (standard deviation) of the Gaussian
        x           (Tensor): the quantized Gaussian is evaluated at `x`
        bin_size    (float):  quantization bin size
        prob_clamp  (float):  when prob < prob_clamp, use approximation \
            to improve numerical stability.
    """
    mean, scale, x = _to_float32(mean, scale, x)
    assert scale.min() > 0, f'invalid scale value = {scale.min()}'
    log_prob = _safe_log_prob_mass(td.Normal(mean, scale), x, bin_size, prob_clamp)
    return log_prob


class DiscretizedGaussian(GaussianConditional):
    """ Custom discretized gaussian.
    """
    def __init__(self, scale_table=None):
        """
        Args:
            scale_table (torch.Tensor, optional): a 1-D tensor.
        """
        super(GaussianConditional, self).__init__()

        if scale_table is None:
            scale_table = self._get_default_scale_table()
        _sanity_check_scale_table(scale_table)
        self.register_buffer("scale_table", scale_table, persistent=False)

        self.tail_mass = float(1e-9)
        self.lower_bound_scale = LowerBound(scale_table[0])

        self.standard_gaussian = td.Normal(loc=0, scale=1)

    @staticmethod
    def _get_default_scale_table():
        scale_table = torch.exp(torch.linspace(math.log(0.11), math.log(20.0), steps=64))
        return scale_table

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def _standardized_cumulative(self, inputs: torch.Tensor):
        return self.standard_gaussian.cdf(inputs)


def laplace_log_prob_mass(mean, scale, x, bin_size=1.0, prob_clamp=1e-6):
    mean, scale, x = _to_float32(mean, scale, x)
    assert scale.min() > 0, f'invalid scale value = {scale.min()}'
    log_prob = _safe_log_prob_mass(td.Laplace(mean, scale), x, bin_size, prob_clamp)
    return log_prob


class DiscretizedLaplace(GaussianConditional):
    def __init__(self, scale_table=None):
        """
        Args:
            scale_table (torch.Tensor, optional): a 1-D tensor.
        """
        super(GaussianConditional, self).__init__()

        if scale_table is None:
            scale_table = self._get_default_scale_table()
        # sanity check
        _sanity_check_scale_table(scale_table)
        self.register_buffer("scale_table", scale_table, persistent=False)

        self.tail_mass = float(1e-9)
        self.lower_bound_scale = LowerBound(scale_table[0])

        self.standard_laplace = td.Laplace(loc=0, scale=1)

    @staticmethod
    def _get_default_scale_table():
        scale_table = torch.exp(torch.linspace(math.log(0.01), math.log(20.0), steps=64))
        return scale_table

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.laplace.ppf(quantile)

    def _standardized_cumulative(self, inputs: torch.Tensor):
        return self.standard_laplace.cdf(inputs)
