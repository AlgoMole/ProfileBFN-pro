import torch
from torch.functional import F



def sampling_tensor_discreteBayesianFlow_mbcltbf(
    time, tokens, beta1, dict_size, torder
):
    """
    Args:
        time: [..., T]
        tokens: [..., T, K], simplex already
        beta1: [..., T]
        mask: [K], to identify valid amminoacids
    """
    mask = torch.tensor(
        [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        dtype=torch.bool,
        device=beta1.device,
    )

    valid_k = 20
    x = tokens
    beta = beta1 * (time**torder)  # [T]
    beta = beta.unsqueeze(-1)  # (T, 1)
    _mean = beta * (
        valid_k * x - 1
    )  # (T, K) #TODO fix it by setting irrelevant values to -inf
    mean = torch.where(mask, _mean, -float("inf"))
    std = (beta * valid_k).sqrt()  # (T, 1)
    eps = torch.randn_like(mean)  # (T, K)
    y = mean + std * eps  # (T, K)
    theta = F.softmax(y, dim=-1)  #  profile as prior   [  y + log(profile) ]
    return theta