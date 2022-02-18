import ot
import numpy as np
import torch


def emd(p, q, pE, qE):
    M = ot.dist(p, q, metric="euclidean")
    return ot.emd2(pE, qE, M)


def cos_sine(theta):
    if type(theta) == torch.Tensor:
        return torch.concat([torch.cos(theta), torch.sin(theta)], dim=1)
    elif type(theta) == np.ndarray:
        return np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
