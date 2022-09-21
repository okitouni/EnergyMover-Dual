from cmath import isclose
from typing import Iterable
from torch.utils.data import Dataset
import ot
import numpy as np
import torch
from itertools import combinations


class EMDLoss(torch.nn.Module):
    def forward(self, p, q, pE=None, qE=None):
        fp = (p * pE).sum() if pE is not None else p.mean()
        fq = (q * qE).sum() if qE is not None else q.mean()
        return fp - fq 

emd_loss = EMDLoss()

def get_emd(p, q, pE, qE, sinkhorn=False):
    if sinkhorn:
        M = ot.dist(p, q, metric="euclidean")
        T = ot.unbalanced.sinkhorn_unbalanced(pE, qE, M, 1, 1e-1)
        T = T / T.sum()
        if T.sum().isnan():
            print("T is nan")
            print(pE, qE)
            print(p, q)
            print(M)
            print(T)
            raise ValueError("T is nan")
        return (M * T).sum()
    assert p.shape[0] == pE.shape[0]
    assert q.shape[0] == qE.shape[0]
    if isinstance(p, torch.Tensor):
        p = p.detach().cpu().numpy()
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()
    if isinstance(pE, torch.Tensor):
        pE = pE.detach().cpu().numpy().flatten()
    if isinstance(qE, torch.Tensor):
        qE = qE.detach().cpu().numpy().flatten()
    # normalize because numpy and OT don't work well together        
    # if np.isclose(pE.sum(), 1.0) and pE.sum() != 1.0:
    #     print("Normalizing pE, sum is", pE.sum())
    #     pE = pE / pE.sum()
    #     print("pE was normalized, sum = ", pE.sum())
    # if np.isclose(qE.sum(), 1.0) and qE.sum() != 1.0:
    #     print("Normalizing qE, sum is", qE.sum())
    #     qE = qE / qE.sum()
    #     print("qE was normalized, sum = ", qE.sum())

    # assert np.sum(pE) == np.sum(qE)

    M = ot.dist(p, q, metric="euclidean")
    emd = ot.emd2(pE, qE, M)
    # # this is not a correct check
    # if p.shape[0] == q.shape[0]:
    #     if (p != q).all() and (pE != qE).all():
    #         assert not isclose(emd, 0.0)
    # else:
        # assert not isclose(emd, 0.0)
    if isclose(emd, 0.0):
        return -1
    return emd


def get_emd_batch(x, y, xp, yp):
    assert x.size(0) == y.size(0) == xp.size(0) == yp.size(0)
    emds = []
    for i in range(x.size(0)):
        emds.append(get_emd(x[i], y[i], xp[i], yp[i]))
    # return same as type of x[0]

    if isinstance(x, torch.Tensor):
        return torch.tensor(emds).view(-1, 1)
    elif isinstance(x, np.ndarray):
        return np.array(emds).reshape(-1, 1)
    return emds


def cos_sine(theta):
    if type(theta) == torch.Tensor:
        return torch.concat([torch.cos(theta), torch.sin(theta)], dim=1)
    elif type(theta) == np.ndarray:
        return np.concatenate([np.cos(theta), np.sin(theta)], axis=1)


def hinge_loss(pred, target, margin=1, reduction="mean"):
    if reduction not in ["mean", "sum", "none"]:
        raise ValueError("reduction must be one of ['mean', 'sum', 'none']")
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if pred.dim() <= 1:
        raise ValueError("pred must have at least 2 dimensions")

    loss = torch.max(torch.zeros_like(pred), margin - pred * target)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


# gradient reversal layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd=1):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        lambd = ctx.lambd
        return grad_output.neg() * lambd, None

grad_reverse = GradReverse().apply


def nearest_neighbor_distance():
    pass