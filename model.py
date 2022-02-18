import torch
from monotonenorm import direct_norm, GroupSort


class TimesN(torch.nn.Module):
    def __init__(self, n: float):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n * x


def get_model(dev=None, size=1024, use_norm=True, alway_norm=True):
    if use_norm:
        return torch.nn.Sequential(
            direct_norm(torch.nn.Linear(2, size), kind="two-inf", always_norm=alway_norm),
            GroupSort(2),
            direct_norm(torch.nn.Linear(size, size), kind="inf", always_norm=alway_norm),
            GroupSort(2),
            direct_norm(torch.nn.Linear(size, size), kind="inf", always_norm=alway_norm),
            GroupSort(2),
            direct_norm(torch.nn.Linear(size, 1), kind="inf", always_norm=alway_norm),
            TimesN(1.01)
        ).to(dev or "cpu")
    else:
        return torch.nn.Sequential(
            torch.nn.Linear(2, size),
            GroupSort(2),
            torch.nn.Linear(size, size),
            GroupSort(2),
            torch.nn.Linear(size, size),
            GroupSort(2),
            torch.nn.Linear(size, 1),
        ).to(dev or "cpu")
