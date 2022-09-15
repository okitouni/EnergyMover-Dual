import torch
from monotonenorm import direct_norm, GroupSort
from functools import partial
from torch import nn


class TimesN(torch.nn.Module):
    def __init__(self, n: float):
        super().__init__()
        self.n = n

    def forward(self, x):
        return self.n * x

def reset_parameters(model):
    for m in model.children():
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

def get_model(
    dev=None,
    input_dim=2,
    latent_dim=1024,
    output_dim=1,
    depth=4,
    ngroups=2,
    use_norm=True,
    always_norm=True,
    dropout=0.0,
    lip_const = 1.0,
    metric=2
):
    max_norm = lip_const ** (1 / depth + 1)
    
    assert metric in [1, 2]
    norm_kind_input_layer = "two-inf" if metric == 2 else "one-inf"
    norm_func_input = (
        partial(direct_norm, kind=norm_kind_input_layer, always_norm=always_norm, max_norm=max_norm)
        if use_norm
        else lambda x: x
    )
    norm_func = (
        partial(direct_norm, kind="inf", always_norm=always_norm, max_norm=max_norm)
        if use_norm
        else lambda x: x
    )

    layers = [norm_func_input(torch.nn.Linear(input_dim, latent_dim))]
    layers.append(GroupSort(n_groups=ngroups))
    layers.append(torch.nn.Dropout(dropout))
    for _ in range(depth - 1):
        layers.append(norm_func(torch.nn.Linear(latent_dim, latent_dim)))
        layers.append(GroupSort(n_groups=ngroups))
        layers.append(torch.nn.Dropout(dropout))
    layers.append(norm_func(torch.nn.Linear(latent_dim, output_dim, bias=False)))
    return torch.nn.Sequential(*layers).to(dev)


class DeepSet(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        depth=1,
        dropout=0.0,
        act=torch.nn.functional.relu,
        pool=torch.mean,
    ):
        super(DeepSet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.act = act
        self.pool = pool
        self.dropout = dropout

        self.boundary_message = nn.ModuleList()
        self.boundary_message.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(depth - 1):
            self.boundary_message.append(nn.Linear(hidden_channels, hidden_channels))

        self.regressor = nn.ModuleList()
        for _ in range(depth - 1):
            self.regressor.append(nn.Linear(hidden_channels, hidden_channels))
        self.regressor.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        # x is a vector of size (n_nodes, n_boundary, in_channels)
        for i in range(self.depth - 1):
            x = self.boundary_message[i](x)
            x = self.act(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.boundary_message[-1](x)
        x = self.pool(x, dim=1)
        if self.pool == torch.max:
            x = x.values
        x = x.view(x.size(0), self.hidden_channels)

        for i in range(self.depth - 1):
            x = self.regressor[i](x)
            x = self.act(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
        x = self.regressor[-1](x)
        return x
