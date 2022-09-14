import torch
from torch.utils.data import Dataset
from typing import Iterable
from itertools import combinations


class MakeBatch:
    def __init__(
        self, p: torch.Tensor, q: torch.Tensor, Ep: torch.Tensor, Eq: torch.Tensor
    ) -> None:
        self.p = p
        self.q = q
        self.Ep = Ep.view(-1)
        self.Eq = Eq.view(-1)

    def __call__(self, batch_size: int, seed=None) -> Iterable[torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        p_sample = torch.multinomial(self.Ep, batch_size, replacement=True)
        q_sample = torch.multinomial(self.Eq, batch_size, replacement=True)
        p_sample = self.p[p_sample]
        q_sample = self.q[q_sample]
        return p_sample, q_sample


def slice(arr, center, width=None):
    n = len(arr)
    width = width or n // 2
    lower = max(0, center - width)
    upper = min(n, center + width)
    return arr[lower:upper]


class knnBatch(MakeBatch):
    def __init__(self, p, q, Ep, Eq):
        super().__init__(p, q, Ep, Eq)
        self.p_union_q, sorting = torch.cat((p, q), 0).sort(0)
        self.Ep_union_Eq = torch.cat((Ep, Eq), 0).view(-1)[sorting]
        self.union_labels = torch.cat((torch.zeros(len(p)), torch.ones(len(q))), 0)[
            sorting
        ]

        # self.p, sorting = p.sort(0)
        # self.Ep = Ep[sorting]
        # self.q, sorting = q.sort(0)
        # self.Eq = Eq[sorting]

    def __call__(self, batch_size: int, seed=None) -> Iterable[torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        # center = torch.multinomial(self.Ep_union_Eq.view(-1), 1)
        # sample = slice(self.p_union_q, center, width=batch_size)
        # sample_labels = slice(self.union_labels, center, width=batch_size)
        # p_sample_ = sample[sample_labels == 0].view(-1, self.p.shape[1]).detach()
        # q_sample_ = sample[sample_labels == 1].view(-1, self.q.shape[1])
        # return p_sample_, q_sample_
        pidx = torch.multinomial(self.Ep.view(-1), batch_size, replacement=True)
        qidx = torch.multinomial(self.Eq.view(-1), batch_size, replacement=True)
        if torch.rand(1) > 0.5:
            return self.p[pidx], torch.tensor([])
        else:
            return torch.tensor([]), self.q[qidx]

class MultinomialBatch(MakeBatch):
    def __call__(self, batch_size: int, seed=None) -> Iterable[torch.Tensor]:
        if seed is not None:
            torch.manual_seed(seed)
        p_sample = torch.multinomial(self.Ep, batch_size, replacement=True)
        q_sample = torch.multinomial(self.Eq, batch_size, replacement=True)
        p_sample = self.p[p_sample]
        q_sample = self.q[q_sample]
        return p_sample, q_sample

def generate_data(n_samples, n_points=None, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    if n_points is None:
        n_points = torch.randint(low=1, high=100, size=(n_samples,))
    else:
        assert isinstance(n_points, int)
        n_points = torch.tensor([n_points] * n_samples)
    xs = []
    probs = []
    for n in n_points:
        xs.append(torch.rand(n, 2))
        probabilities = torch.rand(n - 1)
        probabilities = probabilities.sort(descending=False).values
        probabilities = torch.cat(
            [torch.tensor([0.0]), probabilities, torch.tensor([1.0])]
        )
        probabilities = probabilities.diff().view(n, 1)
        probs.append(probabilities)
    return xs, probs


class IrregularDataSet(Dataset):
    def __init__(self, *args):
        assert len(args) >= 1
        for i, arg in enumerate(args):
            assert isinstance(arg, Iterable)
            assert isinstance(arg[0], torch.Tensor)
            for arg_ in args[i + 1 :]:
                assert len(arg_) == len(arg), "all inputs must be the same length"
        self.tensor_sequences = args if len(args) > 1 else [args]
        self.indices = list(combinations(range(len(self.tensor_sequences[0])), 2))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        idx_0, idx_1 = self.indices[idx]
        return [x[idx_0] for x in self.tensor_sequences] + [
            x[idx_1] for x in self.tensor_sequences
        ]
