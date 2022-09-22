import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from model import get_model
from util import get_emd, emd_loss, grad_reverse, cos_sine
import os
from multiprocessing import Pool
from itertools import product
from functools import partial
# from ray.util.multiprocessing import Pool

def run(
    nsubjet=3,
    nclusters=3,
    nparticles=10,
    scale=4,
    seed=0,
):
    torch.manual_seed(seed)

    model = get_model(
        use_norm=True,
        input_dim=2,
        latent_dim=128,
        always_norm=False,
        ngroups=32 // 2,
        metric=2,
    )
    p = (torch.arange(0, nsubjet) + 0.5) * (scale / nsubjet)
    p = torch.vstack([p, torch.ones(nsubjet) * scale / 2]).T.float()
    p = p.requires_grad_()
    E_p = torch.ones(len(p)).view(-1, 1) / len(p)

    q = torch.rand(nclusters, 2) * scale
    q = q.view(1, -1, 2) + torch.randn(nparticles, nclusters, 2) * 0.1
    q = q.view(-1, 2)
    E_q = torch.ones(len(q)).view(-1, 1) / len(q)

    def tensors_numpy(*args):
        return list(map(lambda x: x.detach().numpy(), args))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    optimizer_p = torch.optim.SGD((p,), lr=1e-2, momentum=0.1, dampening=0.6)
    SCHEDULER = False
    EPOCHS = 5000
    STEPS = 20
    WAIT = 100
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, EPOCHS, eta_min=1e-5
        )
    pbar = tqdm(range(EPOCHS), leave=False)

    # for epoch in pbar:
    def step(i, pbar=pbar):
        converged = False
        for _ in range(i):
            if converged:
                break
            counter = 0
            emd_nn = 0
            while abs(emd_nn) <= 1e-3:
                counter += 1
                optimizer.zero_grad()
                optimizer_p.zero_grad()
                fp = model(grad_reverse(p))
                fq = model(q)
                loss = (fp * E_p).sum() - (fq * E_q).sum()
                # loss = get_emd(p, q, E_p.view(-1), E_q.view(-1), sinkhorn=True)
                loss.backward()
                optimizer.step()
                emd_nn = -loss.item()
                if counter == WAIT:
                    print("done waiting")
                    converged = True
                    break
            optimizer_p.step()
            if SCHEDULER:
                scheduler.step()
            emd_true = get_emd(p, q, E_p.view(-1), E_q.view(-1), sinkhorn=True)
            if torch.is_tensor(emd_true):
                emd_true = emd_true.item()
            emd_diff = (emd_nn - emd_true) / emd_true * 100
            msg = f"emd_nn: {emd_nn:+.3f}| "
            msg += f"true: {emd_true:.3f}| "
            msg += f"Delta: {emd_diff:.1f}%"
            pbar.set_description(msg)
            pbar.update()
        return emd_nn, emd_true

    def update(i, *args):
        step(STEPS)
        sc = args[0]
        sc.set_offsets(p.detach().numpy())
        return args

    def init():
        fig, ax = plt.subplots()
        sc = ax.scatter(*tensors_numpy(p)[0].T, c="r")
        ax.scatter(*tensors_numpy(q)[0].T, c="b")
        return fig, ax, sc

    # fig, ax, sc = init()
    # ani = FuncAnimation(fig, update, frames=range(EPOCHS//STEPS), init_func=init, fargs=(sc, ), blit=True)
    # ani.save("3subjet.mp4", dpi=300, fps=30)
    emd_nn_min = 1e10
    emd_min = -1
    for i in range(EPOCHS):
        emds = step(1)
        if emds[0] < emd_nn_min:
            emd_nn_min = emds[0]
            emd_min = emds[1]
            p_min = p.detach().clone()
    return emd_min, emd_nn_min, *emds, p_min, p.data.detach(), q


def main(
i,
nsubjet = 3,
nclusters = 3,
nparticles = 10,
):
    fname = f"logs/subjet_{nsubjet}_{nclusters}_{nparticles}({i}).txt"
    if os.path.exists(fname):
        return
    with open(fname, "w") as f:
        seed = nsubjet * i**2 + nclusters * i + nparticles
        emd_nn, emd_true, last_emd_nn, last_emd, p_min, p, q = run(
            nsubjet=nsubjet, nclusters=nclusters, nparticles=nparticles, seed=seed
        )
        f.write(f"{emd_nn:.3f}, {emd_true:.3f}, {last_emd_nn:.3f}, {last_emd:.3f}\n")
        np.savetxt(f, p_min.detach().numpy())
        np.savetxt(f, p.detach().numpy())
        np.savetxt(f, q.detach().numpy())

nsubjets = np.arange(3, 7)
nclusters = np.arange(3, 7)
nparticles = np.arange(10, 21, 10)
all_inputs = list(product(nsubjets, nclusters, nparticles))
pbar = tqdm(range(len(all_inputs)), position=1)

# # need to multiprocess this
# pool = Pool(32)
# pool.imap_unordered(main, all_inputs)

for i in pbar:
    nsubjet, nclusters, nparticles = all_inputs[i]
    pbar.set_description(f"{nsubjet}, {nclusters}, {nparticles}")
    run_wrapper = partial(main, nsubjet=nsubjet, nclusters=nclusters, nparticles=nparticles)
    pool = Pool(1)
    pool.map(run_wrapper, range(10))