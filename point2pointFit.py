import torch
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import matplotlib as mpl
from model import get_model
from util import get_emd, emd_loss, grad_reverse
from data import MultinomialBatch

torch.manual_seed(42)
torch.set_default_dtype(torch.float32)

EPOCHS = 20000
PLOT = True
SAVE = True
N = 8
EMBED_DIM = 2

# q = torch.randn(2 * N, EMBED_DIM)
# E_q = torch.ones(2 * N, 1) / (2 * N)
# p = torch.randn(N, EMBED_DIM)
# p = torch.repeat_interleave(p, 2, dim=0).requires_grad_()
# E_p = torch.ones(2 * N, 1) / (2 * N)
scale = 2
q = torch.rand(N, EMBED_DIM) * scale
# E_q = torch.ones(N, 1) / N
E_q = torch.rand(N, 1)
E_q = E_q / E_q.sum()
# p = (q.clone() + torch.randn_like(q)).requires_grad_()
p = (torch.rand(N, EMBED_DIM) * scale).requires_grad_()
E_p_ = torch.ones(p.shape[0], 1) / p.shape[0]
E_p_.requires_grad_(False)
# E_p = torch.rand(N, 1) + 1
E_p = E_p_.detach().softmax(dim=0)
print(E_p.sum(), E_q.sum())


def tensors_numpy():
    E_p = E_p_.detach().softmax(dim=0)
    return map(lambda x: x.detach().numpy(), [p, q, E_p.view(-1), E_q.view(-1)])


emd_true = get_emd(*tensors_numpy())
print(emd_true)


torch.manual_seed(0)
model = get_model(
    use_norm=True,
    input_dim=EMBED_DIM,
    latent_dim=128,
    always_norm=True,
    ngroups=32 // 2,
    metric=1
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# optimizer_p = torch.optim.SGD((p,), lr=1e-1, momentum=0.1, dampening=0.6)
optimizer_p = torch.optim.SGD((p,), lr=1e-1, momentum=0.1, dampening=0.6)
SCHEDULER = False
if SCHEDULER:
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-2, steps_per_epoch=1, epochs=EPOCHS + 2
    )


if PLOT:
    E_scale = 100
    p_numpy, q_numpy, E_p_numpy, E_q_numpy = tensors_numpy()
    yscale = 2 / E_p_numpy.max()
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig, (ax2) = plt.subplots(1, 1, figsize=(5, 5))
    if EMBED_DIM == 1:
        domain = torch.linspace(0, 1, 100).view(-1, 1) * scale
        p_numpy = np.hstack([p_numpy, E_p_numpy.reshape(-1, 1) * yscale])
        q_numpy = np.hstack([q_numpy, -E_q_numpy.reshape(-1, 1) * yscale])
        (line,) = ax2.plot(domain.numpy().flatten(), np.zeros(100), c="black")
    else:
        linspace = torch.linspace(0, 1, 100) * scale
        domain = torch.cartesian_prod(linspace, linspace)
        heatmap = ax2.scatter(*domain.T, cmap="blues_r", s=6, marker="s")
        # vector field on domain
        subdomain = domain.view(200, 100)[::5, ::5].reshape(-1, 2)
        subdomain.requires_grad_()
        n = subdomain.shape[0]
        vfield = ax2.quiver(
            *subdomain.detach().T,
            np.ones(n),
            np.ones(n),
            scale=5,
            scale_units="xy",
            color="black",
        )
    # ax1.scatter(p_numpy[:, 0], p_numpy[:, 1], s=E_p * E_scale, c="crimson")
    # ax1.scatter(q_numpy[:, 0], q_numpy[:, 1], s=E_q * E_scale, c="royalblue")
    ax2.scatter(q_numpy[:, 0], q_numpy[:, 1], s=E_q_numpy * E_scale, c="royalblue")
    sc = ax2.scatter(p_numpy[:, 0], p_numpy[:, 1], s=E_p_numpy * E_scale, c="crimson")
    text = ax2.text(0., 0.9, "", transform=ax2.transAxes)
    plt.tight_layout()


pbar = tqdm(total=EPOCHS)



def train_step(p, q, Ep=None, Eq=None):
    fp = model(p)
    fq = model(q)
    return emd_loss(fp, fq, Ep, Eq)


def update(i):
    optimizer.zero_grad()
    optimizer_p.zero_grad()
    # some stochastic steps
    # for _ in range(10):
    #     optimizer.zero_grad()
    #     optimizer_p.zero_grad()
    #     p_sample, q_sample = batcher(2)
    #     loss = train_step(p_sample.detach(), q_sample)
    #     loss.backward()
    #     optimizer.step()
    # optimizer_p.step()
    E_p = E_p_.softmax(dim=0)
    loss = train_step(grad_reverse(p), q, grad_reverse(E_p), E_q)
    # fp = model(p)
    # fq = model(q)
    # loss = (fp * E_p).sum() - (fq * E_q).sum()
    emd_nn = -loss.item()
    # loss += hkr_lambda * F.hinge_embedding_loss(
    # torch.vstack([fp, fq]), - targets, margin=1)
    # loss = F.cross_entropy(torch.vstack([fp, fq]), targets)
    # loss = torch.log(1  + loss)
    # loss = torch.log(loss + 1)
    loss.backward()
    # p.grad = - p.grad * (1 + torch.rand_like(p) * 0.01)
    optimizer.step()
    optimizer_p.step()
    if SCHEDULER:
        scheduler.step()
    emd_true = get_emd(*tensors_numpy())
    emd_diff = (emd_nn - emd_true) / emd_true * 100
    msg = f"emd_nn: {emd_nn:+.3f}| "
    msg += f"true: {emd_true:.3f}| "
    msg += f"Delta: {emd_diff:.1f}%"
    pbar.set_description(msg)
    pbar.update()
    if i % 10 == 0:
        if PLOT:
            p_numpy, q_numpy, E_p_numpy, E_q_numpy = tensors_numpy()
            text.set_text(
                f"epoch: {i} | emd:{emd_true:.3f}"
            )
            output = model(subdomain)
            grads = torch.autograd.grad(
                output, subdomain, grad_outputs=torch.ones_like(output)
            )[0]
            with torch.no_grad():
                output = model(domain).numpy().flatten()
            if EMBED_DIM == 1:
                line.set_ydata(output)
            elif EMBED_DIM == 2:
                # if True:
                # vmax = output.max()
                # vmin = output.min()
                # vmin, vmax = -0.004905564, -0.0045922818
                output = output - output.min()
                vfield.set_UVC(*grads.T)
                heatmap.set_color(cm.get_cmap("Blues_r")(output))
                sc.set_offsets(p_numpy)


if PLOT:
    animation = FuncAnimation(
        fig,
        update,
        frames=range(EPOCHS),
        repeat=False,
    )
    if SAVE:
        timestamp = time.strftime("%m%d-%H%M%S")
        name = f"animations/{timestamp}-{N}ptKR.mp4"
        mpl.use("Agg")
        animation.save(name, writer="ffmpeg", fps=60)
        print("saved to {}".format(name))
    else:
        plt.show()
else:
    for i in range(EPOCHS):
        update(i)
