import torch
from matplotlib import pyplot as plt
from model import get_model
from tqdm import tqdm
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
from util import get_emd, cos_sine, hinge_loss
from matplotlib.animation import FFMpegWriter
import time


SEED = 0
np.random.seed(SEED)
torch.manual_seed(2)


EPOCHS = 1000
N = 10
N_circ = 5
N_theta = 10
theta = np.linspace(0, 2 * np.pi, N_theta).reshape(-1, 1)
ps = []
for i in range(N_circ):
    ps.append(np.random.rand(N, 2) + np.random.randint(0, 10, size=(2,)) - 5)
ps = np.concatenate(ps, axis=0)

qs = []
for i in range(N_circ):
    loc = np.zeros(2)
    r = 1
    qs.append(r * (cos_sine(theta) + loc))
qs = np.concatenate(qs, axis=0)

# Eps = np.ones(len(ps)) / len(ps)
Eps = torch.rand(len(ps))
Eps /= Eps.sum()
Eqs = torch.rand(N_circ)
Eqs = Eqs.repeat(N_theta) / N_theta
Eqs /= Eqs.sum()


emd_true = get_emd(ps, qs, Eps, Eqs)
print(emd_true)


SAVE = True
if SAVE:
    mpl.use("Agg")
else:
    mpl.use("WebAgg")
LR = 10
LR_f = 1
gamma = (LR_f / LR)**(1 / EPOCHS)
LRq = 1
LRq_f = 5e-2
gammaq = (LRq_f / LRq)**(1 / EPOCHS)
USE_NORM = True
ALWAYS_NORM = False
pbar = tqdm(total=EPOCHS) if SAVE else EPOCHS

# Plot
fig, ax = plt.subplots()
plt.scatter(ps[:, 0], ps[:, 1], s=Eps * 200, c="crimson")
scatter = plt.scatter(qs[:, 0], qs[:, 1], s=Eqs * 100, c="royalblue")
text = plt.text(0.01, 0.01, "", transform=ax.transAxes)

ps = torch.tensor(ps, dtype=torch.float32)
r = torch.randint(1, 2, size=(N_circ, 1), dtype=torch.float32, requires_grad=True)
loc = torch.randn((N_circ, 2), dtype=torch.float32, requires_grad=True)
theta = torch.tensor(theta, dtype=torch.float32)

qs = torch.concat([r * cos_sine(theta) + loc for r, loc in zip(r, loc)])
Eps = torch.tensor(Eps, dtype=torch.float32).view(-1, 1)
# Eqs = torch.tensor(Eqs, dtype=torch.float32).view(-1, 1)
Ei = torch.rand(N_circ, dtype=torch.float32, requires_grad=True)


model = get_model(latent_dim=32, use_norm=USE_NORM, always_norm=ALWAYS_NORM)
optim = torch.optim.SGD(model.parameters(), lr=LR, dampening=0.9)
params = [{'params': r, 'lr': 6e-2, 'dampening': 0.},
          {'params': loc, 'lr': 10},
        #   {'params': Ei, 'lr': 0}
          ]
optim_q = torch.optim.SGD(params, lr=LRq, momentum=0.02, dampening=0.9)
# scheduler_q = torch.optim.lr_scheduler.OneCycleLR(
#     optim_q, max_lr=LRq, steps_per_epoch=1, epochs=EPOCHS)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=gamma)
scheduler_q = torch.optim.lr_scheduler.StepLR(optim_q, step_size=1, gamma=gammaq)
max_emd = 0

targets = torch.vstack([-torch.ones_like(Eps), torch.ones(len(qs), 1)])


def update(i, save=False):
    Eqs = torch.softmax(Ei.repeat_interleave(N_theta), 0).view(-1, 1)
    qs = torch.concat([r * cos_sine(theta) + loc for r, loc in zip(r, loc)])
    emd_true = get_emd(ps, qs, Eps, Eqs)
    optim.zero_grad()
    optim_q.zero_grad()
    mp = model(ps)
    mq = model(qs)
    E0 = (mp * Eps).sum()
    E1 = (mq * Eqs).sum()
    loss = E0 - E1
    # breakpoint()
    with torch.no_grad():
        emd_kr = -loss.item()
    # loss = hinge_loss(torch.vstack([mp, mq]), targets, reduction="mean")
    loss.backward()
    for group in params:
        param = group["params"][0]
        param.grad = - param.grad * (1 + torch.randn_like(param) * 1e-2)
    optim_q.step()
    # if emd_kr > 0:
    max_emd = emd_kr
    delta = (max_emd - emd_true) / emd_true * 100
    message = f"{max_emd:.3f} vs {emd_true:.3f} - Delta: {delta:.2f}% @ {i}"
    if save:
        pbar.set_description(message)
        pbar.update()
    if i % 1 == 0:
        optim.step()
        scatter.set_offsets(qs.detach().numpy())
        scatter.set_sizes(Eqs.detach().numpy().flatten() * 100)
        text.set_text(message)
    scheduler_q.step()
    scheduler.step()
    return [scatter, text]


animation = FuncAnimation(fig, update, frames=EPOCHS,
                          repeat=False, blit=True, fargs=(SAVE, ))
plt.tight_layout()

if SAVE:
    timestamp = time.strftime("%m%d%H%M%S")
    # animation.save("joint_train_OC1.mp4", fps=60)
    name = f"animations/{timestamp}Circles{N_circ}_parts{N}.mp4"
    if ALWAYS_NORM:
        name = name.replace(".mp4", "_alwaysnorm.mp4")
    writer = FFMpegWriter(fps=60, metadata=dict(artist="Me"), bitrate=1800)
    animation.save(name, writer=writer)
    print("saved to ", name)
    plt.close(fig)
else:
    plt.show()
