"""
exp_traj_viz.py — Mountain Car trajectory spread under increasing dynamics noise.
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
import matplotlib.pyplot as plt

from envs import ENVS

# ── config ────────────────────────────────────────────────────────────────────
SIGMAS = [0.0, 5e-3, 1e-2, 1.5e-2, 2.5e-2, 5e-2, 1e-1]
N_TRAJ = 10

# ── rollout ───────────────────────────────────────────────────────────────────
env         = ENVS['Mountain Car']
step_fn     = env['step_fn']
state_range = env['state_range']
sigma_u     = float(env['sigma_u'])
T           = int(env['max_steps'])
init_state  = jnp.array([-0.5, 0.0])

def rollout(key, sigma):
    ka, kd = random.split(key)
    actions = random.normal(ka, (T,)) * sigma_u
    noise   = random.normal(kd, (T, 2)) * (sigma * state_range)
    def step(s, inp):
        u, d = inp
        sn = step_fn(s, u) + d
        return sn, sn
    _, traj = jax.lax.scan(step, init_state, (actions, noise))
    return traj  # (T, 2)

sample = jax.jit(vmap(rollout, in_axes=(0, None)))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(SIGMAS), figsize=(len(SIGMAS)*2.25, 2.25),
                         sharey=True)

traj_colors = plt.cm.turbo(np.linspace(0.05, 0.95, N_TRAJ))

for i, (ax, sigma) in enumerate(zip(axes, SIGMAS)):
    key  = random.PRNGKey(int(sigma * 1e6) + 1)
    keys = random.split(key, N_TRAJ)
    trajs = np.array(sample(keys, float(sigma)))
    for traj, c in zip(trajs, traj_colors):
        ax.plot(traj[:, 0], traj[:, 1], color=c, alpha=0.6, lw=2.)
    ax.set_title(rf'$\sigma = {sigma}$', fontsize=18)
    # ax.set_xlabel('position', fontsize=16)
    # if i == 0:
    #     ax.set_ylabel('velocity', fontsize=16)
    ax.set_xlim(-1.2, 0.6)
    ax.set_ylim(-0.08, 0.08)
    ax.axvline(0.45, linestyle='--', color='black', alpha=0.25)
    ax.set_box_aspect(1)
    # ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout(w_pad=0.5)
plt.savefig('figures/traj_viz.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/traj_viz.png', bbox_inches='tight', dpi=150)
print('Saved -> traj_viz.pdf/png')