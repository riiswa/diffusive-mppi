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
SIGMAS = [0.0, 5e-3, 1e-2, 1.5e-2, 2.5e-2, 5e-2, 0.1]
N_TRAJ = 25

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
    noise   = random.normal(kd, (T,)) * (sigma * state_range[1])
    def step(s, inp):
        u, d = inp
        sn = step_fn(s, u)
        sn = sn.at[1].add(d)
        return sn, sn
    _, traj = jax.lax.scan(step, init_state, (actions, noise))
    return traj  # (T, 2)

sample = jax.jit(vmap(rollout, in_axes=(0, None)))

# ── plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, len(SIGMAS), figsize=(len(SIGMAS) * 2.25, 2.25),
                         sharey=True)

traj_colors = plt.cm.plasma(np.linspace(0.05, 0.88, N_TRAJ))  # swap turbo → plasma

PANEL_BG   = '#f4f4f4'   # barely-there gray
GOAL_COLOR = '#aaaaaa'   # soft gray for the goal line

from matplotlib.collections import LineCollection

# one colormap per trajectory is replaced by a single shared time colormap
TIME_CMAP = plt.cm.magma_r

for i, (ax, sigma) in enumerate(zip(axes, SIGMAS)):
    key  = random.PRNGKey(int(sigma * 1e6) + 1)
    keys = random.split(key, N_TRAJ)
    trajs = np.array(sample(keys, float(sigma)))

    ax.set_facecolor('#fafafa')

    for traj in trajs:
        # build segments: shape (T-1, 2, 2)
        points   = traj[:, :2]                          # (T, 2)
        segs     = np.stack([points[:-1], points[1:]], axis=1)  # (T-1, 2, 2)
        t_vals   = np.linspace(0, 1, len(segs))         # time normalised 0→1

        lc = LineCollection(segs, cmap=TIME_CMAP,
                            norm=plt.Normalize(0, 1),
                            alpha=0.9, lw=2)
        lc.set_array(t_vals)
        ax.add_collection(lc)

    percent = sigma * 100
    if sigma == 0 or percent == int(percent):
        label = rf'${int(percent)}\%$'
    else:
        label = rf'${percent:.1f}\%$'
    ax.text(0.05, 0.95, label, transform=ax.transAxes,
            ha='left', va='top', fontsize=20, color='#333',
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.8))

    ax.set_xlim(-1.2, 0.6)
    ax.set_ylim(-0.08, 0.08)
    ax.axvline(0.45, linestyle='--', color='#aaaaaa', alpha=0.9, lw=1.2)
    ax.set_box_aspect(1)
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])



plt.tight_layout(w_pad=0.3)
os.makedirs("figures", exist_ok=True)
plt.savefig('figures/traj_viz.pdf', bbox_inches='tight', dpi=150)
plt.savefig('figures/traj_viz.png', bbox_inches='tight', dpi=150)
print('Saved -> traj_viz.pdf/png')