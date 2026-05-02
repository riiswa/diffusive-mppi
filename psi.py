"""
exp_psi_curve.py — plot Ψ(σ) for all sparse environments.

Usage:  python exp_psi_curve.py
Output: psi_curve.pdf
"""
import os

import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
import matplotlib.pyplot as plt

from envs import ENVS

# ── config ────────────────────────────────────────────────────────────────────
K        = 1_000
N_SEEDS  = 256
N_SIGMA  = 200
SIGMAS   = np.geomspace(1e-3, 1., N_SIGMA)
LAM      = 0.05
SEED     = 42

# ── psi estimator ─────────────────────────────────────────────────────────────

def make_psi_fn(env, K):
    step_fn     = env['step_fn']
    cost_fn     = env['cost_fn']
    init_state  = env['reset_fn'](random.PRNGKey(0))   # fixed start
    state_range = env['state_range']
    action_clip = float(env['action_clip'])
    sigma_u     = float(env['sigma_u'])
    N           = int(env['N'])
    U           = jnp.zeros(N)

    def psi_fn(key, sigma):
        ka, kd = random.split(key)
        eps = random.normal(ka, (K, N)) * sigma_u
        dn  = random.normal(kd, (K, N, state_range.shape[0])) * (sigma * state_range)

        def rollout(eps_k, dn_k):
            def step(s, inp):
                u_nom, e, d = inp
                u  = jnp.clip(u_nom + e, -action_clip, action_clip)
                return step_fn(s, u) + d, cost_fn(step_fn(s, u) + d)
            _, costs = jax.lax.scan(step, init_state, (U, eps_k, dn_k))
            return costs.sum()

        G = vmap(rollout)(eps, dn)
        w = jnp.exp(-(G - G.min()) / LAM)
        w = w / w.sum()
        H = -jnp.sum(w * jnp.log(w + 1e-12))
        return 1. - H / jnp.log(float(K))

    return psi_fn


# ── run ───────────────────────────────────────────────────────────────────────

keys = random.split(random.PRNGKey(SEED), N_SEEDS)
results = {}

for name, env in ENVS.items():
    print(f'  {name}')
    run = jax.jit(vmap(make_psi_fn(env, K), in_axes=(0, None)))
    means, stds = [], []
    for sigma in SIGMAS:
        v = np.array(run(keys, float(sigma)))
        means.append(v.mean()); stds.append(v.std())
    results[name] = (np.array(means), np.array(stds))

# ── plot ──────────────────────────────────────────────────────────────────────

n = len(ENVS)
fig, axes = plt.subplots(1, n, figsize=(2.25*n, 2.), sharey=True)

for ax, (name, (mu, std)) in zip(axes, results.items()):
    ax.plot(SIGMAS, mu, color='steelblue', lw=2.)
    ax.fill_between(SIGMAS, mu-std, mu+std, color='steelblue', alpha=0.2)
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_title(name, fontsize=16)
    ax.set_xlabel(r'$\sigma$', fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel(r'$\psi(\sigma)$', fontsize=14)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/" + 'psi_curve.pdf', bbox_inches='tight', dpi=200)
plt.savefig("figures/" +'psi_curve.png', bbox_inches='tight', dpi=200)
print('Saved -> psi_curve.pdf')