"""
exp_maxdiff_alpha.py — sweep MaxDiff alpha on sparse environments.

Output: maxdiff_alpha_sweep.pdf/png, maxdiff_alpha_sweep.csv
"""

import csv
import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
import matplotlib.pyplot as plt

from envs import ENVS
from planning import make_mppi, metrics

# ── config ────────────────────────────────────────────────────────────────────
K            = 500
LAM          = 0.05
N_SEEDS      = 64
MAXDIFF_EPS  = 1e-3
SEED         = 0

ALPHAS = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]

keys = random.split(random.PRNGKey(SEED), N_SEEDS)

# ── run ───────────────────────────────────────────────────────────────────────

# results[env_name][alpha] = metrics dict
results = {name: {} for name in ENVS}

for alpha in ALPHAS:
    jax.clear_caches()   # release cuFFT scratch allocator between runs
    for name, env in ENVS.items():
        T   = int(env['max_steps'])
        sch = jnp.zeros(T)
        fn  = jax.jit(vmap(
            make_mppi(env, K, LAM, maxdiff_alpha=alpha, maxdiff_eps=MAXDIFF_EPS),
            in_axes=(0, None)
        ))
        _, costs, _, _ = fn(keys, sch)
        m = metrics(np.array(costs))
        results[name][alpha] = m
        print(f'  alpha={alpha:6.2f}  {name:18s}  '
              f'sr={m["success_rate"]*100:5.1f}%  '
              f'cum={m["cum_cost_mean"]:.1f}±{m["cum_cost_std"]:.1f}')

# ── CSV ───────────────────────────────────────────────────────────────────────

with open('maxdiff_alpha_sweep.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['env', 'alpha', 'success_rate',
                                           'mean_step', 'std_step',
                                           'cum_cost_mean', 'cum_cost_std'])
    writer.writeheader()
    for name in ENVS:
        for alpha in ALPHAS:
            m = results[name][alpha]
            writer.writerow(dict(
                env=name, alpha=alpha,
                success_rate=f"{m['success_rate']:.4f}",
                mean_step=f"{m['mean_step']:.2f}",
                std_step=f"{m['std_step']:.2f}",
                cum_cost_mean=f"{m['cum_cost_mean']:.4f}",
                cum_cost_std=f"{m['cum_cost_std']:.4f}",
            ))
print('Saved -> maxdiff_alpha_sweep.csv')

# ── plot ──────────────────────────────────────────────────────────────────────

env_names = list(ENVS.keys())
n         = len(env_names)
fig, axes = plt.subplots(1, n, figsize=(3*n, 2.5), sharey=True)
if n == 1: axes = [axes]

x      = np.arange(len(ALPHAS))
labels = [str(a) for a in ALPHAS]

for ax, name in zip(axes, env_names):
    sr  = [results[name][a]['success_rate'] * 100 for a in ALPHAS]
    ax.bar(x, sr, color='crimson', alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_title(name)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylim(0, 105)
    ax.spines[['top', 'right']].set_visible(False)

axes[0].set_ylabel('Success rate (%)')
plt.tight_layout()
plt.savefig('maxdiff_alpha_sweep.pdf', bbox_inches='tight')
plt.savefig('maxdiff_alpha_sweep.png', bbox_inches='tight', dpi=200)
print('Saved -> maxdiff_alpha_sweep.pdf/png')