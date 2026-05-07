"""
exp_psi_star.py — ablation on ψ* for DNI-MPPI.

For each φ* in PHI_STARS: adaptive DNI-MPPI (bars).
One additional run per env: constant noise (frozen=True) shown as dashed line.

Outputs: psi_star_cost.pdf/png, psi_star_success.pdf/png
"""

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
import matplotlib.pyplot as plt

from envs import ENVS
from planning import make_mppi, metrics

# ── config ────────────────────────────────────────────────────────────────────
K         = 500
LAM       = 0.05
N_SEEDS   = 64
ETA_NU    = 0.10
NU_MAX    = 5.0
SEED      = 0

PHI_STARS = [0.0, 0.01, 0.1, 0.25, 0.5, 0.9]
COLORS    = plt.cm.plasma(np.linspace(0.1, 0.85, len(PHI_STARS)))

keys = random.split(random.PRNGKey(SEED), N_SEEDS)

# ── run adaptive ──────────────────────────────────────────────────────────────

results = {name: {} for name in ENVS}

for phi_star in PHI_STARS:
    jax.clear_caches()
    for name, env in ENVS.items():
        T  = int(env['max_steps'])
        fn = jax.jit(vmap(
            make_mppi(env, K, LAM, phi_star, ETA_NU, NU_MAX,
                      use_noise=True),
            in_axes=(0, None)
        ))
        _, costs, _, _ = fn(keys, jnp.zeros(T))
        m = metrics(np.array(costs))
        results[name][phi_star] = m
        print(f'  φ*={phi_star:.2f}  {name:18s}  '
              f'sr={m["success_rate"]*100:5.1f}%  '
              f'cum={m["cum_cost_mean"]:.1f}±{m["cum_cost_std"]:.1f}')

# ── run constant (one per env) ────────────────────────────────────────────────

jax.clear_caches()
results_const = {}

for name, env in ENVS.items():
    T  = int(env['max_steps'])
    fn = jax.jit(vmap(
        make_mppi(env, K, LAM, use_noise=True, frozen=True),
        in_axes=(0, None)
    ))
    _, costs, _, _ = fn(keys, jnp.zeros(T))
    m = metrics(np.array(costs))
    results_const[name] = m
    print(f'  constant  {name:18s}  '
          f'sr={m["success_rate"]*100:5.1f}%  '
          f'cum={m["cum_cost_mean"]:.1f}±{m["cum_cost_std"]:.1f}')

# ── plot ──────────────────────────────────────────────────────────────────────

env_names = list(ENVS.keys())
n_envs    = len(env_names)
x         = np.arange(len(PHI_STARS))
labels    = [str(ps) for ps in PHI_STARS]


def bar_plot(metric_key, ylabel, fname, ylim=None):
    fig, axes = plt.subplots(1, n_envs, figsize=(2.25*n_envs, 2.5), sharey=True)
    if n_envs == 1: axes = [axes]
    # x positions: PHI_STARS bars + 1 constant bar at the end
    x_all    = np.arange(len(PHI_STARS) + 1)
    x_phi    = x_all[:len(PHI_STARS)]
    x_const  = x_all[-1]
    lbl_all  = labels + ['cnst']
    colors_all = list(COLORS) + ['black']
    for ax, name in zip(axes, env_names):
        vals = [results[name][ps][metric_key] for ps in PHI_STARS]
        errs = [results[name][ps]['cum_cost_std'] for ps in PHI_STARS] \
               if metric_key == 'cum_cost_mean' else [0.] * len(PHI_STARS)
        if metric_key == 'success_rate':
            vals = [v * 100 for v in vals]
        # adaptive bars
        ax.bar(x_phi, vals, color=COLORS, alpha=0.85,
               yerr=errs, error_kw=dict(lw=1., capsize=2.))
        # constant bar
        ref  = results_const[name][metric_key]
        rerr = results_const[name]['cum_cost_std'] if metric_key == 'cum_cost_mean' else 0.
        if metric_key == 'success_rate': ref *= 100
        ax.bar(x_const, ref, color='black', alpha=0.85,
               yerr=rerr, error_kw=dict(lw=1., capsize=2.))
        ax.set_xticks(x_all)
        ax.set_xticklabels(lbl_all, rotation=45, ha='right', fontsize=12)
        ax.set_xlabel(r'$\psi^*$', fontsize=14)
        ax.set_title(name, fontsize=16)
        if ylim: ax.set_ylim(*ylim)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/" + fname + '.pdf', bbox_inches='tight')
    plt.savefig("figures/" + fname + '.png', bbox_inches='tight', dpi=200)
    print(f'Saved -> {fname}.pdf/png')
    plt.close()


bar_plot('cum_cost_mean', 'Cumulative cost',  'psi_star_cost')
bar_plot('success_rate',  'Success rate (%)', 'psi_star_success', ylim=(0, 105))