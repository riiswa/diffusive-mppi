"""
exp_benchmark.py — compare five methods on sparse + dense suites.

Methods
-------
  Vanilla MPPI  : white noise, no dynamics noise
  LF-MPPI       : colored noise (beta=2), no dynamics noise
  MaxDiff MPPI  : white noise + MaxDiff log-det bonus (Berrueta et al. 2024)
  DNI-MPPI      : white noise + adaptive dynamics noise (ours)
  DNI-LF-MPPI   : colored noise + adaptive dynamics noise (combined)

Outputs
-------
  benchmark_sparse.pdf/png  — obs trajectories (mean ± std)
  benchmark_dense.pdf/png   — obs trajectories (mean ± std)
  benchmark_nu.pdf/png      — ν(t) for adaptive methods (sparse suite)
  benchmark_results.csv     — all metrics
"""

import csv
import os

import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
import matplotlib.pyplot as plt

from envs import ENVS, ENVS_DENSE
from planning import make_mppi, metrics

# ── config ────────────────────────────────────────────────────────────────────
K        = 500
LAM      = 0.05
N_SEEDS  = 64
PHI_STAR = 0.25
ETA_NU   = 0.10
NU_MAX   = 5.0
SEED     = 0

# MaxDiff: alpha=1.0 recommended for sparse binary rewards in [0,1].
# The paper uses alpha=100 for dense MuJoCo rewards in [-500,500]; scaling
# by the reward range gives ~1.0 here. eps regularises the rank-1 covariance
# C = sigma_u^2 * J_u J_u^T + eps*I (scalar action -> rank 1 without eps).
MAXDIFF_ALPHA = 1.0
MAXDIFF_EPS   = 1e-3

PALETTE = {
    'Vanilla':      'steelblue',
    'LF-MPPI':      'seagreen',
    'MaxDiff':      'crimson',
    'DNI-MPPI':     'darkorange',
    'DNI-LF-MPPI':  'purple',
}

keys = random.split(random.PRNGKey(SEED), N_SEEDS)


# ── runner ────────────────────────────────────────────────────────────────────

def run_env(env, name):
    beta      = float(env['lf_beta'])
    T         = int(env['max_steps'])
    sch_white = jnp.zeros(T)
    sch_color = jnp.full(T, beta)

    out = {}
    for label, sch, adaptive, md_alpha in [
        ('Vanilla',     sch_white, False, 0.),
        ('LF-MPPI',     sch_color, False, 0.),
        # ('MaxDiff',     sch_white, False, MAXDIFF_ALPHA),
        ('DNI-MPPI',    sch_white, True,  0.),
        ('DNI-LF-MPPI', sch_color, True,  0.),
    ]:
        ps = PHI_STAR if adaptive else 0.
        en = ETA_NU   if adaptive else 0.
        use_noise = adaptive
        fn = jax.jit(vmap(
            make_mppi(env, K, LAM, ps, en, NU_MAX,
                      use_noise=use_noise,
                      maxdiff_alpha=md_alpha, maxdiff_eps=MAXDIFF_EPS),
            in_axes=(0, None)
        ))
        obs, costs, phis, nus = fn(keys, sch)
        m = metrics(np.array(costs))
        print(f'  {label:16s}  sr={m["success_rate"]*100:5.1f}%  '
              f'step={m["mean_step"]:.1f}±{m["std_step"]:.1f}  '
              f'cum={m["cum_cost_mean"]:.1f}±{m["cum_cost_std"]:.1f}')
        out[label] = dict(obs=np.array(obs), costs=np.array(costs),
                          phis=np.array(phis), nus=np.array(nus), metrics=m)
    return out


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_suite(env_dict, suite_results, fname):
    n = len(env_dict)
    fig, axes = plt.subplots(1, n, figsize=(2.25*n, 2), sharey=False)
    if n == 1: axes = [axes]

    for ax, (name, env) in zip(axes, env_dict.items()):
        res = suite_results[name]
        T   = int(env['max_steps'])
        t   = np.arange(T)
        for label, color in PALETTE.items():
            mu  = res[label]['obs'].mean(0)
            std = res[label]['obs'].std(0)
            ax.plot(t, mu, color=color, lw=1.5, label=label)
            ax.fill_between(t, mu-std, mu+std, color=color, alpha=0.15)
        ax.axhline(env['goal_line'], color='k', ls='--', lw=1., alpha=0.5)
        ax.set_ylim(*env['obs_ylim'])
        ax.set_title(name, fontsize=16)
        ax.set_xlabel('step',fontsize=14)
        ax.set_ylabel(env['obs_label'], fontsize=14)
        ax.tick_params()
        ax.spines[['top', 'right']].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5,
               frameon=False, bbox_to_anchor=(0.5, -0.15),  fontsize=12)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/" + fname + '.pdf', bbox_inches='tight', dpi=200)
    plt.savefig("figures/" + fname + '.png', bbox_inches='tight', dpi=200)
    print(f'Saved -> {fname}.pdf/png')
    plt.close()


def plot_nu(env_dict, sparse_results, fname):
    """ν(t) = log(σ/σ₀) evolution for adaptive methods on sparse suite."""
    n = len(env_dict)
    fig, axes = plt.subplots(1, n, figsize=(2.25*n, 2), sharey=True)
    if n == 1: axes = [axes]

    for ax, (name, env) in zip(axes, env_dict.items()):
        res = sparse_results[name]
        T   = int(env['max_steps'])
        t   = np.arange(T)
        for label in ('DNI-MPPI', 'DNI-LF-MPPI'):
            nu     = res[label]['nus']          # (n_seeds, T)
            mu_nu  = nu.mean(0)
            std_nu = nu.std(0)
            ax.plot(t, mu_nu, color=PALETTE[label], lw=1.5, label=label)
            ax.fill_between(t, mu_nu - std_nu, mu_nu + std_nu,
                            color=PALETTE[label], alpha=0.15)
        ax.set_title(name, fontsize=16)
        ax.set_xlabel('step', fontsize=14)
        ax.tick_params()
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].set_ylabel(r'$\nu(t) = \log(\sigma/\sigma_0)$', fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.15), fontsize=12)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/" + fname + '.pdf', bbox_inches='tight')
    plt.savefig("figures/" + fname + '.png', bbox_inches='tight')
    print(f'Saved -> {fname}.pdf/png')
    plt.close()


# ── CSV export ────────────────────────────────────────────────────────────────

def save_csv(sparse_results, dense_results, fname='benchmark_results.csv'):
    fields = ['suite', 'env', 'method',
              'success_rate', 'mean_step', 'std_step',
              'cum_cost_mean', 'cum_cost_std']
    rows = []
    for suite_name, suite_results in [('sparse', sparse_results),
                                       ('dense',  dense_results)]:
        for env_name, env_out in suite_results.items():
            for method, data in env_out.items():
                m = data['metrics']
                rows.append({
                    'suite':         suite_name,
                    'env':           env_name,
                    'method':        method,
                    'success_rate':  f"{m['success_rate']:.4f}",
                    'mean_step':     f"{m['mean_step']:.2f}",
                    'std_step':      f"{m['std_step']:.2f}",
                    'cum_cost_mean': f"{m['cum_cost_mean']:.4f}",
                    'cum_cost_std':  f"{m['cum_cost_std']:.4f}",
                })
    with open(fname, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved -> {fname}')


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('\n── Sparse suite ──')
    sparse_results = {}
    for name, env in ENVS.items():
        print(f'\n{name}')
        sparse_results[name] = run_env(env, name)

    print('\n── Dense suite ──')
    dense_results = {}
    for name, env in ENVS_DENSE.items():
        print(f'\n{name}')
        dense_results[name] = run_env(env, name)

    plot_suite(ENVS,       sparse_results, 'benchmark_sparse')
    plot_suite(ENVS_DENSE, dense_results,  'benchmark_dense')
    plot_nu(ENVS, sparse_results,          'benchmark_nu')
    save_csv(sparse_results, dense_results)