"""
planning.py — MPPI planning primitives for DNI-MPPI experiments.

Public API
----------
colored_noise_batch(key, K, N, beta)  ->  (K, N) array
make_mppi(env, K, LAM, phi_star, eta_nu, nu_max, maxdiff_alpha, maxdiff_eps)
    returns  episode(key, eps_schedule) -> (obs_traj, cost_traj, phi_traj, nu_traj)
metrics(costs)  ->  dict
"""

import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np


# ── colored noise ─────────────────────────────────────────────────────────────

def colored_noise_batch(key, K, N, beta):
    """(K, N) colored noise with PSD ∝ 1/f^beta, unit variance per row."""
    k1, k2 = random.split(key)
    nf    = N // 2 + 1
    scale = jnp.arange(1, nf + 1, dtype=jnp.float32) ** (-beta / 2.)
    r = random.normal(k1, (K, nf)) * scale
    i = random.normal(k2, (K, nf)) * scale
    cn = jnp.fft.irfft(r + 1j * i, n=N, axis=-1)
    return cn / (cn.std(axis=-1, keepdims=True) + 1e-8)


# ── MPPI factory ──────────────────────────────────────────────────────────────

def make_mppi(env, K, LAM, phi_star=0.0, eta_nu=0.0, nu_max=5.0,
              use_noise=False, nu_init=0.0, frozen=False,
              maxdiff_alpha=0.0, maxdiff_eps=1e-3):
    """
    Build a jit/vmap-able episode function.

    Parameters
    ----------
    use_noise         : inject state noise (sigma_0 baseline always active)
    nu_init           : initial (and fixed when phi_star=0) log noise multiplier
    frozen            : if True, nu is never updated (constant noise at sigma_0*exp(nu_init))
    phi_star > 0      : activate adaptive noise scheduling (DNI)
    eta_nu            : dual step-size for nu = log(sigma / sigma_0)
    nu_max            : cap on nu
    maxdiff_alpha > 0 : activate MaxDiff bonus -(alpha/2) * log det C[s, u]
                        C = sigma_u^2 * J_u J_u^T + maxdiff_eps * I
                        For scalar action (rank-1 + eps*I):
                        log det C = (n-1)*log(eps) + log(sigma_u^2*||J||^2 + eps)
                        (Berrueta et al., NMI 2024, Eq. 6)
    maxdiff_eps       : regularisation for rank-deficient C (all our envs have
                        scalar actions, making C rank-1 without regularisation)

    Returns
    -------
    episode(key, eps_schedule) -> (obs_traj, cost_traj, phi_traj, nu_traj)
      eps_schedule : (max_steps,) float  — colored-noise beta per step
                     0 = white noise, >0 = LF-MPPI
    """
    step_fn     = env['step_fn']
    cost_fn     = env['cost_fn']
    obs_fn      = env['obs_fn']
    action_clip = float(env['action_clip'])
    state_range = env['state_range']
    reset_fn    = env['reset_fn']
    sigma_u     = float(env['sigma_u'])
    sigma_0     = float(env['sigma_0'])
    N           = int(env['N'])
    max_steps   = int(env['max_steps'])
    state_dim   = int(state_range.shape[0])
    log_K       = jnp.log(float(K))
    use_ac      = bool(env.get('action_cost', False))
    use_noise_s = use_noise             # static Python flag
    use_md      = maxdiff_alpha > 0.0   # static Python flag — resolved at trace time

    def episode(key, eps_schedule):

        def scan_step(carry, beta_t):
            key, state, U, nu = carry
            key, kc, kd = random.split(key, 3)

            # dynamics noise: active whenever use_noise=True
            sigma = sigma_0 * jnp.exp(nu) if use_noise_s else 0.

            # action perturbations: lax.cond avoids executing the FFT at
            # runtime when beta_t==0, preventing cuFFT scratch-allocator
            # exhaustion across many back-to-back jit calls.
            eps = jax.lax.cond(
                beta_t > 0.,
                lambda: colored_noise_batch(kc, K, N, beta_t) * sigma_u,
                lambda: random.normal(kc, (K, N)) * sigma_u,
            )

            # state noise
            dn = random.normal(kd, (K, N, state_dim)) * (sigma * state_range)

            def rollout(eps_k, dn_k):
                def step(s, inp):
                    u_nom, e, d = inp
                    u  = jnp.clip(u_nom + e, -action_clip, action_clip)
                    sn = step_fn(s, u) + d
                    ac = jnp.where(use_ac,
                                   (LAM / (2. * sigma_u**2)) * jnp.sum(u_nom * e), 0.)
                    if use_md:
                        # J_u : (state_dim,) for scalar action
                        J       = jax.jacfwd(lambda a: step_fn(s, a))(u)
                        JJ      = jnp.dot(J, J)
                        # log det(sigma_u^2 * J J^T + eps * I) for rank-1 matrix:
                        log_det = ((state_dim - 1) * jnp.log(maxdiff_eps)
                                   + jnp.log(sigma_u**2 * JJ + maxdiff_eps))
                        md = -maxdiff_alpha / 2. * log_det   # subtract bonus from cost
                    else:
                        md = 0.
                    return sn, cost_fn(sn) + ac + md
                _, costs = jax.lax.scan(step, state, (U, eps_k, dn_k))
                return costs.sum()

            S = vmap(rollout)(eps, dn)
            w = jnp.exp(-(S - S.min()) / LAM)
            w = w / w.sum()
            U = jnp.clip(U + jnp.einsum('k,kn->n', w, eps), -action_clip, action_clip)

            # normalised weight entropy
            phi = 1. - (-jnp.sum(w * jnp.log(w + 1e-8))) / log_K

            # dual update on nu — runs whenever not frozen
            # frozen=False, phi_star=0  -> nu decreases (noise anneals away)
            # frozen=False, phi_star>0  -> nu adapts to maintain psi >= phi_star
            # frozen=True               -> nu fixed at nu_init throughout
            nu_new = jnp.where(
                not frozen,
                jnp.clip(nu + eta_nu * (phi_star - phi), -10., nu_max),
                nu,
            )

            state_new = step_fn(state, U[0])
            U_shifted = jnp.concatenate([U[1:], jnp.zeros(1)])

            return (key, state_new, U_shifted, nu_new), \
                   (obs_fn(state_new), cost_fn(state_new), phi, nu_new)

        key, kr = random.split(key)
        init = (key, reset_fn(kr), jnp.zeros(N), jnp.array(float(nu_init)))
        _, (obs, costs, phis, nus) = jax.lax.scan(scan_step, init, eps_schedule)
        return obs, costs, phis, nus

    return episode


# ── metrics ───────────────────────────────────────────────────────────────────

def metrics(costs: np.ndarray) -> dict:
    """
    costs : (n_seeds, T)  — per-step cost (< 0.5 = goal reached)
    """
    steps  = [next((i for i, c in enumerate(row) if c < 0.5), None) for row in costs]
    solved = [s for s in steps if s is not None]
    cum    = costs.sum(axis=1)
    return dict(
        success_rate  = len(solved) / len(costs),
        mean_step     = float(np.mean(solved))  if solved else np.inf,
        std_step      = float(np.std(solved))   if solved else np.inf,
        cum_cost_mean = float(cum.mean()),
        cum_cost_std  = float(cum.std()),
    )