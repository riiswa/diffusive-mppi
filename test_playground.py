"""
mjp_mppi.py — Vanilla MPPI vs DNI-MPPI on MuJoCo Playground environments.

Convention: cost=0 at goal, cost=1 otherwise (same as toy experiments).
MPPI minimizes cumulative cost.

Usage:
    python mjp_mppi.py                    # CartpoleSwingupSparse
    python mjp_mppi.py --env WalkerStand

Results saved to results/<env_name>/
"""

import os, json, argparse
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['MUJOCO_GL'] = 'egl'

import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
import mediapy
from tqdm import tqdm
from mujoco_playground import registry

# ── hyperparameters ───────────────────────────────────────────────────────────
K        = 500
LAM      = 0.05
SIGMA_U  = 1.0
SIGMA_0  = 0.01
N        = 100
PHI_STAR = 0.25
ETA_NU   = 0.10
NU_MAX   = 5.0
N_SEEDS  = 8

# ── cost functions (0=goal, 1=not goal) ──────────────────────────────────────
# Reacher obs: [qpos(2), finger_to_target(2), qvel(2)]
COSTS = {
    'ReacherEasy': lambda s: jnp.where(jnp.linalg.norm(s.obs[2:4]) < 0.06,  0., 1.),
    'ReacherHard': lambda s: jnp.where(jnp.linalg.norm(s.obs[2:4]) < 0.025, 0., 1.),
    'BallInCup': lambda s: 1. - s.reward,
    'CheetahRun': lambda s: jnp.where(s.reward > 0.25, 0., 1.),
    'FingerSpin':     lambda s: 1. - s.reward,  # already binary
    'FingerTurnEasy': lambda s: 1. - s.reward,  # already binary
    'FingerTurnHard': lambda s: 1. - s.reward,  # already binary, smaller target
    'WalkerStand': lambda s: jnp.where(s.obs[0] > 1.2, 0., 1.),
    'WalkerWalk':  lambda s: jnp.where((s.obs[0] > 1.2) & (s.obs[1] > 1.0), 0., 1.),
    'WalkerRun':   lambda s: jnp.where((s.obs[0] > 1.2) & (s.obs[1] > 3.0), 0., 1.),
    'HopperStand': lambda s: jnp.where(s.obs[0] > 0.6, 0., 1.),
}

MJX_ENVS = ['ReacherEasy', 'ReacherHard']

# ── progress bar ──────────────────────────────────────────────────────────────

def make_pbar(total, desc, n_seeds):
    pbar       = {}
    call_count = [0]
    n_update   = max(1, total // 100)
    remainder  = total % n_update

    def _init(_):
        call_count[0] = 0
        pbar[0] = tqdm(total=total, desc=desc, ncols=90)

    def _update(args):
        c = float(args[0])
        call_count[0] += 1
        if call_count[0] % n_seeds != 0:
            return
        step = call_count[0] // n_seeds - 1
        if step % n_update == 0 and step != total - (remainder or n_update):
            pbar[0].update(n_update)
        if step == total - 1:
            pbar[0].update(remainder or n_update)
            pbar[0].close()
        pbar[0].set_postfix(cost=f'{c:.2f}')

    return _init, _update


# ── MPPI ──────────────────────────────────────────────────────────────────────

def make_run(env, cost_fn, use_dni, T):
    nq, nv, na = env.mjx_model.nq, env.mjx_model.nv, env.action_size
    log_K      = jnp.log(float(K))
    init_cb, update_cb = make_pbar(
        T, f'MPPI [{"DNI" if use_dni else "Vanilla"}]', N_SEEDS)

    jax.debug.callback(init_cb, 0.)

    def episode(key):
        key, kr = random.split(key)
        state   = env.reset(kr)
        U       = jnp.zeros((N, na))
        nu      = jnp.array(0.)

        def mpc_step(carry, step_idx):
            state, U, nu, key = carry
            key, keps, kdn = random.split(key, 3)

            sigma = jnp.where(use_dni, SIGMA_0 * jnp.exp(nu), 0.)
            eps   = random.normal(keps, (K, N, na)) * SIGMA_U
            dn    = random.normal(kdn,  (K, N, nq + nv)) * sigma

            def rollout(eps_k, dn_k):
                def step(s, inp):
                    u_nom, e, d = inp
                    u = jnp.clip(u_nom + e, -1., 1.)
                    s = env.step(s, u)
                    s = s.replace(data=s.data.replace(
                        qpos=s.data.qpos + d[:nq],
                        qvel=s.data.qvel + d[nq:],
                    ))
                    return s, cost_fn(s)
                _, costs = jax.lax.scan(step, state, (U, eps_k, dn_k))
                return costs.sum()

            G   = vmap(rollout)(eps, dn)          # (K,) total cost per rollout
            w   = jax.nn.softmax(-G / LAM)        # lower cost → higher weight
            U   = jnp.clip(U + jnp.einsum('k,kni->ni', w, eps), -1., 1.)
            phi = 1. - (-jnp.sum(w * jnp.log(w + 1e-8))) / log_K
            nu  = jnp.where(use_dni,
                            jnp.clip(nu + ETA_NU * (PHI_STAR - phi), -10., NU_MAX),
                            0.)
            state = env.step(state, U[0])
            cost  = cost_fn(state)
            U     = jnp.concatenate([U[1:], jnp.zeros((1, na))])

            jax.debug.callback(update_cb, jnp.array([cost]))
            return (state, U, nu, key), (state, cost, phi)

        _, (states, costs, phis) = jax.lax.scan(
            mpc_step, (state, U, nu, key), jnp.arange(T))
        return states, costs, phis

    return jax.jit(vmap(episode))


# ── utils ─────────────────────────────────────────────────────────────────────

def states_to_list(batched_states, seed_idx, T):
    seed_states = jax.tree_util.tree_map(
        lambda x: np.array(x[seed_idx]), batched_states)
    def safe_idx(x, t):
        return x[t] if x.ndim > 0 and x.shape[0] == T else x
    return [jax.tree_util.tree_map(lambda x: safe_idx(x, t), seed_states)
            for t in range(T)]


# ── main ──────────────────────────────────────────────────────────────────────

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='CartpoleSwingupSparse')
    args     = parser.parse_args()
    env_name = args.env

    impl = 'jax' if env_name in MJX_ENVS else 'warp'

    env     = registry.load(env_name, config_overrides={'impl': impl})
    T       = registry.get_default_config(env_name).episode_length
    cost_fn = COSTS.get(env_name, lambda s: 1. - s.reward)

    _s = jax.jit(env.step)(jax.jit(env.reset)(random.PRNGKey(0)),
                            jnp.zeros(env.action_size))

    outdir = os.path.join('results', env_name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(dict(env=env_name, K=K, LAM=LAM, SIGMA_U=SIGMA_U,
                       SIGMA_0=SIGMA_0, N=N, PHI_STAR=PHI_STAR,
                       ETA_NU=ETA_NU, NU_MAX=NU_MAX, N_SEEDS=N_SEEDS), f, indent=2)

    # ── MuJoCo renderer setup (shared across both runs) ──────────────────────
    import mujoco
    mj_model  = env.mj_model if hasattr(env, 'mj_model') else env.model
    mj_data   = mujoco.MjData(mj_model)
    renderer  = mujoco.Renderer(mj_model)
    ncam      = mj_model.ncam
    fps       = int(1. / env.dt)
    print(f'num cameras: {ncam}')

    keys = random.split(random.PRNGKey(0), N_SEEDS)

    for use_dni, label in [(False, 'vanilla'), (True, 'dni')]:
        print(f'\n[{env_name}] {label.upper()}...')
        run                  = make_run(env, cost_fn, use_dni, T)
        states, costs, phis  = run(keys)
        costs = np.array(costs)
        phis  = np.array(phis)

        np.save(os.path.join(outdir, f'costs_{label}.npy'), costs)
        if use_dni:
            np.save(os.path.join(outdir, 'phi_dni.npy'), phis)

        sr  = (costs.min(1) <= 0.).mean() * 100
        cum = costs.sum(1)
        print(f'  sr={sr:5.1f}%  cum_cost={cum.mean():.1f}±{cum.std():.1f}')

        best        = int(costs.sum(1).argmin())
        seed_states = states_to_list(states, best, T)   # list of T state structs

        for cam_id in range(ncam):
            print(f'  Rendering {label} camera {cam_id}...')
            frames = []
            for t in range(T):
                mj_data.qpos[:] = seed_states[t].data.qpos
                mj_data.qvel[:] = seed_states[t].data.qvel
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=cam_id)
                frames.append(renderer.render())
            mediapy.write_video(
                os.path.join(outdir, f'video_{label}_cam{cam_id}.mp4'),
                frames, fps=fps
            )

    print(f'\nSaved -> {outdir}/')