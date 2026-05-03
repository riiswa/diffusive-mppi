"""
mjp_mppi.py — Vanilla / DNI / LF-MPPI / LF-DNI on MuJoCo Playground + GCRL.

All variants run in a single parallel flat vmap (methods × seeds).

Usage:
    python mjp_mppi.py --env CartpoleSwingupSparse
    python mjp_mppi.py --env ant --method all
    python mjp_mppi.py --env arm_push_easy --method lf_dni
"""

import os, sys, json, argparse
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['MUJOCO_GL'] = 'egl'

import jax
import jax.numpy as jnp
from jax import vmap, random
import numpy as np
import mujoco
import mediapy
from tqdm import tqdm
from mujoco_playground import registry

# ── GCRL envs ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gcrl'))

from ant           import Ant,          default_config as ant_cfg
from ant_ball      import AntBall,      default_config as ant_ball_cfg
from ant_push      import AntPush,      default_config as ant_push_cfg
from ant_maze      import AntMaze,      default_config as ant_maze_cfg
from humanoid      import Humanoid,     default_config as humanoid_cfg
from humanoid_maze import HumanoidMaze, default_config as humanoid_maze_cfg
from manipulation.arm_envs             import default_config as arm_cfg
from manipulation.arm_reach            import ArmReach
from manipulation.arm_grasp            import ArmGrasp
from manipulation.arm_push_easy        import ArmPushEasy
from manipulation.arm_push_hard        import ArmPushHard
from manipulation.arm_binpick_easy     import ArmBinpickEasy
from manipulation.arm_binpick_hard     import ArmBinpickHard
from manipulation.arm_binpick_easy_eef import ArmBinpickEasyEEF


# ── hyperparameters ───────────────────────────────────────────────────────────
K        = 500
LAM      = 0.05
SIGMA_U  = 1.0
SIGMA_0  = 0.01
BETA     = 2.0      # color exponent for LF variants
N        = 100
PHI_STAR = 0.25
ETA_NU   = 0.10
NU_MAX   = 5.0
N_SEEDS  = 16

MJX_ENVS = []

# ── methods: (sigma_0, beta) ──────────────────────────────────────────────────
# vanilla   sigma_0=0      beta=0    → white noise, no state noise
# dni       sigma_0=SIGMA_0 beta=0   → white noise, adaptive state noise
# lf_mppi   sigma_0=0      beta=BETA → colored noise, no state noise
# lf_dni    sigma_0=SIGMA_0 beta=BETA → colored noise, adaptive state noise
ALL_METHODS = {
    'vanilla': (0.0,    0.0 ),
    'dni':     (SIGMA_0, 0.0),
    'lf_mppi': (0.0,    BETA),
    'lf_dni':  (SIGMA_0, BETA),
}


# ── colored noise ─────────────────────────────────────────────────────────────
def colored_noise_batch(key, K, N, na, beta):
    nf    = N // 2 + 1
    scale = jnp.arange(1, nf + 1, dtype=jnp.float32) ** (-beta / 2.)
    keys  = random.split(key, na)   # (na, 2) — one key per action dim, no reshape needed

    def _one_dim(k):
        k1, k2 = random.split(k)   # split the per-dim key into two
        r  = random.normal(k1, (K, nf)) * scale
        im = random.normal(k2, (K, nf)) * scale
        cn = jnp.fft.irfft(r + 1j * im, n=N, axis=-1)   # (K, N)
        return cn / (cn.std(axis=-1, keepdims=True) + 1e-8)

    return vmap(_one_dim)(keys).transpose(1, 2, 0)   # (K, N, na)


# ── env registry ──────────────────────────────────────────────────────────────
def _maze_layout(env_name):
    if 'hardest_maze' in env_name: return 'hardest_maze'
    if 'big_maze'     in env_name: return 'big_maze'
    return 'u_maze'


def load_gcrl_env(env_name):
    ov = {'impl': 'warp'}
    if env_name == 'ant':              return Ant(ant_cfg(), ov)
    if env_name == 'ant_ball':         return AntBall(ant_ball_cfg(), ov)
    if env_name == 'ant_push':         return AntPush(ant_push_cfg(), ov)
    if env_name == 'humanoid':         return Humanoid(humanoid_cfg(), ov)
    if 'ant'      in env_name and 'maze' in env_name:
        return AntMaze(maze_layout_name=_maze_layout(env_name),
                       config=ant_maze_cfg(), config_overrides=ov)
    if 'humanoid' in env_name and 'maze' in env_name:
        return HumanoidMaze(maze_layout_name=_maze_layout(env_name),
                            config=humanoid_maze_cfg(), config_overrides=ov)
    if env_name == 'arm_reach':            return ArmReach(arm_cfg(), ov)
    if env_name == 'arm_grasp':            return ArmGrasp(arm_cfg(), ov)
    if env_name == 'arm_push_easy':        return ArmPushEasy(arm_cfg(), ov)
    if env_name == 'arm_push_hard':        return ArmPushHard(arm_cfg(), ov)
    if env_name == 'arm_binpick_easy':     return ArmBinpickEasy(arm_cfg(), ov)
    if env_name == 'arm_binpick_hard':     return ArmBinpickHard(arm_cfg(), ov)
    if env_name == 'arm_binpick_easy_eef': return ArmBinpickEasyEEF(arm_cfg(), ov)
    raise ValueError(f'Unknown GCRL env: {env_name!r}')


GCRL_ENVS = {
    'ant', 'ant_ball', 'ant_push', 'humanoid',
    'ant_u_maze', 'ant_big_maze', 'ant_hardest_maze',
    'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze',
    'arm_reach', 'arm_grasp',
    'arm_push_easy', 'arm_push_hard',
    'arm_binpick_easy', 'arm_binpick_hard', 'arm_binpick_easy_eef',
}

NACONMAX_CONFIGS = {
    "HopperHop": 200000,
    "HopperStand": 200000,
    "WalkerStand": 150000,
    "WalkerWalk": 150000,
    "WalkerRun": 200000,
    "BallInCup": 100000,
    "HumanoidStand": 600000,
    "HumanoidWalk": 750000,
    "HumanoidRun": 750000,
}


def load_any_env(env_name):
    if env_name in GCRL_ENVS:
        return load_gcrl_env(env_name)
    config_overrides = {}
    config_overrides['impl'] = 'jax' if env_name in MJX_ENVS else 'warp'
    if env_name in NACONMAX_CONFIGS.keys():
        config_overrides['naconmax'] = NACONMAX_CONFIGS[env_name]
    if "Reacher" in env_name :
        config_overrides['njmax'] = 1
    return registry.load(
        env_name,
        config_overrides=config_overrides
    )


def get_episode_length(env_name, env):
    if env_name in GCRL_ENVS:
        return env._config.episode_length
    return registry.get_default_config(env_name).episode_length


def get_fps(env):
    try:
        return max(1, int(round(1.0 / env._config.ctrl_dt)))
    except Exception:
        return max(1, int(1.0 / env.dt))


# ── cost functions ────────────────────────────────────────────────────────────
_gcrl_cost = lambda s: 1. - s.reward

COSTS = {
    'ReacherEasy':    lambda s: jnp.where(jnp.linalg.norm(s.obs[2:4]) < 0.06,  0., 1.),
    'ReacherHard':    lambda s: jnp.where(jnp.linalg.norm(s.obs[2:4]) < 0.025, 0., 1.),
    'BallInCup':      lambda s: 1. - s.reward,
    'CheetahRun':     lambda s: jnp.where(s.reward > 0.25, 0., 1.),
    'FingerSpin':     lambda s: 1. - s.reward,
    'FingerTurnEasy': lambda s: 1. - s.reward,
    'FingerTurnHard': lambda s: 1. - s.reward,
    'FishSwim':       lambda s: jnp.where(jnp.linalg.norm(s.obs[8:11]) < 0.1, 0., 1.),
    'HopperStand': lambda s: jnp.where(s.reward > 0.5, 0., 1.),
    'HopperHop':   lambda s: jnp.where(s.reward > 0.5, 0., 1.),
    'HumanoidStand':  lambda s: jnp.where((s.obs[21] > 1.4) & (s.obs[36] > 0.9), 0., 1.),
    'HumanoidWalk':   lambda s: jnp.where((s.obs[21] > 1.4) & (s.obs[36] > 0.9) & (jnp.linalg.norm(s.obs[37:39]) > 1.0), 0., 1.),
    'HumanoidRun':    lambda s: jnp.where((s.obs[21] > 1.4) & (s.obs[36] > 0.9) & (jnp.linalg.norm(s.obs[37:39]) > 3.0), 0., 1.),
    'PointMass':      lambda s: jnp.where(jnp.linalg.norm(s.obs[:2]) < 0.05, 0., 1.),
    'WalkerStand':    lambda s: jnp.where(s.reward > 0.5, 0., 1.),
    'WalkerWalk':     lambda s: jnp.where((s.obs[14] > 0.6) & (s.obs[16] > 1.0), 0., 1.),
    'WalkerRun':      lambda s: jnp.where((s.obs[14] > 0.6) & (s.obs[16] > 8.0), 0., 1.),
    'SwimmerSwimmer6':  lambda s: jnp.where(jnp.linalg.norm(s.obs[5:7])   < 0.05, 0., 1.),
    'SwimmerSwimmer15': lambda s: jnp.where(jnp.linalg.norm(s.obs[14:16]) < 0.05, 0., 1.),
    **{name: _gcrl_cost for name in GCRL_ENVS},
}


# ── progress bar ──────────────────────────────────────────────────────────────
def make_pbar(total, desc, n_total):
    """n_total = N_METHODS * N_SEEDS — callbacks per step."""
    pbar, call_count = {}, [0]
    n_update  = max(1, total // 100)
    remainder = total % n_update

    def _init(_):
        call_count[0] = 0
        pbar[0] = tqdm(total=total, desc=desc, ncols=90)

    def _update(args):
        c = float(args[0])
        call_count[0] += 1
        if call_count[0] % n_total != 0:
            return
        step = call_count[0] // n_total - 1
        if step % n_update == 0 and step != total - (remainder or n_update):
            pbar[0].update(n_update)
        if step == total - 1:
            pbar[0].update(remainder or n_update)
            pbar[0].close()
        pbar[0].set_postfix(cost=f'{c:.2f}')

    return _init, _update


# ── MPPI ──────────────────────────────────────────────────────────────────────
def make_run(env, cost_fn, T, n_methods):
    nq, nv, na = env.mjx_model.nq, env.mjx_model.nv, env.action_size
    log_K      = jnp.log(float(K))
    N_TOTAL    = n_methods * N_SEEDS

    init_cb, update_cb = make_pbar(T, 'MPPI', N_TOTAL)
    jax.debug.callback(init_cb, 0.)

    def episode(key, sigma_0, beta):
        key, kr = random.split(key)
        state   = env.reset(kr)
        U       = jnp.zeros((N, na))
        nu      = jnp.array(0.)

        def mpc_step(carry, _):
            state, U, nu, key = carry
            key, keps, kdn = random.split(key, 3)

            # sigma_0=0 → no state noise (vanilla / lf_mppi)
            sigma = sigma_0 * jnp.exp(nu)

            # beta=0 → white noise; beta>0 → colored noise (LF variants)
            # lax.cond avoids executing the FFT when beta=0
            eps = jax.lax.cond(
                beta > 0.,
                lambda: colored_noise_batch(keps, K, N, na, beta) * SIGMA_U,
                lambda: random.normal(keps, (K, N, na)) * SIGMA_U,
            )                                                   # (K, N, na)

            dn = random.normal(kdn, (K, N, nq + nv)) * sigma   # (K, N, nq+nv)

            def rollout(eps_k, dn_k):
                def step(s, inp):
                    u_nom, e, d = inp
                    u = jnp.clip(u_nom + e, -1., 1.)
                    s = s.replace(data=s.data.replace(
                        qvel=s.data.qvel + d[nq:],
                    ))
                    s = env.step(s, u)
                    return s, cost_fn(s)
                _, costs = jax.lax.scan(step, state, (U, eps_k, dn_k))
                return costs.sum()

            G   = vmap(rollout)(eps, dn)
            w   = jax.nn.softmax(-G / LAM)
            U   = jnp.clip(U + jnp.einsum('k,kni->ni', w, eps), -1., 1.)
            phi = 1. - (-jnp.sum(w * jnp.log(w + 1e-8))) / log_K

            # nu update — zero-effect when sigma_0=0 (vanilla / lf_mppi)
            nu  = jnp.clip(nu + ETA_NU * (PHI_STAR - phi), -10., NU_MAX)

            state = env.step(state, U[0])
            cost  = cost_fn(state)
            U     = jnp.concatenate([U[1:], jnp.zeros((1, na))])

            jax.debug.callback(update_cb, jnp.array([cost]))
            return (state, U, nu, key), (state.data.qpos, state.data.qvel, cost, phi)

        _, (qposes, qvels, costs, phis) = jax.lax.scan(
            mpc_step, (state, U, nu, key), jnp.arange(T))
        return qposes, qvels, costs, phis

    return jax.jit(vmap(episode, in_axes=(0, 0, 0)))


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',    default='CartpoleSwingupSparse')
    parser.add_argument('--method', default='all',
                        choices=['vanilla', 'dni', 'lf_mppi', 'lf_dni', 'all'])
    args = parser.parse_args()
    env_name = args.env

    # select methods to run
    if args.method == 'all':
        methods = ALL_METHODS
    else:
        methods = {args.method: ALL_METHODS[args.method]}

    labels    = list(methods.keys())
    sigma_0s  = jnp.array([v[0] for v in methods.values()])
    betas     = jnp.array([v[1] for v in methods.values()])
    n_methods = len(methods)
    N_TOTAL   = n_methods * N_SEEDS

    # ── build env ─────────────────────────────────────────────────────────────
    print(f'Loading {env_name} ...')
    env     = load_any_env(env_name)
    T       = get_episode_length(env_name, env)
    cost_fn = COSTS.get(env_name, lambda s: 1. - s.reward)
    fps     = get_fps(env)
    print(f'  T={T}  na={env.action_size}  fps={fps}  '
          f'methods={labels}  N_TOTAL={N_TOTAL}')

    # warm-up JIT
    _s = jax.jit(env.step)(
        jax.jit(env.reset)(random.PRNGKey(0)),
        jnp.zeros(env.action_size),
    )

    # ── output dir + config ───────────────────────────────────────────────────
    outdir = os.path.join('results', env_name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'config.json'), 'w') as f:
        json.dump(dict(env=env_name, K=K, LAM=LAM, SIGMA_U=SIGMA_U,
                       SIGMA_0=SIGMA_0, BETA=BETA, N=N, PHI_STAR=PHI_STAR,
                       ETA_NU=ETA_NU, NU_MAX=NU_MAX, N_SEEDS=N_SEEDS,
                       methods=labels), f, indent=2)

    # ── flatten: tile keys per method, repeat params per seed ────────────────
    keys         = random.split(random.PRNGKey(0), N_SEEDS)
    keys_flat    = jnp.tile(keys, (n_methods, 1, 1)).reshape(N_TOTAL, 2)
    sigma_0_flat = jnp.repeat(sigma_0s, N_SEEDS)   # (N_TOTAL,)
    beta_flat    = jnp.repeat(betas,    N_SEEDS)   # (N_TOTAL,)

    # ── single compiled call covers all methods + seeds ───────────────────────
    print(f'\n[{env_name}] running {labels} ...')
    run = make_run(env, cost_fn, T, n_methods)
    qposes, qvels, costs, phis = run(keys_flat, sigma_0_flat, beta_flat)

    # reshape → (n_methods, N_SEEDS, T, ...)
    qposes = np.array(qposes).reshape(n_methods, N_SEEDS, T, -1)
    qvels  = np.array(qvels) .reshape(n_methods, N_SEEDS, T, -1)
    costs  = np.array(costs) .reshape(n_methods, N_SEEDS, T)
    phis   = np.array(phis)  .reshape(n_methods, N_SEEDS, T)

    # ── renderer ──────────────────────────────────────────────────────────────
    mj_model = env.mj_model
    mj_data  = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)
    cam_ids  = list(range(mj_model.ncam)) if mj_model.ncam > 0 else [-1]

    # ── save per method ───────────────────────────────────────────────────────
    for i, label in enumerate(labels):
        c   = costs[i]                          # (N_SEEDS, T)
        sr  = (c.min(1) <= 0.).mean() * 100
        cum = c.sum(1)
        print(f'  [{label}]  sr={sr:5.1f}%  '
              f'cum_cost={cum.mean():.1f}±{cum.std():.1f}')

        np.save(os.path.join(outdir, f'costs_{label}.npy'), c)
        np.save(os.path.join(outdir, f'phi_{label}.npy'),   phis[i])

        best = int(c.sum(1).argmin())
        for cam_id in cam_ids:
            frames = []
            for t in range(T):
                mj_data.qpos[:] = qposes[i, best, t]
                mj_data.qvel[:] = qvels [i, best, t]
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=cam_id)
                frames.append(renderer.render())
            cam_label = f'cam{cam_id}' if cam_id >= 0 else 'free'
            mediapy.write_video(
                os.path.join(outdir, f'video_{label}_{cam_label}.mp4'),
                frames, fps=fps,
            )

    renderer.close()
    print(f'\nSaved → {outdir}/')