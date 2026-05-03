"""
mjp_mppi.py — Vanilla MPPI vs DNI-MPPI on MuJoCo Playground + GCRL environments.

Convention: cost=0 at goal, cost=1 otherwise.
MPPI minimizes cumulative cost.

Usage:
    python mjp_mppi.py --env CartpoleSwingupSparse
    python mjp_mppi.py --env ant
    python mjp_mppi.py --env ant_u_maze
    python mjp_mppi.py --env arm_push_easy --method vanilla
"""

import os, sys, json, argparse
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
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

from ant          import Ant,          default_config as ant_cfg
from ant_ball     import AntBall,      default_config as ant_ball_cfg
from ant_push     import AntPush,      default_config as ant_push_cfg
from ant_maze     import AntMaze,      default_config as ant_maze_cfg
from humanoid     import Humanoid,     default_config as humanoid_cfg
from humanoid_maze import HumanoidMaze, default_config as humanoid_maze_cfg
from manipulation.arm_envs        import default_config as arm_cfg
from manipulation.arm_reach       import ArmReach
from manipulation.arm_grasp       import ArmGrasp
from manipulation.arm_push_easy   import ArmPushEasy
from manipulation.arm_push_hard   import ArmPushHard
from manipulation.arm_binpick_easy     import ArmBinpickEasy
from manipulation.arm_binpick_hard     import ArmBinpickHard
from manipulation.arm_binpick_easy_eef import ArmBinpickEasyEEF


# ── hyperparameters ───────────────────────────────────────────────────────────
K        = 500
LAM      = 0.05
SIGMA_U  = 1.0
SIGMA_0  = 0.01
N        = 100
PHI_STAR = 0.25
ETA_NU   = 0.10
NU_MAX   = 5.0
N_SEEDS  = 64

# ── MJX_ENVS: playground envs that need impl='jax' ───────────────────────────
MJX_ENVS = set()   # add playground env names here if they need jax backend


# ── env registry ──────────────────────────────────────────────────────────────
def _maze_layout(env_name: str) -> str:
    """Infer maze layout from env name."""
    if 'hardest_maze' in env_name: return 'hardest_maze'
    if 'big_maze'     in env_name: return 'big_maze'
    if 'u_maze'       in env_name: return 'u_maze'
    return 'u_maze'


def load_gcrl_env(env_name: str) -> object:
    """Instantiate a GCRL environment. Always uses impl=warp."""
    ov = {'impl': 'warp'}

    # ── locomotion ────────────────────────────────────────────────────────────
    if env_name == 'ant':
        return Ant(ant_cfg(), ov)
    if env_name == 'ant_ball':
        return AntBall(ant_ball_cfg(), ov)
    if env_name == 'ant_push':
        return AntPush(ant_push_cfg(), ov)
    if env_name == 'humanoid':
        return Humanoid(humanoid_cfg(), ov)

    # ── mazes ─────────────────────────────────────────────────────────────────
    if 'ant' in env_name and 'maze' in env_name:
        return AntMaze(maze_layout_name=_maze_layout(env_name),
                       config=ant_maze_cfg(), config_overrides=ov)
    if 'humanoid' in env_name and 'maze' in env_name:
        return HumanoidMaze(maze_layout_name=_maze_layout(env_name),
                            config=humanoid_maze_cfg(), config_overrides=ov)

    # ── manipulation ──────────────────────────────────────────────────────────
    if env_name == 'arm_reach':          return ArmReach(arm_cfg(), ov)
    if env_name == 'arm_grasp':          return ArmGrasp(arm_cfg(), ov)
    if env_name == 'arm_push_easy':      return ArmPushEasy(arm_cfg(), ov)
    if env_name == 'arm_push_hard':      return ArmPushHard(arm_cfg(), ov)
    if env_name == 'arm_binpick_easy':   return ArmBinpickEasy(arm_cfg(), ov)
    if env_name == 'arm_binpick_hard':   return ArmBinpickHard(arm_cfg(), ov)
    if env_name == 'arm_binpick_easy_eef': return ArmBinpickEasyEEF(arm_cfg(), ov)

    raise ValueError(f'Unknown GCRL env: {env_name!r}')


# Canonical set — used to route env_name to the right loader
GCRL_ENVS = {
    'ant', 'ant_ball', 'ant_push', 'humanoid',
    'ant_u_maze', 'ant_big_maze', 'ant_hardest_maze',
    'humanoid_u_maze', 'humanoid_big_maze', 'humanoid_hardest_maze',
    'arm_reach', 'arm_grasp',
    'arm_push_easy', 'arm_push_hard',
    'arm_binpick_easy', 'arm_binpick_hard', 'arm_binpick_easy_eef',
}


def load_any_env(env_name: str):
    if env_name in GCRL_ENVS:
        return load_gcrl_env(env_name)
    impl = 'jax' if env_name in MJX_ENVS else 'warp'
    return registry.load(env_name, config_overrides={'impl': impl})


def get_episode_length(env_name: str, env) -> int:
    if env_name in GCRL_ENVS:
        return env._config.episode_length
    return registry.get_default_config(env_name).episode_length


def get_fps(env) -> int:
    try:
        return max(1, int(round(1.0 / env._config.ctrl_dt)))
    except Exception:
        return max(1, int(1.0 / env.dt))


# ── cost functions ────────────────────────────────────────────────────────────
# GCRL: sparse reward ∈ {0,1} → cost = 1 - reward  (0 = goal reached)
_gcrl_cost = lambda s: 1. - s.reward

COSTS = {
    # ── playground ────────────────────────────────────────────────────────────
    'ReacherEasy':    lambda s: jnp.where(jnp.linalg.norm(s.obs[2:4]) < 0.06,  0., 1.),
    'ReacherHard':    lambda s: jnp.where(jnp.linalg.norm(s.obs[2:4]) < 0.025, 0., 1.),

    'BallInCup':      lambda s: 1. - s.reward,   # binary reward ✓

    'CheetahRun':     lambda s: jnp.where(s.reward > 0.25, 0., 1.),

    'FingerSpin':     lambda s: 1. - s.reward,   # binary: spinning fast ✓
    'FingerTurnEasy': lambda s: 1. - s.reward,   # binary: tip in target ✓
    'FingerTurnHard': lambda s: 1. - s.reward,

    'FishSwim':       lambda s: jnp.where(jnp.linalg.norm(s.obs[8:11]) < 0.1, 0., 1.),

    # obs[0]=rootz(height), obs[6]=rootx_vel(forward speed)
    'HopperStand':    lambda s: jnp.where(s.obs[0] > 0.6, 0., 1.),
    'HopperHop':      lambda s: jnp.where((s.obs[0] > 0.6) & (s.obs[6] > 2.0), 0., 1.),

    # obs[21]=head_height, obs[36]=torso zz, obs[37:39]=COM xy velocity
    'HumanoidStand':  lambda s: jnp.where(
        (s.obs[21] > 1.4) & (s.obs[36] > 0.9),
        0., 1.
    ),
    'HumanoidWalk':   lambda s: jnp.where(
        (s.obs[21] > 1.4) & (s.obs[36] > 0.9) & (jnp.linalg.norm(s.obs[37:39]) > 1.0),
        0., 1.
    ),
    'HumanoidRun':    lambda s: jnp.where(
        (s.obs[21] > 1.4) & (s.obs[36] > 0.9) & (jnp.linalg.norm(s.obs[37:39]) > 3.0),
        0., 1.
    ),

    'PointMass':      lambda s: jnp.where(jnp.linalg.norm(s.obs[:2]) < 0.05, 0., 1.),

    # torso height is NOT in obs → use reward for WalkerStand
    # obs[14]=torso zz (uprightness), obs[16]=rootx_vel (forward speed)
    'WalkerStand':    lambda s: jnp.where(s.reward > 0.5, 0., 1.),
    'WalkerWalk':     lambda s: jnp.where(
        (s.obs[14] > 0.6) & (s.obs[16] > 1.0),
        0., 1.
    ),
    'WalkerRun':      lambda s: jnp.where(
        (s.obs[14] > 0.6) & (s.obs[16] > 8.0),
        0., 1.
    ),

    # obs[5:7] = nose_to_target for n_links=6
    'SwimmerSwimmer6':  lambda s: jnp.where(
        jnp.linalg.norm(s.obs[5:7]) < 0.05, 0., 1.
    ),
    # obs[14:16] = nose_to_target for n_links=15
    'SwimmerSwimmer15': lambda s: jnp.where(
        jnp.linalg.norm(s.obs[14:16]) < 0.05, 0., 1.
    ),

    # ── gcrl (sparse reward ∈ {0,1} → cost = 1 - reward) ────────────────────
    **{name: (lambda s: 1. - s.reward) for name in GCRL_ENVS},
}


# ── progress bar ──────────────────────────────────────────────────────────────
def make_pbar(total, desc, n_seeds):
    pbar, call_count = {}, [0]
    n_update  = max(1, total // 100)
    remainder = total % n_update

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
    log_K = jnp.log(float(K))
    init_cb, update_cb = make_pbar(
        T, f'MPPI [{"DNI" if use_dni else "Vanilla"}]', N_SEEDS)
    jax.debug.callback(init_cb, 0.)

    def episode(key):
        key, kr = random.split(key)
        state = env.reset(kr)
        U     = jnp.zeros((N, na))
        nu    = jnp.array(0.)

        def mpc_step(carry, _):
            state, U, nu, key = carry
            key, keps, kdn = random.split(key, 3)

            sigma = jnp.where(use_dni, SIGMA_0 * jnp.exp(nu), 0.)
            eps   = random.normal(keps, (K, N, na)) * SIGMA_U
            dn    = random.normal(kdn,  (K, N, nq + nv)) * sigma

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
            nu  = jnp.where(use_dni,
                            jnp.clip(nu + ETA_NU * (PHI_STAR - phi), -10., NU_MAX),
                            0.)
            state = env.step(state, U[0])
            cost  = cost_fn(state)
            U     = jnp.concatenate([U[1:], jnp.zeros((1, na))])

            jax.debug.callback(update_cb, jnp.array([cost]))
            return (state, U, nu, key), (state.data.qpos, state.data.qvel, cost, phi)

        _, (qposes, qvels, costs, phis) = jax.lax.scan(
            mpc_step, (state, U, nu, key), jnp.arange(T))
        return qposes, qvels, costs, phis

    return jax.jit(vmap(episode))


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',    default='CartpoleSwingupSparse')
    parser.add_argument('--method', default='both',
                        choices=['vanilla', 'dni', 'both'])
    args = parser.parse_args()
    env_name = args.env

    METHODS = {
        'vanilla': [(False, 'vanilla')],
        'dni':     [(True,  'dni')],
        'both':    [(False, 'vanilla'), (True, 'dni')],
    }

    # ── build env ─────────────────────────────────────────────────────────────
    print(f'Loading {env_name} ...')
    env     = load_any_env(env_name)
    T       = get_episode_length(env_name, env)
    cost_fn = COSTS.get(env_name, lambda s: 1. - s.reward)
    fps     = get_fps(env)

    print(f'  T={T}  na={env.action_size}  fps={fps}')

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
                       SIGMA_0=SIGMA_0, N=N, PHI_STAR=PHI_STAR,
                       ETA_NU=ETA_NU, NU_MAX=NU_MAX, N_SEEDS=N_SEEDS), f, indent=2)

    # ── renderer ──────────────────────────────────────────────────────────────
    mj_model = env.mj_model
    mj_data  = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model)
    ncam     = mj_model.ncam
    print(f'  ncam={ncam}')

    keys = random.split(random.PRNGKey(0), N_SEEDS)

    # ── runs ──────────────────────────────────────────────────────────────────
    for use_dni, label in METHODS[args.method]:
        print(f'\n[{env_name}] {label.upper()} ...')
        run = make_run(env, cost_fn, use_dni, T)
        qposes, qvels, costs, phis = run(keys)

        qposes = np.array(qposes)   # (N_SEEDS, T, nq)
        qvels  = np.array(qvels)    # (N_SEEDS, T, nv)
        costs  = np.array(costs)
        phis   = np.array(phis)

        del run
        jax.clear_caches()

        np.save(os.path.join(outdir, f'costs_{label}.npy'), costs)
        if use_dni:
            np.save(os.path.join(outdir, 'phi_dni.npy'), phis)

        sr  = (costs.min(1) <= 0.).mean() * 100
        cum = costs.sum(1)
        print(f'  sr={sr:5.1f}%  cum_cost={cum.mean():.1f}±{cum.std():.1f}')

        best = int(costs.sum(1).argmin())
        cam_ids = range(ncam) if ncam > 0 else [-1]

        for cam_id in cam_ids:
            frames = []
            for t in range(T):
                mj_data.qpos[:] = qposes[best, t]
                mj_data.qvel[:] = qvels[best, t]
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