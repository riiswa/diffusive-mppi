"""
test_env.py — generic random-rollout smoke-test + visualisation for any
              MJX-Warp environment.

Usage:
    python test_env.py --env ant
    python test_env.py --env ant_maze --maze u_maze
    python test_env.py --env humanoid --steps 300
    python test_env.py --env arm_reach --steps 150
"""

import os, argparse, importlib
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['MUJOCO_GL'] = 'egl'

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
import mediapy

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--env',    default='arm_binpick_easy',
                    help='env name: ant | ant_maze | ant_ball | ant_push | '
                         'humanoid | humanoid_maze | '
                         'arm_reach | arm_grasp | '
                         'arm_push_easy | arm_push_hard | '
                         'arm_binpick_easy | arm_binpick_hard | '
                         'arm_binpick_easy_eef')
parser.add_argument('--steps',  type=int,   default=200)
parser.add_argument('--seed',   type=int,   default=0)
parser.add_argument('--impl',   default='warp', choices=['warp', 'jax'],
                    help='MJX backend')
parser.add_argument('--outdir', default='videos',
                    help='directory for output videos')
# env-specific extras
parser.add_argument('--maze',   default='u_maze',
                    help='maze layout for ant_maze / humanoid_maze')
parser.add_argument('--goal_distance', type=float, default=10.0,
                    help='goal distance for ant')
args = parser.parse_args()


# ── env registry ──────────────────────────────────────────────────────────────
def load_env(name: str, impl: str):
    """
    Instantiate the requested environment.
    Add new entries here as you port more envs.
    """
    overrides = {'impl': impl}

    if name == 'ant':
        from ant import Ant, default_config
        return Ant(default_config(), overrides, goal_distance=args.goal_distance)

    if name == 'ant_maze':
        from ant_maze import AntMaze, default_config
        return AntMaze(maze_layout_name=args.maze,
                       config=default_config(), config_overrides=overrides)

    if name == 'ant_ball':
        from ant_ball import AntBall, default_config
        return AntBall(default_config(), overrides)

    if name == 'ant_push':
        from ant_push import AntPush, default_config
        return AntPush(default_config(), overrides)

    if name == 'humanoid':
        from humanoid import Humanoid, default_config
        return Humanoid(default_config(), overrides)

    if name == 'humanoid_maze':
        from humanoid_maze import HumanoidMaze, default_config
        return HumanoidMaze(maze_layout_name=args.maze,
                            config=default_config(), config_overrides=overrides)

    if name == 'arm_reach':
        from manipulation.arm_reach import ArmReach
        from manipulation.arm_envs import default_config
        return ArmReach(default_config(), overrides)

    if name == 'arm_grasp':
        from manipulation.arm_grasp import ArmGrasp
        from manipulation.arm_envs import default_config
        return ArmGrasp(default_config(), overrides)

    if name == 'arm_push_easy':
        from manipulation.arm_push_easy import ArmPushEasy
        from manipulation.arm_envs import default_config
        return ArmPushEasy(default_config(), overrides)

    if name == 'arm_push_hard':
        from manipulation.arm_push_hard import ArmPushHard
        from manipulation.arm_envs import default_config
        return ArmPushHard(default_config(), overrides)

    if name == 'arm_binpick_easy':
        from manipulation.arm_binpick_easy import ArmBinpickEasy
        from manipulation.arm_envs import default_config
        return ArmBinpickEasy(default_config(), overrides)

    if name == 'arm_binpick_hard':
        from manipulation.arm_binpick_hard import ArmBinpickHard
        from manipulation.arm_envs import default_config
        return ArmBinpickHard(default_config(), overrides)

    if name == 'arm_binpick_easy_eef':
        from manipulation.arm_binpick_easy_eef import ArmBinpickEasyEEF
        from manipulation.arm_envs import default_config
        return ArmBinpickEasyEEF(default_config(), overrides)

    raise ValueError(f'Unknown env: {name!r}')


# ── helpers ───────────────────────────────────────────────────────────────────
def print_state(state, step: int) -> None:
    """Print a compact one-liner for the current state."""
    obs   = state.obs
    r     = float(state.reward)
    done  = float(state.done)
    print(f'  step {step:4d}  obs[0:4]={np.array(obs[:4]).round(3)}  '
          f'reward={r:.3f}  done={done:.0f}')


def render_all_cameras(
    mj_model: mujoco.MjModel,
    qposes: np.ndarray,      # (T, nq)
    qvels:  np.ndarray,      # (T, nv)
    fps:    int,
    outdir: str,
    env_name: str,
) -> None:
    """Render one video per camera and save to outdir."""
    mj_data  = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    ncam     = mj_model.ncam

    # Always include the free/default camera (-1) plus every named camera.
    camera_ids = list(range(ncam)) if ncam > 0 else []
    if ncam == 0:
        # No named cameras — use the free camera only.
        camera_ids = [-1]

    print(f'\nRendering {len(camera_ids)} camera(s) × {len(qposes)} frames ...')

    for cam_id in camera_ids:
        frames = []
        for t in range(len(qposes)):
            mj_data.qpos[:] = qposes[t]
            mj_data.qvel[:] = qvels[t]
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam_id)
            frames.append(renderer.render())

        cam_label = f'cam{cam_id}' if cam_id >= 0 else 'free'
        # Try to get the camera name if available.
        if cam_id >= 0:
            try:
                cam_label = mj_model.camera(cam_id).name or cam_label
            except Exception:
                pass

        path = os.path.join(outdir, f'{env_name}_{cam_label}.mp4')
        mediapy.write_video(path, frames, fps=fps)
        print(f'  saved → {path}')

    renderer.close()


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(args.outdir, exist_ok=True)

    # ── build & JIT ───────────────────────────────────────────────────────────
    print(f'\nLoading env={args.env!r}  impl={args.impl!r} ...')
    env = load_env(args.env, args.impl)

    reset_fn = jax.jit(env.reset)
    step_fn  = jax.jit(env.step)

    mj_model = env.mj_model
    print(f'  nq={mj_model.nq}  nv={mj_model.nv}  '
          f'nu={env.action_size}  ncam={mj_model.ncam}')

    # ── reset ─────────────────────────────────────────────────────────────────
    key = jax.random.PRNGKey(args.seed)
    key, kr = jax.random.split(key)

    print('Compiling reset ...')
    state = reset_fn(kr)
    print(f'  obs shape : {state.obs.shape}')
    print_state(state, 0)

    # ── rollout ───────────────────────────────────────────────────────────────
    print(f'\nRunning {args.steps} random steps ...')
    qposes, qvels = [], []

    for t in range(args.steps):
        key, ka = jax.random.split(key)
        action  = jax.random.uniform(
            ka, (env.action_size,), minval=-1., maxval=1.
        )
        state = step_fn(state, action)
        qposes.append(np.array(state.data.qpos))
        qvels .append(np.array(state.data.qvel))

        if t == 0:
            # Second call triggers the real JIT compile; note it separately.
            print('  (step JIT compiled on first call)')

        if t % max(1, args.steps // 5) == 0:
            print_state(state, t)

    qposes = np.array(qposes)  # (T, nq)
    qvels  = np.array(qvels)   # (T, nv)

    # ── fps from ctrl_dt ──────────────────────────────────────────────────────
    ctrl_dt = getattr(env, '_config', None)
    fps = 30  # safe default
    if ctrl_dt is not None and hasattr(ctrl_dt, 'ctrl_dt'):
        fps = max(1, int(round(1.0 / ctrl_dt.ctrl_dt)))
    elif hasattr(env, 'dt'):
        fps = max(1, int(round(1.0 / env.dt)))
    print(f'\nfps={fps}')

    # ── render & save ─────────────────────────────────────────────────────────
    render_all_cameras(
        mj_model  = mj_model,
        qposes    = qposes,
        qvels     = qvels,
        fps       = fps,
        outdir    = args.outdir,
        env_name  = args.env,
    )

    print('\nDone.')