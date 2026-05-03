"""
visualization.py — reusable viz utilities for MJX environments.

HTML approach mirrors brax.io.html exactly:
  1. brax.io.json.dumps(sys, states)  → JSON string
  2. zlib compress + base64 encode    → safe to embed
  3. inject into HTML template        → self-contained file
"""

import os
import base64
import zlib
import json
import numpy as np
import mujoco
import mediapy
import jax
import jax.numpy as jnp

import brax
from brax.io import json as brax_json
from brax.io import mjcf as brax_mjcf


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_fps(env) -> int:
    try:
        return max(1, int(round(1.0 / env._config.ctrl_dt)))
    except Exception:
        try:
            return max(1, int(round(1.0 / env.dt)))
        except Exception:
            return 30


def _load_brax_sys(xml_path: str, viz_backend: str = 'spring'):
    from brax import generalized, spring, positional
    _pipeline = {
        'spring':      spring,
        'generalized': generalized,
        'positional':  positional,
    }[viz_backend]
    sys = brax_mjcf.load(xml_path)
    if viz_backend in ('spring', 'positional'):
        sys = sys.tree_replace({'opt.timestep': 0.005})
    return sys, _pipeline


def _states_to_brax(sys, pipeline, mjx_states: list) -> list:
    """Convert mjx_env.State list → brax pipeline_state list."""
    jit_init = jax.jit(lambda q, qd: pipeline.init(sys, q, qd))
    brax_states = []
    for t, s in enumerate(mjx_states):
        brax_states.append(
            jit_init(jnp.array(s.data.qpos), jnp.array(s.data.qvel))
        )
        if t % max(1, len(mjx_states) // 5) == 0:
            print(f'       frame {t}/{len(mjx_states)}')
    return brax_states


# ── HTML template (mirrors brax/visualizer/index.html structure) ──────────────

# The Brax viewer JS reads `system_json_b64` from the page and handles
# all 3-D rendering + free camera + scrubber.
_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Trajectory Visualization</title>
  <style>
    body  {{ margin: 0; background: #1a1a1a; color: #eee;
             font-family: sans-serif; display: flex;
             flex-direction: column; align-items: center; }}
    h1    {{ margin: 16px 0 4px; font-size: 1.1rem; }}
    #scene{{ width: 100%; height: {height}px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div id="scene"></div>
  <script>
    // Brax system + trajectory embedded as zlib-compressed, base64-encoded JSON.
    // The viewer JS decodes this automatically.
    const system_json_b64 = "{system_json_b64}";
  </script>
  <script src="{js_url}"></script>
</body>
</html>
"""


def _build_html(
    sys,
    brax_states: list,
    height: int = 480,
    title: str  = 'Trajectory Visualization',
    base_url: str | None = None,
) -> str:
    """
    Serialise sys+states to JSON, compress, base64-encode, inject into template.
    Mirrors brax.io.html.render_from_json exactly.
    """
    if base_url is None:
        base_url = 'https://cdn.jsdelivr.net/gh/google/brax'
    js_url = f'{base_url}@v{brax.__version__}/brax/visualizer/js/viewer.js'

    # Step 1 — serialize (same call Brax uses internally)
    json_str = brax_json.dumps(sys, brax_states)

    # Step 2 — compress + encode
    compressed   = zlib.compress(bytes(json_str, 'utf-8'))
    b64_str      = base64.b64encode(compressed).decode('ascii')

    # Step 3 — inject into template
    html = _HTML_TEMPLATE.format(
        height          = height,
        title           = title,
        system_json_b64 = b64_str,
        js_url          = js_url,
    )
    return html


# ── public API ────────────────────────────────────────────────────────────────

def save_html(
    env,
    states: list,
    path: str,
    height: int      = 480,
    viz_backend: str = 'spring',
    title: str       = 'Trajectory Visualization',
) -> None:
    """
    Save an interactive HTML file with free camera + time scrubber.

    Pipeline:
      mjx states → brax pipeline_state list
      → brax.io.json.dumps → zlib+b64 → HTML template

    Args:
        env:         MJX env with .xml_path (real file on disk)
        states:      list of mjx_env.State
        path:        output .html path
        height:      viewer height in pixels
        viz_backend: 'spring' | 'generalized' | 'positional'
        title:       page title shown above the viewer
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    print(f'[html] loading Brax system  backend={viz_backend}')
    sys, pipeline = _load_brax_sys(env.xml_path, viz_backend)

    print(f'[html] replaying {len(states)} frames ...')
    brax_states = _states_to_brax(sys, pipeline, states)

    print('[html] serialising + compressing ...')
    html_str = _build_html(sys, brax_states, height=height, title=title)

    with open(path, 'w') as f:
        f.write(html_str)
    print(f'[html] saved → {path}  ({len(html_str)//1024} KB)')


# ── mp4 ───────────────────────────────────────────────────────────────────────

def save_videos(
    env,
    states: list,
    outdir: str,
    name: str,
    fps: int | None = None,
    height: int     = 480,
    width: int      = 640,
) -> list[str]:
    """Render one mp4 per camera. Falls back to free camera if none defined."""
    os.makedirs(outdir, exist_ok=True)

    mj_model = env.mj_model
    mj_data  = mujoco.MjData(mj_model)
    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    fps      = fps or _get_fps(env)
    cam_ids  = list(range(mj_model.ncam)) if mj_model.ncam > 0 else [-1]

    print(f'[mp4]  {len(cam_ids)} camera(s)  fps={fps}')
    written = []

    for cam_id in cam_ids:
        frames = []
        for s in states:
            mj_data.qpos[:] = np.array(s.data.qpos)
            mj_data.qvel[:] = np.array(s.data.qvel)
            mujoco.mj_forward(mj_model, mj_data)
            renderer.update_scene(mj_data, camera=cam_id)
            frames.append(renderer.render())

        label = f'cam{cam_id}' if cam_id >= 0 else 'free'
        if cam_id >= 0:
            try:
                label = mj_model.camera(cam_id).name or label
            except Exception:
                pass

        path = os.path.join(outdir, f'{name}_{label}.mp4')
        mediapy.write_video(path, frames, fps=fps)
        print(f'[mp4]  saved → {path}')
        written.append(path)

    renderer.close()
    return written


# ── convenience wrapper ───────────────────────────────────────────────────────

def save_all(
    env,
    states: list,
    outdir: str       = 'videos',
    name: str         = 'rollout',
    fps: int | None   = None,
    html_height: int  = 480,
    video_height: int = 480,
    video_width: int  = 640,
    save_html_: bool  = True,
    save_video_: bool = True,
    viz_backend: str  = 'spring',
) -> dict:
    """Save HTML and/or mp4s. Returns {'html': path, 'videos': [paths]}."""
    out = {'html': None, 'videos': []}

    if save_html_:
        html_path = os.path.join(outdir, f'{name}.html')
        save_html(env, states, html_path,
                  height=html_height, viz_backend=viz_backend, title=name)
        out['html'] = html_path

    if save_video_:
        out['videos'] = save_videos(
            env, states, outdir, name,
            fps=fps, height=video_height, width=video_width,
        )

    return out