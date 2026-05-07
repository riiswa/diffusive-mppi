# Diffusive MPPI

Minimal research code for running Model Predictive Path Integral (MPPI) variants, including vanilla MPPI, low-frequency MPPI, and diffusive/adaptive state-noise MPPI, on toy control tasks, MuJoCo Playground tasks, and GCRL MuJoCo environments.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU runs, install the JAX build that matches your CUDA version before or after installing these requirements. See the JAX installation instructions for the correct wheel.

## Usage

Run the lightweight benchmark suite:

```bash
python exp_benchmark.py
```

Run a MuJoCo Playground or GCRL environment:

```bash
python mjp_mppi.py --env PointMass --method all
python mjp_mppi.py --env ant --method lf_dni
```

Run the batch experiment script:

```bash
bash run_experiments.sh --devices 0
```

Outputs are written to `figures/`, `results/`, and `logs/` depending on the script.

## Main files

- `planning.py` — core MPPI planning utilities for the toy experiments.
- `envs.py` — toy control environments and benchmark suites.
- `exp_benchmark.py` — sparse/dense benchmark runner and plots.
- `mjp_mppi.py` — MuJoCo Playground and GCRL experiment runner.
- `run_experiments.sh` — multi-environment batch runner.
- `gcrl/` — custom goal-conditioned robotics/locomotion environments and assets.
