# WorldLab

## Create the development environment

The repository provides a reproducible Conda environment named
`env_worldlab`. It includes WorldLab, the optional Panel UI, the OpenPI client
integration, and development checks:

```powershell
conda env create -f environment.yml
conda activate env_worldlab
```

Register the local OpenPI source without installing its full GPU serving
dependency set:

```powershell
python -m pip install --no-deps -e third-party/openpi
```

If the environment already exists, update it with:

```powershell
conda env update -f environment.yml --prune
```

The base environment is intended for the synthetic demo, UI, OpenPI client,
and integration tests. The full GPU OpenPI server has its own large
JAX/PyTorch dependency set and should be installed separately in the same
Python 3.11 environment only when real model inference is needed.

To run the local UI and OpenPI acceptance flow:

```powershell
conda activate env_worldlab
python examples/openpi_panel_smoke.py --host 127.0.0.1 --port 5018
```

Open `http://127.0.0.1:5018` in a browser. The page should reach
`completed` and display the synthetic frame, action/state matrices, reward,
chunk index, model latency, and event timeline.

To use the released GE pi05 policy, start the real service in its dedicated
Linux/OpenPI environment:

```bash
PORT=8000 bash scripts/serve_policy_pi05.sh
```

Then start the WorldLab UI client in `env_worldlab` with
`--policy-url ws://127.0.0.1:8000`. The service launcher checks that the GE
The script uses the local OpenPI repository and checkpoint paths by default.
Override them with `OPENPI_ROOT` and `OPENPI_CKPT` when running from another
checkout; the checkpoint must contain `model.safetensors` and
`assets/gesim/norm_stats.json`.

## Observe the closed loop

Run the dependency-free demo with a phase-by-phase runtime trace:

```text
python -m worldlab --goal 3 --trace
```

The trace records episode reset, policy actions, environment steps, committed
transitions, episode completion, per-phase durations, and runtime failures. It
ends with an ordering and completeness diagnosis.

Programmatic use keeps the raw typed events available for inspection:

```python
from worldlab import EnvironmentLoop, TraceRecorder, build_demo

env, agent = build_demo(goal=3)
trace = TraceRecorder()

with EnvironmentLoop(env, agent, trace=trace) as loop:
    result = loop.run_episode(seed=0)

print(trace.format_timeline())
assert trace.diagnose().healthy
```

If reset, policy inference, environment stepping, agent observation, rendering,
or a callback fails, WorldLab records a `RuntimeErrorEvent` with the exact
runtime phase and traceback before re-raising the original exception.


