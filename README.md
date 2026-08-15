# WorldLab

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
