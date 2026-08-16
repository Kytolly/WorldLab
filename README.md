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

在 Dashboard 展示实时 synthetic frames、动作序列、奖励、状态和事件时间线。验证运行时能看到帧、动作、reward、chunk 索引和模型耗时；

OpenPI fake-server 验收
先用本地 fake policy server 验证 payload、端口、动作 shape 和布局转换，再连接真实 OpenPI 服务。

增加自动化验收，验证 chunk_size=1 的 frame-level 兼容性及大于 1 的 chunk 输出。