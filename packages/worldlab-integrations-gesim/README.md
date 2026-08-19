# worldlab-integrations-gesim

Optional GE-Sim World Judge support for WorldLab.

The package is deliberately split into two source namespaces:

- `src/gesim/` contains the vendored GE-Sim author source used by the
  simulator, reward, and policy adapters.
- `src/worldlab_gesim/` contains only WorldLab-facing contracts, adapters, and
  composition helpers.

The package does not depend on or provide `worldlab-transport`. The author's
`gesim.client.transport.WorldModelClient` remains available because it is part
of the GE-Sim source compatibility surface; it is passed into a WorldLab
adapter and is not a WorldLab core service.

The integration keeps GE-Sim semantics outside the WorldLab core package while
preserving the author's runtime boundaries. The World Judge is local: it owns
the VLM processor, frozen vision model, trainable language/checkpoint weights,
and per-frame success head.

```text
WorldJudge (RewardClient)
    -> GESimRewardProvider (WorldLab RewardProvider)
        -> GESimRewardAdapter
            -> worldlab.RewardResult(value)

WorldModel / WorldModelClient
    -> GESimSimulatorAdapter (Simulator)
        -> WorldEnvironment
            -> GESimTask (ComposableTask)
```

`WorldJudge` is not a remote VLM API client. It implements the local
author-compatible `evaluate(head_frames, task: str)` method. The Qwen model
directory is expected at:

```text
model_zoo/qwen/Qwen2.5-VL-3B-Instruct
```

The base model alone is not a trained GE-Sim Judge. A trained success-head
checkpoint must also be supplied to `Qwen25VLWorldJudge`.

```python
from worldlab_integrations_gesim import (
    GESimRewardAdapter,
    GESimTask,
    Qwen25VLWorldJudge,
)

judge = Qwen25VLWorldJudge(
    model_path="model_zoo/qwen/Qwen2.5-VL-3B-Instruct",
    success_head_path="model_zoo/qwen/world_judge_head.pt",
)
task = GESimTask(
    instruction="open the drawer",
    observation=observation_provider,
    termination=termination_provider,
    reward=GESimRewardProvider(
        reward_client=judge,
        adapter=GESimRewardAdapter(policy="last"),
    ),
)
```

`GESimTask` receives already-created signal providers. It does not create a
Judge or own a task instance inside the reward provider. For construction of
the full object graph use `make_gesim_task`, `make_gesim_environment`, and
`make_gesim_loop` from `worldlab_gesim.factory`.

The author reward result retains both per-frame `success` and `progress`; the
paper Judge does not predict progress, so the local implementation marks that
field unavailable rather than fabricating values. WorldLab's scalar task
contract uses the final success value for the current fixed-length chunk and
keeps the success curve in reward diagnostics.

Install local Judge dependencies with:

```text
pip install "worldlab-integrations-gesim[judge]"
```

The Qwen base model can be downloaded with:

```text
hf download Qwen/Qwen2.5-VL-3B-Instruct \
    --local-dir model_zoo/qwen/Qwen2.5-VL-3B-Instruct
```

## Reference main program

`src/worldlab_gesim/main.py` contains a complete composition-root example.
It uses the author's `WorldModelClient` as a backend, constructs the shared
`GESimSimulatorAdapter`, injects the reward provider into `GESimTask`, and
creates `PolicyAgent` plus `EnvironmentLoop` through `factory.py`.

```python
loop = make_gesim_loop(env=env, policy=policy)
result = loop.run_episode(
    options={"episode": bundle_dir, "conditioning": "action"}
)
```

The adapter places generated frames under
`worldlab.simulation.frames`. `GESimRewardProvider` reads those frames,
calls `WorldJudge.evaluate(head_frames, task)`, and
`GESimRewardAdapter` converts the last per-frame success probability into the
core scalar `worldlab.data.RewardResult`. The original author-style
`WorldModelEnv` and `WorldModelClient` remain available in their original
modules; the WorldLab adapter does not replace them.
