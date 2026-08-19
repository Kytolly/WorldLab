# worldlab-integrations-openpi

Optional OpenPI policy integration for WorldLab. The package implements the
same msgpack-numpy WebSocket shape as the official OpenPI remote policy
client. `OpenPIPolicy` delegates connection and inference to
`gesim.policies.openpi.OpenPIPolicy`; WorldLab only exposes the shared world
model observation contract and validates the returned action chunk. The
GE-Sim backend owns its OpenPI payload and action-layout conversion.

The package keeps the author-compatible implementation in `src/gesim/` and
the WorldLab adapter in `src/worldlab_openpi/`. The adapter owns the boundary
between the two contracts; it does not introduce a second WebSocket client or
depend on the removed `worldlab-transport` package.

Install it with:

```text
pip install "worldlab[openpi]"
```

For the repository-pinned OpenPI source and serving scripts:

```text
git submodule update --init --recursive
pip install -e third-party/openpi/packages/openpi-client
```

The submodule is not imported by the WorldLab core package. It is added to
`PYTHONPATH` by the WorldLab serving script when loading the local OpenPI
implementation.

The adapter accepts the shared `gesim.types.Observation` produced by the
GE-Sim world model and returns a WorldLab `PolicyOutput` whose action shape is
`(horizon, 16)`. The fixed target layout is the GE-style world-model layout:

```text
[L7_arm, L_grip, R7_arm, R_grip]
```

For tests or an in-process policy, pass an already-created GE-Sim policy
backend with `OpenPIPolicy(backend=...)`. For a real serving process, pass the
WebSocket URL and the adapter constructs
`gesim.policies.openpi.OpenPIPolicy` as its backend.

```text
GESim world model
    -> gesim.types.Observation
        -> worldlab_openpi.OpenPIPolicy
            -> GE-Sim/OpenPI backend
                -> (T, 16) action chunk
```

The deterministic fake server is kept under the package test module as a
protocol fixture; it is not part of the runtime API. Run the integration tests
before connecting to a real OpenPI service.

For the released pi05 policy, WorldLab provides a GE-style bash launcher that
starts the bundled `worldlab_openpi.openpi_serving` service:

```text
PORT=8000 bash scripts/serve_policy_pi05.sh
```

Run it in the dedicated OpenPI serving environment. The default checkpoint is
`model_zoo/agibot-world/Genie-Envisioner-Sim-v2.0/checkpoints/pi05_gesim_g01op_test`;
override it with `OPENPI_CKPT`. The client connects to the same port with
`OpenPIPolicy("ws://127.0.0.1:8000")`. Use `OPENPI_ROOT` when the OpenPI
checkout is elsewhere.
