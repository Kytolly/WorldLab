from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldlab import ConfigError, TraceRecorder, load_config, run_configured_demo
from worldlab.data import EnvironmentStepped


ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "v0.2.1" / "example.yaml"


def test_v021_yaml_uses_one_rollout_chunk_size() -> None:
    config = load_config(
        CONFIG_PATH,
        overrides=(
            "rollout.chunk_size=1",
            "runtime.step_delay_s=0",
            "observability.dashboard.enabled=false",
        ),
    )

    assert config.rollout.chunk_size == 1
    assert config.world_model.inference.action_dim == 16
    assert config.training.world_model.mode == "disabled"
    assert config.training.policy.mode == "disabled"


def test_configured_example_closed_loop_preserves_chunk_outputs() -> None:
    config = load_config(
        CONFIG_PATH,
        overrides=(
            "rollout.chunk_size=2",
            "environment.task.goal=2",
            "runtime.step_delay_s=0",
            "observability.dashboard.enabled=false",
        ),
    )
    trace = TraceRecorder(max_events=64)

    result = run_configured_demo(config, trace=trace)

    assert result.total_reward == 2.0
    assert result.length == 2
    stepped = [event for event in trace.events if isinstance(event, EnvironmentStepped)]
    assert len(stepped) == 2
    for event in stepped:
        frames = event.info["frames"]
        assert isinstance(frames, np.ndarray)
        assert frames.shape == (2, 3, 3, 32, 32)
    assert trace.diagnose().healthy is True


def test_config_rejects_duplicate_or_unsupported_granularity() -> None:
    with pytest.raises(ConfigError, match="chunk_size"):
        load_config(CONFIG_PATH, overrides=("rollout.chunk_size=0",))

    with pytest.raises(ConfigError, match="action_dim == state_dim"):
        load_config(CONFIG_PATH, overrides=("world_model.inference.state_dim=8",))

    with pytest.raises(ConfigError, match="unknown configuration key"):
        load_config(CONFIG_PATH, overrides=("world_model.typo=1",))


def test_training_modes_are_reserved_without_a_trainer() -> None:
    config = load_config(
        CONFIG_PATH,
        overrides=(
            "training.policy.mode=offline",
            "runtime.step_delay_s=0",
            "observability.dashboard.enabled=false",
        ),
    )
    with pytest.raises(NotImplementedError, match="training.policy.mode"):
        run_configured_demo(config)
