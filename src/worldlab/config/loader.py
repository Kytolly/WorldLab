"""Load, merge, resolve, and validate the public WorldLab configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, cast

from omegaconf import DictConfig, ListConfig, OmegaConf


class ConfigError(ValueError):
    """Raised when a resolved configuration violates a public contract."""


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "run": {"name": "worldlab_demo", "mode": "demo", "seed": 0, "episodes": 1},
    "rollout": {"chunk_size": 4},
    "world_model": {
        "implementation": "example",
        "inference": {
            "action_dim": 16,
            "state_dim": 16,
            "frame": {"channels": 3, "num_views": 3, "height": 32, "width": 32},
            "noise_scale": 0.01,
        },
    },
    "simulator": {"implementation": "world_model"},
    "environment": {
        "implementation": "simulator_environment",
        "task": {"implementation": "chunk_goal", "goal": 4, "max_episode_steps": 8},
    },
    "policy": {"implementation": "constant", "inference": {"action_value": 0.0}},
    "agent": {"implementation": "policy_agent"},
    "runtime": {
        "training": False,
        "deterministic": True,
        "render": False,
        "validate_spaces": True,
        "safety_max_steps": 16,
        "step_delay_s": 0.0,
    },
    "training": {
        "world_model": {"mode": "disabled", "dataset": None, "checkpoint": None},
        "policy": {"mode": "disabled", "dataset": None, "checkpoint": None},
    },
    "observability": {
        "trace": {"enabled": True, "max_events": 4096},
        "dashboard": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8765,
            "poll_interval_s": 1.0,
            "keep_alive_s": 30.0,
        },
    },
    "artifacts": {"root_uri": None},
}


def load_config(
    path: Optional[str | Path] = None,
    *,
    overrides: Iterable[str] = (),
) -> DictConfig:
    """Load a YAML config, apply OmegaConf dotlist overrides, and validate it."""

    base = OmegaConf.create(DEFAULT_CONFIG)
    loaded: DictConfig
    if path is None:
        loaded = OmegaConf.create({})
    else:
        candidate = Path(path)
        if not candidate.is_file():
            raise ConfigError(f"configuration file does not exist: {candidate}")
        loaded = _load_yaml_layers(candidate)

    override_list = list(overrides)
    override_config = OmegaConf.from_dotlist(override_list)
    merged = OmegaConf.merge(base, loaded, override_config)
    if not isinstance(merged, DictConfig):
        raise ConfigError("configuration root must be a YAML mapping")
    config = merged
    OmegaConf.resolve(config)
    _validate_keys(config, DEFAULT_CONFIG)
    _validate(config)
    OmegaConf.set_struct(config, True)
    return config


def config_to_dict(config: Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    """Convert a resolved config into ordinary Python containers at boundaries."""

    source: Any = (
        config
        if isinstance(config, DictConfig)
        else OmegaConf.create(cast(Any, config))
    )
    value = OmegaConf.to_container(source, resolve=True, throw_on_missing=True)
    if not isinstance(value, dict):
        raise ConfigError("resolved configuration root must be a mapping")
    return cast(dict[str, Any], value)


def _validate(config: DictConfig) -> None:
    if int(config.schema_version) != 1:
        raise ConfigError("schema_version must be 1")
    if str(config.run.mode) not in {"demo", "rollout", "train_world_model", "train_policy"}:
        raise ConfigError("run.mode must be demo, rollout, train_world_model, or train_policy")
    _positive_int(config.run.episodes, "run.episodes")
    _positive_int(config.rollout.chunk_size, "rollout.chunk_size")

    if str(config.world_model.implementation) != "example":
        raise ConfigError("v0.2.1 supports only world_model.implementation=example")
    if str(config.simulator.implementation) != "world_model":
        raise ConfigError("simulator.implementation must be world_model")
    if str(config.environment.implementation) != "simulator_environment":
        raise ConfigError("environment.implementation must be simulator_environment")
    if str(config.policy.implementation) != "constant":
        raise ConfigError("v0.2.1 supports only policy.implementation=constant")
    if str(config.agent.implementation) != "policy_agent":
        raise ConfigError("agent.implementation must be policy_agent")
    if float(config.runtime.step_delay_s) < 0.0:
        raise ConfigError("runtime.step_delay_s must be non-negative")
    safety_max_steps = config.runtime.safety_max_steps
    if safety_max_steps is not None:
        _positive_int(safety_max_steps, "runtime.safety_max_steps")

    inference = config.world_model.inference
    _positive_int(inference.action_dim, "world_model.inference.action_dim")
    _positive_int(inference.state_dim, "world_model.inference.state_dim")
    if int(inference.action_dim) != int(inference.state_dim):
        raise ConfigError("ExampleWorldModel currently requires action_dim == state_dim")
    _positive_int(inference.frame.channels, "world_model.inference.frame.channels")
    if int(inference.frame.channels) != 3:
        raise ConfigError("ExampleWorldModel currently supports RGB channels=3 only")
    for name in ("num_views", "height", "width"):
        _positive_int(inference.frame[name], f"world_model.inference.frame.{name}")
    if float(inference.noise_scale) < 0.0:
        raise ConfigError("world_model.inference.noise_scale must be non-negative")

    _positive_int(config.environment.task.goal, "environment.task.goal")
    max_episode_steps = config.environment.task.max_episode_steps
    if max_episode_steps is not None:
        _positive_int(max_episode_steps, "environment.task.max_episode_steps")
    for name in ("world_model", "policy"):
        mode = str(config.training[name].mode)
        if mode not in {"disabled", "offline", "online"}:
            raise ConfigError(f"training.{name}.mode must be disabled, offline, or online")

    _positive_int(config.observability.trace.max_events, "observability.trace.max_events")
    dashboard = config.observability.dashboard
    if not str(dashboard.host):
        raise ConfigError("observability.dashboard.host must not be empty")
    if int(dashboard.port) < 0 or int(dashboard.port) > 65535:
        raise ConfigError("observability.dashboard.port must be between 0 and 65535")
    if float(dashboard.poll_interval_s) <= 0.0:
        raise ConfigError("observability.dashboard.poll_interval_s must be positive")
    if float(dashboard.keep_alive_s) < 0.0:
        raise ConfigError("observability.dashboard.keep_alive_s must be non-negative")
    if bool(dashboard.enabled) and not bool(config.observability.trace.enabled):
        raise ConfigError("dashboard requires observability.trace.enabled=true")


def _load_yaml_layers(path: Path) -> DictConfig:
    loaded_value = OmegaConf.load(path)
    if not isinstance(loaded_value, DictConfig):
        raise ConfigError("configuration root must be a YAML mapping")
    defaults = loaded_value.pop("defaults", [])
    if defaults is None:
        defaults = []
    if not isinstance(defaults, (list, ListConfig)):
        raise ConfigError(f"defaults in {path} must be a YAML list")
    merged: DictConfig = OmegaConf.create({})
    for default in defaults:
        if not isinstance(default, str):
            raise ConfigError(f"defaults in {path} must contain file names")
        default_path = (path.parent / default).resolve()
        if not default_path.is_file():
            raise ConfigError(f"default configuration file does not exist: {default_path}")
        merged_value = OmegaConf.merge(merged, _load_yaml_layers(default_path))
        if not isinstance(merged_value, DictConfig):
            raise ConfigError("configuration root must be a YAML mapping")
        merged = merged_value
    result = OmegaConf.merge(merged, loaded_value)
    if not isinstance(result, DictConfig):
        raise ConfigError("configuration root must be a YAML mapping")
    return result


def _validate_keys(config: DictConfig, template: Mapping[str, Any], prefix: str = "") -> None:
    for key in config.keys():
        name = str(key)
        if name not in template:
            location = f"{prefix}.{name}" if prefix else name
            raise ConfigError(f"unknown configuration key: {location}")
        expected = template[name]
        value = config[key]
        if isinstance(expected, dict) and isinstance(value, DictConfig):
            child_prefix = f"{prefix}.{name}" if prefix else name
            _validate_keys(value, expected, child_prefix)


def _positive_int(value: Any, name: str) -> None:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ConfigError(f"{name} must be greater than zero")
