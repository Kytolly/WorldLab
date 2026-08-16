"""WorldLab pi05 serving transforms for the GE-compatible checkpoint.

This module owns only the OpenPI model configuration and observation/action
transforms. It does not import GE-Sim or WorldLab RL runtime components.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from openpi import transforms as _transforms
from openpi.models import model as _model  # noqa: F401 - registers model types
from openpi.models import pi0_config
from openpi.training import config as _config


DEFAULT_PROMPT = "Pick up the kettle on the table with right arm and pour the water into the cup."

HEAD_KEY = "observation.images.head"
LEFT_KEY = "observation.images.hand_left"
RIGHT_KEY = "observation.images.hand_right"
STATE_KEY = "observation.state"

REAL_DIM = 16
MODEL_DIM = 32


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce one camera image to contiguous HWC RGB uint8."""

    array = np.asarray(img)
    if array.ndim == 3 and array.shape[0] == 3 and array.shape[2] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _square_resize_224(img: np.ndarray) -> np.ndarray:
    """Resize to the checkpoint's training resolution."""

    from PIL import Image

    return np.asarray(
        Image.fromarray(img, mode="RGB").resize((224, 224), Image.BILINEAR),
        dtype=np.uint8,
    )


@dataclasses.dataclass(frozen=True)
class WorldLabInputs(_transforms.DataTransformFn):
    """Map WorldLab's flat observation payload to pi05 model inputs."""

    def __call__(self, data: dict) -> dict:
        images = {
            "base_0_rgb": _square_resize_224(_to_hwc_uint8(data[HEAD_KEY])),
            "left_wrist_0_rgb": _square_resize_224(_to_hwc_uint8(data[LEFT_KEY])),
            "right_wrist_0_rgb": _square_resize_224(_to_hwc_uint8(data[RIGHT_KEY])),
        }
        image_mask = {key: np.True_ for key in images}

        state = np.asarray(data[STATE_KEY], dtype=np.float32).reshape(-1)
        state = _transforms.pad_to_dim(state, MODEL_DIM)
        state[REAL_DIM:] = 0.0

        inputs: dict = {"image": images, "image_mask": image_mask, "state": state}
        prompt = data.get("prompt")
        if prompt is not None:
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else str(prompt)
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        return inputs


@dataclasses.dataclass(frozen=True)
class WorldLabOutputs(_transforms.DataTransformFn):
    """Return WorldLab's [L7, L_grip, R7, R_grip] action layout."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])[:, :REAL_DIM]
        output = np.empty_like(actions)
        output[:, 0:7] = actions[:, 0:7]
        output[:, 7] = actions[:, 14]
        output[:, 8:15] = actions[:, 7:14]
        output[:, 15] = actions[:, 15]
        return {"actions": output}


def make_config(
    asset_id: str = "gesim",
    action_horizon: int = 50,
    compile_mode: str | None = None,
) -> _config.TrainConfig:
    """Build the in-process OpenPI TrainConfig for the checkpoint."""

    return _config.TrainConfig(
        name="pi05_worldlab",
        exp_name="serve",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=MODEL_DIM,
            action_horizon=action_horizon,
            pytorch_compile_mode=compile_mode,
        ),
        data=_config.SimpleDataConfig(
            assets=_config.AssetsConfig(asset_id=asset_id),
            data_transforms=lambda model: _transforms.Group(
                inputs=[WorldLabInputs()],
                outputs=[WorldLabOutputs()],
            ),
        ),
    )
