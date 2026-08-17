from __future__ import annotations

import numpy as np
import pytest
from typing import Any, cast
from numpy.typing import NDArray

from worldlab import ExampleWorldModel, WorldModel
from worldlab.data import WorldModelContext, WorldModelStepResult


def _options(model: ExampleWorldModel) -> dict[str, object]:
    _, views, height, width = model.frame_shape
    intrinsic = np.repeat(np.eye(3, dtype=np.float32)[None], views, axis=0)
    extrinsic = np.repeat(np.eye(4, dtype=np.float32)[None], views, axis=0)
    first_frame = np.zeros(model.frame_shape, dtype=np.float32)
    trajectory = np.zeros(
        (3, views, model.chunk_size, height, width),
        dtype=np.float32,
    )
    c2w = np.repeat(
        np.eye(4, dtype=np.float32)[None, None],
        views * model.chunk_size,
        axis=0,
    ).reshape(views, model.chunk_size, 4, 4)
    return {
        "intrinsic": intrinsic,
        "extrinsic": extrinsic,
        "first_frame": first_frame,
        "trajectory": trajectory,
        "c2w": c2w,
        "task": "synthetic task",
    }


def _run(
    model: ExampleWorldModel,
    actions: NDArray[np.float32],
) -> WorldModelStepResult[int, NDArray[Any]]:
    context = model.initialize(seed=model.seed, options=_options(model))
    assert isinstance(context, WorldModelContext)
    return model.sample_step(context.context, actions)


def test_example_world_model_uses_generic_world_model_contract() -> None:
    model = ExampleWorldModel(
        chunk_size=4,
        num_views=3,
        frame_height=6,
        frame_width=8,
        noise_scale=0.0,
    )

    assert isinstance(model, WorldModel)
    assert model.frame_shape == (3, 3, 6, 8)
    actions = np.ones((4, 16), dtype=np.float32)
    result = _run(model, actions)

    assert result.output.shape == (4, 3, 3, 6, 8)
    assert result.output.dtype == np.float32
    assert np.all((result.output >= 0.0) & (result.output <= 1.0))
    assert result.state is not None
    assert result.state.shape == (4, 16)
    assert np.array_equal(result.state, actions)
    assert result.info["output_length"] == 4
    assert model.chunk_index == 1


def test_example_world_model_is_repeatable_for_a_seed() -> None:
    def run() -> tuple[NDArray[Any], NDArray[Any]]:
        model = ExampleWorldModel(
            chunk_size=3,
            frame_height=4,
            frame_width=5,
            seed=17,
        )
        output = _run(model, np.arange(48, dtype=np.float32).reshape(3, 16))
        return (
            cast(NDArray[Any], output.output),
            output.state,
        )

    frames_a, state_a = run()
    frames_b, state_b = run()

    assert np.array_equal(frames_a, frames_b)
    assert np.array_equal(state_a, state_b)


def test_example_world_model_is_action_conditioned() -> None:
    def run(action: float) -> NDArray[Any]:
        model = ExampleWorldModel(
            chunk_size=2,
            frame_height=4,
            frame_width=4,
            noise_scale=0.0,
        )
        return cast(
            NDArray[Any],
            _run(model, np.full((2, 16), action, dtype=np.float32)).output,
        )

    assert not np.array_equal(run(0.0), run(1.0))


def test_example_world_model_requires_generic_initial_conditions() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        ExampleWorldModel.from_config({})

    model = ExampleWorldModel(chunk_size=2)
    with pytest.raises(ValueError, match="first_frame"):
        model.initialize(options={})
    with pytest.raises(ValueError, match="trajectory"):
        model.initialize(options={"first_frame": np.zeros(model.frame_shape)})


def test_example_world_model_rejects_invalid_chunk_lengths_and_shapes() -> None:
    model = ExampleWorldModel(chunk_size=2)
    context = model.initialize(options=_options(model))

    with pytest.raises(ValueError, match="equal chunk_size"):
        model.sample_step(context.context, np.zeros((1, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="equal chunk_size"):
        model.sample_step(context.context, np.zeros((3, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        model.sample_step(context.context, np.zeros((2, 15), dtype=np.float32))
