from __future__ import annotations

import numpy as np

from worldlab_openpi import (
    ActionLayout,
    OpenPIObservation,
    OpenPIPolicy,
    build_openpi_payload,
    convert_actions,
)
from fake_server import FakeOpenPIServer


def _observation() -> OpenPIObservation:
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    return OpenPIObservation(
        images={"head": image, "left_wrist": image.copy(), "right_wrist": image.copy()},
        state=np.arange(16, dtype=np.float32),
        task="pick up the object",
    )


def test_payload_contains_openpi_keys_and_numpy_arrays() -> None:
    payload = build_openpi_payload(_observation())

    assert payload["prompt"] == "pick up the object"
    assert payload["observation.state"].shape == (16,)
    assert payload["observation.images.head"].shape == (4, 5, 3)
    assert payload["observation.images.head"].flags.c_contiguous


def test_policy_to_world_model_layout_conversion() -> None:
    policy = np.arange(32, dtype=np.float32).reshape(2, 16)
    world_model = convert_actions(
        policy,
        source=ActionLayout.POLICY,
        target=ActionLayout.WORLD_MODEL,
    )

    assert np.array_equal(world_model[0, :7], policy[0, :7])
    assert world_model[0, 7] == policy[0, 14]
    assert np.array_equal(world_model[0, 8:15], policy[0, 7:14])
    assert world_model[0, 15] == policy[0, 15]


def test_fake_server_validates_network_payload_shape_and_layout() -> None:
    def fake_policy(payload):
        assert payload["observation.state"].shape == (16,)
        assert payload["observation.images.head"].shape == (4, 5, 3)
        policy_actions = np.arange(2 * 16, dtype=np.float32).reshape(2, 16)
        return {"actions": policy_actions}

    with FakeOpenPIServer(
        action_horizon=2,
        action_layout=ActionLayout.POLICY.value,
        policy=fake_policy,
    ) as server:
        policy = OpenPIPolicy(
            server.url,
            action_horizon=2,
            server_action_layout=ActionLayout.POLICY,
        )
        output = policy.act(_observation(), info={}, deterministic=True)

        assert output.action.shape == (2, 16)
        assert output.action.dtype == np.float32
        assert output.action[0, 7] == 14.0
        assert output.action[0, 8] == 7.0
        assert output.info["openpi.server_action_layout"] == "policy"
        assert len(server.requests) == 1
        assert server.requests[0]["prompt"] == "pick up the object"
