from __future__ import annotations

import numpy as np
import pytest

from worldlab.core import ObservationProvider, TerminationProvider
from worldlab.data import (
    ObservationResult,
    SIMULATION_FRAMES,
    SimulationReset,
    SimulationStep,
    TaskStepContext,
    TerminationResult,
)
from worldlab_gesim import (
    GESimRewardAdapter,
    GESimRewardProvider,
    GESimTask,
    GESimWorldModelEnv,
    Qwen25VLWorldJudge,
    RewardClient,
    RewardResult,
    WorldJudge,
    head_view_frames,
)


class _Observation(ObservationProvider):
    def reset(self, context):
        return ObservationResult(context.simulation.state)

    def compute(self, context):
        return ObservationResult(context.simulation.state)


class _Termination(TerminationProvider):
    def compute(self, context):
        return TerminationResult(False, False)


class _FakeWorldJudge(WorldJudge):
    def evaluate(self, head_frames: np.ndarray, task: str) -> RewardResult:
        assert head_frames.dtype == np.uint8
        assert head_frames.ndim == 4
        assert task == "open the drawer"
        return RewardResult(
            success=np.array([0.2, 0.9], dtype=np.float32),
            progress=None,
        )


class _WorldModelClient:
    def reset(self, *, seed=None, options=None):
        return SimulationReset(0)

    def step(self, action):
        return SimulationStep(
            state=action,
            info={SIMULATION_FRAMES: np.zeros((2, 3, 1, 4, 5), dtype=np.float32)},
        )

    def close(self):
        pass


def test_world_judge_is_the_local_reward_client_boundary() -> None:
    judge = _FakeWorldJudge()

    assert isinstance(judge, RewardClient)
    result = judge.evaluate(np.zeros((2, 4, 5, 3), dtype=np.uint8), "open the drawer")
    assert result.progress is None
    assert result.success.shape == (2,)


def test_head_view_conversion_matches_author_reward_input() -> None:
    frames = np.zeros((2, 3, 1, 4, 5), dtype=np.float32)
    frames[1, :, 0] = 1.0

    head = head_view_frames(frames)

    assert head.shape == (2, 4, 5, 3)
    assert head.dtype == np.uint8
    assert int(head[-1].max()) == 255


def test_reward_adapter_uses_last_success_without_fabricating_progress() -> None:
    reward = GESimRewardAdapter().adapt(
        RewardResult(
            success=np.array([0.1, 0.75], dtype=np.float32),
            progress=None,
        )
    )

    assert reward.value == pytest.approx(0.75)
    assert "gesim.judge.progress" not in reward.info


def test_gesim_task_passes_string_task_to_local_judge() -> None:
    task = GESimTask(
        instruction="open the drawer",
        observation=_Observation(),
        termination=_Termination(),
        world_judge=_FakeWorldJudge(),
    )
    simulation = SimulationStep(
        state=1,
        info={SIMULATION_FRAMES: np.zeros((2, 3, 1, 4, 5), dtype=np.float32)},
    )
    result = task.step(0, 1, simulation)

    assert str(task) == "open the drawer"
    assert result.reward == pytest.approx(0.9)
    assert result.info["worldlab.task.reward"]["gesim.judge.score_policy"] == "last"


def test_gesim_task_accepts_prebuilt_reward_provider_without_task_ownership() -> None:
    judge = _FakeWorldJudge()
    provider = GESimRewardProvider(reward_client=judge)
    task = GESimTask(
        instruction="open the drawer",
        observation=_Observation(),
        termination=_Termination(),
        reward=provider,
    )
    simulation = SimulationStep(
        state=1,
        info={
            SIMULATION_FRAMES: np.zeros((2, 3, 1, 4, 5), dtype=np.float32),
            "gesim.task": "open the drawer",
        },
    )

    result = task.step(0, 1, simulation)

    assert result.reward == pytest.approx(0.9)
    assert not hasattr(provider, "task_instance")


def test_gesim_world_model_env_adapts_worldlab_lifecycle() -> None:
    task = GESimTask(
        instruction="open the drawer",
        observation=_Observation(),
        termination=_Termination(),
    )
    env = GESimWorldModelEnv(
        _WorldModelClient(),
        task,
        observation_space=None,
        action_space=None,
    )

    assert env.reset().observation == 0
    assert env.step(1).reward == 0.0
    assert env.task_instruction == "open the drawer"
    env.close()


def test_qwen_judge_requires_local_model_and_success_head() -> None:
    judge = Qwen25VLWorldJudge(
        model_path="model_zoo/qwen/Qwen2.5-VL-3B-Instruct",
        success_head_path="model_zoo/qwen/world_judge_head.pt",
    )

    with pytest.raises(FileNotFoundError):
        judge.load()


def test_reward_result_rejects_mismatched_training_outputs() -> None:
    with pytest.raises(ValueError):
        RewardResult(
            success=np.array([0.2], dtype=np.float32),
            progress=np.array([0.1, 0.7], dtype=np.float32),
        )
