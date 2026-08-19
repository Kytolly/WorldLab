"""Reference GE-Sim closed-loop entry point.

The object construction lives in :mod:`worldlab_gesim.factory`; this module
only provides example-specific observation/termination components and CLI
argument handling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from worldlab import ArraySpace
from worldlab.core import ObservationProvider, Policy, TerminationProvider
from worldlab.data import ObservationResult, PolicyOutput, TerminationResult
from worldlab.runtime import LoopConfig

from gesim.client.transport import WorldModelClient
from gesim.episode import EpisodeBundle
from gesim.types import ACTION_DIM, Observation

from .factory import make_gesim_environment, make_gesim_loop, make_gesim_task
from .judge import WorldJudge


class ObservationStateProvider(ObservationProvider):
    """Expose the simulator's current multi-camera observation to the task."""

    def reset(self, context: Any) -> ObservationResult[Observation]:
        return ObservationResult(context.simulation.state)

    def compute(self, context: Any) -> ObservationResult[Observation]:
        return ObservationResult(context.simulation.state)


class ChunkLimitTermination(TerminationProvider):
    """Example finite-horizon termination term for replay experiments."""

    def __init__(self, max_chunks: int) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be greater than zero")
        self.max_chunks = int(max_chunks)
        self._chunks = 0

    def reset(self, context: Any) -> None:
        self._chunks = 0

    def compute(self, context: Any) -> TerminationResult:
        self._chunks += 1
        return TerminationResult(
            terminated=False,
            truncated=self._chunks >= self.max_chunks,
            info={"gesim.rollout.chunk_count": self._chunks},
        )


class ReplayChunkPolicy(Policy[Observation, np.ndarray]):
    """Example policy that replays complete fixed-size action chunks."""

    def __init__(self, actions: np.ndarray, *, chunk_size: int) -> None:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"actions must have shape (T, {ACTION_DIM})")
        if actions.shape[0] < chunk_size:
            raise ValueError("episode does not contain one complete action chunk")
        self.actions = actions
        self.chunk_size = int(chunk_size)
        self._offset = 0

    def reset(self, *, seed: int | None = None) -> None:
        del seed
        self._offset = 0

    def act(
        self,
        observation: Observation,
        *,
        info: dict[str, Any],
        deterministic: bool = False,
    ) -> PolicyOutput[np.ndarray]:
        del observation, info, deterministic
        end = self._offset + self.chunk_size
        if end > self.actions.shape[0]:
            raise StopIteration("replay policy has no complete action chunk left")
        chunk = self.actions[self._offset : end].copy()
        self._offset = end
        return PolicyOutput(chunk)


class ObservationSpace:
    """Opaque Space for asset-backed author-style observations."""

    def sample(self) -> Observation:
        raise RuntimeError("sample() is not defined for an asset-backed observation")

    def contains(self, value: object) -> bool:
        return isinstance(value, Observation)

    def seed(self, seed: int | None = None) -> None:
        del seed


def build_reference(
    *,
    world_model_url: str,
    episode: str | Path | EpisodeBundle,
    chunk_size: int = 25,
    judge_head: str | Path | None = None,
    model_path: str | Path = "model_zoo/qwen/Qwen2.5-VL-3B-Instruct",
    device: str | None = None,
):
    """Build a complete reference loop using a remote GE-Sim backend."""

    bundle = episode if isinstance(episode, EpisodeBundle) else EpisodeBundle.load(episode)
    if bundle.actions is None:
        raise ValueError("the episode bundle must contain actions_0.npy")
    judge = None
    if judge_head is not None:
        judge = WorldJudge(
            model_path=model_path,
            success_head_path=judge_head,
            device=device,
            load_in_constructor=True,
        )
    task = make_gesim_task(
        instruction=bundle.task,
        observation=ObservationStateProvider(),
        termination=ChunkLimitTermination(
            max_chunks=max(1, bundle.actions.shape[0] // chunk_size)
        ),
        reward_client=judge,
    )
    env = make_gesim_environment(
        backend=WorldModelClient(world_model_url),
        task=task,
        observation_space=ObservationSpace(),
        action_space=ArraySpace(
            (chunk_size, ACTION_DIM), dtype=np.float32, low=-np.inf, high=np.inf
        ),
        chunk_size=chunk_size,
    )
    return make_gesim_loop(
        env=env,
        policy=ReplayChunkPolicy(bundle.actions, chunk_size=chunk_size),
        config=LoopConfig(training=False, deterministic=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--world-model", default="http://localhost:9000")
    parser.add_argument("--judge-head")
    parser.add_argument("--model-path", default="model_zoo/qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--device", default=None)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--conditioning", choices=("action", "episode"), default="action")
    args = parser.parse_args()

    loop = build_reference(
        world_model_url=args.world_model,
        episode=args.episode,
        chunk_size=args.chunk_size,
        judge_head=args.judge_head,
        model_path=args.model_path,
        device=args.device,
    )
    with loop:
        result = loop.run_episode(
            options={"episode": args.episode, "conditioning": args.conditioning}
        )
    print(
        f"episode length={result.length} total_reward={result.total_reward:.4f} "
        f"terminated={result.terminated} truncated={result.truncated}"
    )


if __name__ == "__main__":
    main()
