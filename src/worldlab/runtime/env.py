"""The minimal WorldLab environment/agent interaction loop."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, replace
from typing import Any, Generic, Iterable, List, Mapping, Optional, Tuple, TypeVar

from worldlab.core import Agent, Environment
from worldlab.data import (
    EnvironmentStepped,
    EpisodeEnded,
    EpisodeResult,
    EpisodeStarted,
    PolicyActed,
    RuntimeErrorEvent,
    RuntimeEvent,
    RuntimePhase,
    RuntimeSnapshot,
    Transition,
    TransitionCommitted,
)

from .callbacks import LoopCallback
from .tracing import TraceSink


ObservationT = TypeVar("ObservationT")
ActionT = TypeVar("ActionT")


@dataclass(frozen=True)
class LoopConfig:
    training: bool = True
    deterministic: bool = False
    render: bool = False
    validate_spaces: bool = True
    safety_max_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.safety_max_steps is not None and self.safety_max_steps <= 0:
            raise ValueError("safety_max_steps must be greater than zero")


class EnvironmentLoop(Generic[ObservationT, ActionT]):
    def __init__(
        self,
        env: Environment[ObservationT, ActionT],
        agent: Agent[ObservationT, ActionT],
        *,
        config: Optional[LoopConfig] = None,
        callbacks: Iterable[LoopCallback[ObservationT, ActionT]] = (),
        trace: Optional[TraceSink] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config or LoopConfig()
        self.callbacks: Tuple[LoopCallback[ObservationT, ActionT], ...] = tuple(callbacks)
        self.trace = trace
        self._trace_sequence = 0

    def run_episode(
        self,
        *,
        episode_index: int = 0,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> EpisodeResult[ObservationT]:
        step_index = 0
        attempted_step = 0
        phase = RuntimePhase.ENVIRONMENT_RESET
        phase_started = time.perf_counter()
        episode_started = phase_started

        try:
            reset = self.env.reset(seed=seed, options=options)
            self._validate_observation(reset.observation)

            phase = RuntimePhase.AGENT_RESET
            phase_started = time.perf_counter()
            self.agent.reset(seed=seed)

            observation = reset.observation
            info = dict(reset.info)
            total_reward = 0.0
            self._emit(
                EpisodeStarted(
                    sequence=self._next_trace_sequence(),
                    timestamp_s=time.time(),
                    monotonic_s=time.perf_counter(),
                    episode_index=episode_index,
                    step_index=0,
                    duration_s=time.perf_counter() - episode_started,
                    seed=seed,
                    observation=observation,
                    info=info,
                )
            )

            phase = RuntimePhase.EPISODE_START_CALLBACK
            phase_started = time.perf_counter()
            for callback in self.callbacks:
                callback.on_episode_start(episode_index, observation, info)

            phase = RuntimePhase.RENDER
            phase_started = time.perf_counter()
            self._render_if_requested()

            while True:
                attempted_step = step_index + 1
                phase = RuntimePhase.POLICY_ACT
                phase_started = time.perf_counter()
                policy_output = self.agent.act(
                    observation,
                    info=info,
                    training=self.config.training,
                    deterministic=self.config.deterministic,
                )
                self._validate_action(policy_output.action)
                self._emit(
                    PolicyActed(
                        sequence=self._next_trace_sequence(),
                        timestamp_s=time.time(),
                        monotonic_s=time.perf_counter(),
                        episode_index=episode_index,
                        step_index=attempted_step,
                        duration_s=time.perf_counter() - phase_started,
                        observation=observation,
                        action=policy_output.action,
                        policy_info=dict(policy_output.info),
                        training=self.config.training,
                        deterministic=self.config.deterministic,
                    )
                )

                phase = RuntimePhase.ENVIRONMENT_STEP
                phase_started = time.perf_counter()
                step = self.env.step(policy_output.action)
                self._validate_observation(step.observation)
                step_index = attempted_step

                if (
                    self.config.safety_max_steps is not None
                    and step_index >= self.config.safety_max_steps
                    and not step.done
                ):
                    step_info = dict(step.info)
                    step_info["worldlab.runtime.safety_limit_reached"] = True
                    step = replace(step, truncated=True, info=step_info)

                self._emit(
                    EnvironmentStepped(
                        sequence=self._next_trace_sequence(),
                        timestamp_s=time.time(),
                        monotonic_s=time.perf_counter(),
                        episode_index=episode_index,
                        step_index=step_index,
                        duration_s=time.perf_counter() - phase_started,
                        action=policy_output.action,
                        observation=step.observation,
                        reward=float(step.reward),
                        terminated=step.terminated,
                        truncated=step.truncated,
                        info=dict(step.info),
                    )
                )

                transition_started = time.perf_counter()
                phase = RuntimePhase.TRANSITION_COMMIT
                phase_started = transition_started
                transition = Transition(
                    observation=observation,
                    action=policy_output.action,
                    reward=float(step.reward),
                    next_observation=step.observation,
                    terminated=step.terminated,
                    truncated=step.truncated,
                    info=dict(step.info),
                    policy_info=dict(policy_output.info),
                )

                if self.config.training:
                    phase = RuntimePhase.AGENT_OBSERVE
                    phase_started = time.perf_counter()
                    self.agent.observe(transition)

                phase = RuntimePhase.STEP_CALLBACK
                phase_started = time.perf_counter()
                for callback in self.callbacks:
                    callback.on_step(episode_index, step_index, transition)

                total_reward += transition.reward
                observation = transition.next_observation
                info = dict(transition.info)
                self._emit(
                    TransitionCommitted(
                        sequence=self._next_trace_sequence(),
                        timestamp_s=time.time(),
                        monotonic_s=time.perf_counter(),
                        episode_index=episode_index,
                        step_index=step_index,
                        duration_s=time.perf_counter() - transition_started,
                        transition=transition,
                        total_reward=total_reward,
                        training=self.config.training,
                    )
                )

                phase = RuntimePhase.RENDER
                phase_started = time.perf_counter()
                self._render_if_requested()

                if transition.done:
                    result = EpisodeResult(
                        episode_index=episode_index,
                        total_reward=total_reward,
                        length=step_index,
                        terminated=transition.terminated,
                        truncated=transition.truncated,
                        final_observation=observation,
                        final_info=info,
                    )
                    episode_end_started = time.perf_counter()
                    phase = RuntimePhase.AGENT_END_EPISODE
                    phase_started = episode_end_started
                    self.agent.end_episode(result, training=self.config.training)

                    phase = RuntimePhase.EPISODE_END_CALLBACK
                    phase_started = time.perf_counter()
                    for callback in self.callbacks:
                        callback.on_episode_end(result)

                    self._emit(
                        EpisodeEnded(
                            sequence=self._next_trace_sequence(),
                            timestamp_s=time.time(),
                            monotonic_s=time.perf_counter(),
                            episode_index=episode_index,
                            step_index=step_index,
                            duration_s=time.perf_counter() - episode_end_started,
                            result=result,
                        )
                    )
                    return result
        except Exception as error:
            error_event = RuntimeErrorEvent(
                sequence=self._next_trace_sequence(),
                timestamp_s=time.time(),
                monotonic_s=time.perf_counter(),
                episode_index=episode_index,
                step_index=attempted_step,
                duration_s=time.perf_counter() - phase_started,
                phase=phase,
                error_type=type(error).__name__,
                message=str(error),
                traceback=traceback.format_exc(),
            )
            try:
                self._emit(error_event)
            except Exception:
                # A diagnostic sink must never replace the original loop error.
                pass
            raise

    def run(
        self,
        episodes: int,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> List[EpisodeResult[ObservationT]]:
        if episodes <= 0:
            raise ValueError("episodes must be greater than zero")
        return [
            self.run_episode(
                episode_index=index,
                seed=None if seed is None else seed + index,
                options=options,
            )
            for index in range(episodes)
        ]

    def close(self) -> None:
        self.env.close()

    @property
    def snapshot(self) -> Optional[RuntimeSnapshot]:
        """Return the live snapshot when the configured trace sink provides one."""

        if self.trace is None:
            return None
        snapshot = getattr(self.trace, "snapshot", None)
        return snapshot if isinstance(snapshot, RuntimeSnapshot) else None

    def __enter__(self) -> "EnvironmentLoop[ObservationT, ActionT]":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _validate_observation(self, observation: ObservationT) -> None:
        if self.config.validate_spaces and not self.env.observation_space.contains(observation):
            raise ValueError("environment produced an observation outside observation_space")

    def _validate_action(self, action: ActionT) -> None:
        if self.config.validate_spaces and not self.env.action_space.contains(action):
            raise ValueError("agent produced an action outside action_space")

    def _render_if_requested(self) -> None:
        if self.config.render:
            self.env.render()

    def _next_trace_sequence(self) -> int:
        self._trace_sequence += 1
        return self._trace_sequence

    def _emit(self, event: RuntimeEvent) -> None:
        if self.trace is not None:
            self.trace.record(event)
