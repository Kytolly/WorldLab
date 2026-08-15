"""Deterministic end-to-end acceptance checks for the built-in demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

from .data import (
    EnvironmentStepped,
    EpisodeEnded,
    EpisodeStarted,
    PolicyActed,
    RuntimeEvent,
    RuntimeSnapshot,
    TransitionCommitted,
)
from .demo import run_deterministic_demo
from .runtime import TraceRecorder


@dataclass(frozen=True)
class AcceptanceReport:
    """Machine-readable result of the deterministic closed-loop checks."""

    passed: bool
    checks: Tuple[str, ...]
    failures: Tuple[str, ...]
    goal: int
    seed: int
    total_reward: float
    episode_length: int
    event_count: int
    snapshot: RuntimeSnapshot

    def format(self) -> str:
        lines = [
            f"acceptance_passed={'true' if self.passed else 'false'}",
            f"goal={self.goal}",
            f"seed={self.seed}",
            f"total_reward={self.total_reward}",
            f"episode_length={self.episode_length}",
            f"event_count={self.event_count}",
            f"snapshot_status={self.snapshot.status.value}",
            "checks:",
        ]
        lines.extend(f"- {check}" for check in self.checks)
        if self.failures:
            lines.append("failures:")
            lines.extend(f"- {failure}" for failure in self.failures)
        return "\n".join(lines)


def run_acceptance(*, goal: int = 3, seed: int = 0) -> AcceptanceReport:
    """Run the deterministic demo twice and compare semantic runtime output."""

    if goal <= 0:
        raise ValueError("goal must be greater than zero")

    first_result, first_trace = _run_once(goal=goal, seed=seed)
    second_result, second_trace = _run_once(goal=goal, seed=seed)
    checks: List[str] = []
    failures: List[str] = []

    def check(name: str, condition: bool) -> None:
        (checks if condition else failures).append(name)

    check(
        "result reaches the goal with one reward per step",
        first_result.total_reward == float(goal)
        and first_result.length == goal
        and first_result.terminated
        and not first_result.truncated
        and first_result.final_observation == goal,
    )
    check(
        "event sequence contains one start, three events per step, and one end",
        _event_kinds(first_trace) == _expected_kinds(goal),
    )
    check(
        "event sequence numbers are contiguous",
        [event.sequence for event in first_trace.events]
        == list(range(1, 3 * goal + 3)),
    )
    check(
        "policy actions and observations follow the deterministic counter path",
        _counter_path_is_valid(first_trace, goal),
    )
    check(
        "trace diagnosis reports a healthy closed loop",
        first_trace.diagnose().healthy,
    )
    check(
        "final snapshot matches the episode result",
        _snapshot_is_valid(first_trace.snapshot, first_result, goal),
    )
    check(
        "repeated run has the same result and semantic events",
        _result_signature(first_result) == _result_signature(second_result)
        and _semantic_events(first_trace.events)
        == _semantic_events(second_trace.events),
    )

    return AcceptanceReport(
        passed=not failures,
        checks=tuple(checks),
        failures=tuple(failures),
        goal=goal,
        seed=seed,
        total_reward=first_result.total_reward,
        episode_length=first_result.length,
        event_count=len(first_trace.events),
        snapshot=first_trace.snapshot,
    )


def _run_once(*, goal: int, seed: int) -> tuple[Any, TraceRecorder]:
    trace = TraceRecorder()
    result = run_deterministic_demo(goal=goal, seed=seed, trace=trace)
    return result, trace


def _expected_kinds(goal: int) -> Tuple[str, ...]:
    kinds = ["episode_started"]
    for _ in range(goal):
        kinds.extend(("policy_acted", "environment_stepped", "transition_committed"))
    kinds.append("episode_ended")
    return tuple(kinds)


def _event_kinds(trace: TraceRecorder) -> Tuple[str, ...]:
    return tuple(event.kind.value for event in trace.events)


def _counter_path_is_valid(trace: TraceRecorder, goal: int) -> bool:
    policy_events = [event for event in trace.events if isinstance(event, PolicyActed)]
    step_events = [
        event for event in trace.events if isinstance(event, EnvironmentStepped)
    ]
    commit_events = [
        event for event in trace.events if isinstance(event, TransitionCommitted)
    ]
    return (
        len(policy_events) == goal
        and len(step_events) == goal
        and len(commit_events) == goal
        and [event.action for event in policy_events] == [1] * goal
        and [event.observation for event in step_events] == list(range(1, goal + 1))
        and [event.reward for event in step_events] == [1.0] * goal
        and [event.total_reward for event in commit_events]
        == [float(index) for index in range(1, goal + 1)]
        and step_events[-1].terminated
        and not step_events[-1].truncated
    )


def _snapshot_is_valid(
    snapshot: RuntimeSnapshot,
    result: Any,
    goal: int,
) -> bool:
    return (
        snapshot.status.value == "completed"
        and snapshot.sequence == 3 * goal + 2
        and snapshot.step_index == result.length == goal
        and snapshot.total_reward == result.total_reward == float(goal)
        and snapshot.terminated is True
        and snapshot.truncated is False
        and snapshot.observation == goal
        and snapshot.next_observation == goal
    )


def _result_signature(result: Any) -> tuple[Any, ...]:
    return (
        result.episode_index,
        result.total_reward,
        result.length,
        result.terminated,
        result.truncated,
        result.final_observation,
        dict(result.final_info),
    )


def _semantic_events(events: Tuple[RuntimeEvent, ...]) -> Tuple[Any, ...]:
    return tuple(_semantic_event(event) for event in events)


def _semantic_event(event: RuntimeEvent) -> tuple[Any, ...]:
    common = (event.kind.value, event.episode_index, event.step_index)
    if isinstance(event, EpisodeStarted):
        return common + (event.seed, event.observation, dict(event.info))
    if isinstance(event, PolicyActed):
        return common + (
            event.observation,
            event.action,
            dict(event.policy_info),
            event.training,
            event.deterministic,
        )
    if isinstance(event, EnvironmentStepped):
        return common + (
            event.action,
            event.observation,
            event.reward,
            event.terminated,
            event.truncated,
            dict(event.info),
        )
    if isinstance(event, TransitionCommitted):
        transition = event.transition
        return common + (
            transition.observation,
            transition.action,
            transition.reward,
            transition.next_observation,
            transition.terminated,
            transition.truncated,
            dict(transition.info),
            dict(transition.policy_info),
            event.total_reward,
            event.training,
        )
    if isinstance(event, EpisodeEnded):
        result = event.result
        return common + (
            result.episode_index,
            result.total_reward,
            result.length,
            result.terminated,
            result.truncated,
            result.final_observation,
            dict(result.final_info),
        )
    return common
