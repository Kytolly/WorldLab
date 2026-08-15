"""Runtime trace sinks, recording, formatting, and sequence diagnosis."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import replace
from typing import Any, Deque, List, Optional, Protocol, Tuple, runtime_checkable

from worldlab.data.runtime import (
    EnvironmentStepped,
    EpisodeEnded,
    EpisodeStarted,
    PolicyActed,
    RuntimeErrorEvent,
    RuntimeEvent,
    RuntimeEventKind,
    RuntimeSnapshot,
    RuntimeStatus,
    RuntimePhase,
    TraceDiagnosis,
    TransitionCommitted,
)


@runtime_checkable
class TraceSink(Protocol):
    """Receives immutable events without participating in loop control."""

    def record(self, event: RuntimeEvent) -> None:
        ...


class EventBuffer:
    """Thread-safe bounded event history with a live runtime snapshot."""

    def __init__(self, max_events: int = 1024) -> None:
        if max_events <= 0:
            raise ValueError("max_events must be greater than zero")
        self._events: Deque[RuntimeEvent] = deque(maxlen=max_events)
        self._lock = threading.RLock()
        self._snapshot = _initial_snapshot()

    @property
    def events(self) -> Tuple[RuntimeEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return self._snapshot

    def since(self, sequence: int = 0) -> Tuple[RuntimeEvent, ...]:
        """Return buffered events whose sequence is greater than ``sequence``."""

        with self._lock:
            return tuple(event for event in self._events if event.sequence > sequence)

    def record(self, event: RuntimeEvent) -> None:
        with self._lock:
            self._events.append(event)
            self._snapshot = _snapshot_for_event(self._snapshot, event)

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
            self._snapshot = _initial_snapshot()

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)


class TraceRecorder(EventBuffer):
    """Keep runtime events in memory for direct inspection and diagnosis."""

    def __init__(self, max_events: Optional[int] = None) -> None:
        # A recorder is intended for complete post-run diagnosis. The optional
        # bound makes it suitable for long-running live monitoring as well.
        if max_events is None:
            max_events = 2**31 - 1
        super().__init__(max_events=max_events)

    @property
    def errors(self) -> Tuple[RuntimeErrorEvent, ...]:
        return tuple(
            event for event in self.events if isinstance(event, RuntimeErrorEvent)
        )

    def diagnose(self) -> TraceDiagnosis:
        """Check event ordering and report incomplete or failed closed loops."""

        events = self.events
        if not events:
            return TraceDiagnosis(False, ("no runtime events were recorded",))

        issues: List[str] = []
        expected: Optional[RuntimeEventKind] = RuntimeEventKind.EPISODE_STARTED
        active_episode: Optional[int] = None
        expected_step = 0
        previous_sequence = 0
        previous_monotonic: Optional[float] = None
        transitions = 0
        episodes = 0

        for event in events:
            if event.sequence != previous_sequence + 1:
                issues.append(
                    f"event sequence jumped from {previous_sequence} to {event.sequence}"
                )
            previous_sequence = event.sequence
            if previous_monotonic is not None and event.monotonic_s < previous_monotonic:
                issues.append(f"event {event.sequence} has a non-monotonic timestamp")
            previous_monotonic = event.monotonic_s

            if isinstance(event, RuntimeErrorEvent):
                issues.append(
                    f"episode {event.episode_index} step {event.step_index} failed "
                    f"during {event.phase.value}: {event.error_type}: {event.message}"
                )
                continue

            if expected is not None and event.kind is not expected:
                issues.append(
                    f"event {event.sequence} expected {expected.value}, got {event.kind.value}"
                )

            if isinstance(event, EpisodeStarted):
                if active_episode is not None:
                    issues.append(
                        f"episode {active_episode} did not end before episode "
                        f"{event.episode_index} started"
                    )
                active_episode = event.episode_index
                expected_step = 1
                if event.step_index != 0:
                    issues.append(
                        f"episode_started event {event.sequence} must use step 0"
                    )
                expected = RuntimeEventKind.POLICY_ACTED
            elif isinstance(event, PolicyActed):
                self._check_episode(event, active_episode, issues)
                self._check_step(event, expected_step, issues)
                expected = RuntimeEventKind.ENVIRONMENT_STEPPED
            elif isinstance(event, EnvironmentStepped):
                self._check_episode(event, active_episode, issues)
                self._check_step(event, expected_step, issues)
                expected = RuntimeEventKind.TRANSITION_COMMITTED
            elif isinstance(event, TransitionCommitted):
                self._check_episode(event, active_episode, issues)
                self._check_step(event, expected_step, issues)
                transitions += 1
                expected = (
                    RuntimeEventKind.EPISODE_ENDED
                    if event.transition.done
                    else RuntimeEventKind.POLICY_ACTED
                )
                if not event.transition.done:
                    expected_step += 1
            elif isinstance(event, EpisodeEnded):
                self._check_episode(event, active_episode, issues)
                self._check_step(event, expected_step, issues)
                episodes += 1
                active_episode = None
                expected = RuntimeEventKind.EPISODE_STARTED

        if active_episode is not None and not self.errors:
            issues.append(f"episode {active_episode} has no episode_ended event")

        if issues:
            return TraceDiagnosis(False, tuple(issues))
        return TraceDiagnosis(
            True,
            (
                f"closed loop completed: {episodes} episode(s), "
                f"{transitions} transition(s)",
            ),
        )

    def format_timeline(self) -> str:
        """Return a compact, human-readable trace without losing raw events."""

        events = self.events
        if not events:
            return "(no runtime events)"
        started_at = events[0].monotonic_s
        lines: List[str] = []
        for event in events:
            elapsed = event.monotonic_s - started_at
            prefix = (
                f"#{event.sequence:03d} +{elapsed:.6f}s "
                f"{event.kind.value} episode={event.episode_index} "
                f"step={event.step_index} duration={event.duration_s:.6f}s"
            )
            lines.append(f"{prefix} {self._details(event)}".rstrip())
        diagnosis = self.diagnose()
        status = "healthy" if diagnosis.healthy else "unhealthy"
        lines.append(f"diagnosis={status}: {'; '.join(diagnosis.messages)}")
        return "\n".join(lines)

    @staticmethod
    def _check_episode(
        event: RuntimeEvent,
        active_episode: Optional[int],
        issues: List[str],
    ) -> None:
        if active_episode is None:
            issues.append(
                f"event {event.sequence} {event.kind.value} has no active episode"
            )
        elif event.episode_index != active_episode:
            issues.append(
                f"event {event.sequence} belongs to episode {event.episode_index}, "
                f"expected {active_episode}"
            )

    @staticmethod
    def _check_step(
        event: RuntimeEvent,
        expected_step: int,
        issues: List[str],
    ) -> None:
        if event.step_index != expected_step:
            issues.append(
                f"event {event.sequence} uses step {event.step_index}, "
                f"expected {expected_step}"
            )

    @staticmethod
    def _details(event: RuntimeEvent) -> str:
        if isinstance(event, EpisodeStarted):
            return f"observation={_short_repr(event.observation)} info={dict(event.info)!r}"
        if isinstance(event, PolicyActed):
            return f"action={_short_repr(event.action)} policy_info={dict(event.policy_info)!r}"
        if isinstance(event, EnvironmentStepped):
            return (
                f"observation={_short_repr(event.observation)} reward={event.reward} "
                f"terminated={event.terminated} truncated={event.truncated}"
            )
        if isinstance(event, TransitionCommitted):
            return f"total_reward={event.total_reward} training={event.training}"
        if isinstance(event, EpisodeEnded):
            return (
                f"total_reward={event.result.total_reward} length={event.result.length} "
                f"terminated={event.result.terminated} truncated={event.result.truncated}"
            )
        if isinstance(event, RuntimeErrorEvent):
            return (
                f"phase={event.phase.value} error={event.error_type}: "
                f"{event.message}"
            )
        return ""


def _short_repr(value: object, limit: int = 120) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _initial_snapshot() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        status=RuntimeStatus.IDLE,
        sequence=0,
        timestamp_s=time.time(),
        monotonic_s=time.perf_counter(),
    )


def _snapshot_for_event(previous: RuntimeSnapshot, event: RuntimeEvent) -> RuntimeSnapshot:
    common: dict[str, Any] = {
        "sequence": event.sequence,
        "timestamp_s": event.timestamp_s,
        "monotonic_s": event.monotonic_s,
        "episode_index": event.episode_index,
        "step_index": event.step_index,
        "last_event": event.kind,
    }
    if isinstance(event, EpisodeStarted):
        return replace(
            previous,
            **common,
            status=RuntimeStatus.READY,
            phase=None,
            error_type=None,
            error_message=None,
            total_reward=0.0,
            reward=None,
            terminated=False,
            truncated=False,
            observation=event.observation,
            action=None,
            next_observation=None,
            info=dict(event.info),
            policy_info={},
        )
    if isinstance(event, PolicyActed):
        return replace(
            previous,
            **common,
            status=RuntimeStatus.RUNNING,
            phase=None,
            error_type=None,
            error_message=None,
            observation=event.observation,
            action=event.action,
            policy_info=dict(event.policy_info),
        )
    if isinstance(event, EnvironmentStepped):
        return replace(
            previous,
            **common,
            status=RuntimeStatus.RUNNING,
            phase=None,
            error_type=None,
            error_message=None,
            action=event.action,
            next_observation=event.observation,
            reward=event.reward,
            terminated=event.terminated,
            truncated=event.truncated,
            info=dict(event.info),
        )
    if isinstance(event, TransitionCommitted):
        transition = event.transition
        return replace(
            previous,
            **common,
            status=RuntimeStatus.RUNNING,
            phase=None,
            error_type=None,
            error_message=None,
            total_reward=event.total_reward,
            reward=transition.reward,
            terminated=transition.terminated,
            truncated=transition.truncated,
            observation=transition.observation,
            action=transition.action,
            next_observation=transition.next_observation,
            info=dict(transition.info),
            policy_info=dict(transition.policy_info),
        )
    if isinstance(event, EpisodeEnded):
        result = event.result
        return replace(
            previous,
            **common,
            status=RuntimeStatus.COMPLETED,
            phase=None,
            error_type=None,
            error_message=None,
            total_reward=result.total_reward,
            reward=None,
            terminated=result.terminated,
            truncated=result.truncated,
            observation=result.final_observation,
            next_observation=result.final_observation,
            info=dict(result.final_info),
        )
    if isinstance(event, RuntimeErrorEvent):
        return replace(
            previous,
            **common,
            status=RuntimeStatus.FAILED,
            phase=event.phase,
            error_type=event.error_type,
            error_message=event.message,
        )
    return previous
