from __future__ import annotations

import sys

import pytest

from worldlab import (
    EventBuffer,
    EnvironmentLoop,
    EnvironmentStepped,
    EpisodeEnded,
    EpisodeStarted,
    PolicyActed,
    RuntimeErrorEvent,
    RuntimeEventKind,
    RuntimePhase,
    RuntimeStatus,
    TraceRecorder,
    Transition,
    TransitionCommitted,
    build_demo,
)
from worldlab.__main__ import main
from worldlab.agents import PolicyAgent
from worldlab.envs import make_counter_environment
from worldlab.policies import CallablePolicy
from worldlab.runtime import LoopCallback


def test_trace_records_a_complete_closed_loop() -> None:
    env, agent = build_demo(goal=2)
    trace = TraceRecorder()

    with EnvironmentLoop(env, agent, trace=trace) as loop:
        result = loop.run_episode(seed=7)

    assert result.total_reward == 2.0
    assert [event.kind for event in trace.events] == [
        RuntimeEventKind.EPISODE_STARTED,
        RuntimeEventKind.POLICY_ACTED,
        RuntimeEventKind.ENVIRONMENT_STEPPED,
        RuntimeEventKind.TRANSITION_COMMITTED,
        RuntimeEventKind.POLICY_ACTED,
        RuntimeEventKind.ENVIRONMENT_STEPPED,
        RuntimeEventKind.TRANSITION_COMMITTED,
        RuntimeEventKind.EPISODE_ENDED,
    ]
    assert [event.sequence for event in trace.events] == list(range(1, 9))
    assert all(event.duration_s >= 0.0 for event in trace.events)

    assert isinstance(trace.events[0], EpisodeStarted)
    assert trace.events[0].observation == 0
    assert isinstance(trace.events[1], PolicyActed)
    assert trace.events[1].action == 1
    assert isinstance(trace.events[2], EnvironmentStepped)
    assert trace.events[2].observation == 1
    assert isinstance(trace.events[3], TransitionCommitted)
    assert trace.events[3].total_reward == 1.0
    assert isinstance(trace.events[-1], EpisodeEnded)
    assert trace.events[-1].result == result

    diagnosis = trace.diagnose()
    assert diagnosis.healthy is True
    assert diagnosis.messages == (
        "closed loop completed: 1 episode(s), 2 transition(s)",
    )


def test_trace_records_the_exact_failure_phase_and_reraises() -> None:
    env = make_counter_environment(goal=2)
    agent = PolicyAgent(
        CallablePolicy[int, int](lambda observation, info, deterministic: 4)
    )
    trace = TraceRecorder()

    with pytest.raises(ValueError, match="action outside action_space"):
        EnvironmentLoop(env, agent, trace=trace).run_episode(seed=0)

    assert isinstance(trace.events[0], EpisodeStarted)
    assert len(trace.errors) == 1
    error = trace.errors[0]
    assert isinstance(error, RuntimeErrorEvent)
    assert error.phase is RuntimePhase.POLICY_ACT
    assert error.step_index == 1
    assert error.error_type == "ValueError"
    assert "action outside action_space" in error.message
    assert "ValueError" in error.traceback
    assert trace.snapshot.status is RuntimeStatus.FAILED
    assert trace.snapshot.phase is RuntimePhase.POLICY_ACT
    assert trace.snapshot.error_type == "ValueError"
    assert trace.diagnose().healthy is False


def test_trace_identifies_callback_failures_after_environment_step() -> None:
    class FailingCallback(LoopCallback[int, int]):
        def on_step(
            self,
            episode_index: int,
            step_index: int,
            transition: Transition[int, int],
        ) -> None:
            del episode_index, step_index, transition
            raise RuntimeError("diagnostic callback failure")

    env, agent = build_demo(goal=2)
    trace = TraceRecorder()

    with pytest.raises(RuntimeError, match="diagnostic callback failure"):
        with EnvironmentLoop(
            env,
            agent,
            callbacks=[FailingCallback()],
            trace=trace,
        ) as loop:
            loop.run_episode(seed=0)

    assert [event.kind for event in trace.events] == [
        RuntimeEventKind.EPISODE_STARTED,
        RuntimeEventKind.POLICY_ACTED,
        RuntimeEventKind.ENVIRONMENT_STEPPED,
        RuntimeEventKind.RUNTIME_ERROR,
    ]
    assert trace.errors[0].phase is RuntimePhase.STEP_CALLBACK
    assert "diagnostic callback failure" in trace.format_timeline()


def test_trace_timeline_is_directly_readable() -> None:
    env, agent = build_demo(goal=1)
    trace = TraceRecorder()
    EnvironmentLoop(env, agent, trace=trace).run_episode(seed=0)

    timeline = trace.format_timeline()

    assert "episode_started episode=0 step=0" in timeline
    assert "policy_acted episode=0 step=1" in timeline
    assert "environment_stepped episode=0 step=1" in timeline
    assert "transition_committed episode=0 step=1" in timeline
    assert "episode_ended episode=0 step=1" in timeline
    assert "diagnosis=healthy" in timeline


def test_event_buffer_keeps_a_live_snapshot_and_bounded_history() -> None:
    env, agent = build_demo(goal=2)
    buffer = EventBuffer(max_events=3)

    with EnvironmentLoop(env, agent, trace=buffer) as loop:
        result = loop.run_episode(seed=0)
        assert loop.snapshot is buffer.snapshot

    assert result.total_reward == 2.0
    assert len(buffer) == 3
    assert [event.sequence for event in buffer.events] == [6, 7, 8]
    assert [event.sequence for event in buffer.since(6)] == [7, 8]

    snapshot = buffer.snapshot
    assert snapshot.status is RuntimeStatus.COMPLETED
    assert snapshot.sequence == 8
    assert snapshot.episode_index == 0
    assert snapshot.step_index == 2
    assert snapshot.total_reward == 2.0
    assert snapshot.terminated is True
    assert snapshot.truncated is False
    assert snapshot.observation == 2
    assert snapshot.next_observation == 2

    buffer.clear()
    assert len(buffer) == 0
    assert buffer.snapshot.status is RuntimeStatus.IDLE
    assert buffer.snapshot.sequence == 0


def test_event_buffer_rejects_non_positive_capacity() -> None:
    with pytest.raises(ValueError, match="max_events"):
        EventBuffer(max_events=0)


def test_cli_trace_prints_timeline(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["worldlab", "--goal", "1", "--trace"])

    assert main() == 0

    output = capsys.readouterr().out
    assert "WorldLab demo" in output
    assert "closed_loop_trace" in output
    assert "episode_started" in output
    assert "diagnosis=healthy" in output
