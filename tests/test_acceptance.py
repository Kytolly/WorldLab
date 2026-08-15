from __future__ import annotations

import sys

import pytest

from worldlab import run_acceptance, run_deterministic_demo
from worldlab.__main__ import main


def test_deterministic_demo_reaches_the_declared_goal() -> None:
    result = run_deterministic_demo(goal=4, seed=123)

    assert result.total_reward == 4.0
    assert result.length == 4
    assert result.terminated is True
    assert result.truncated is False
    assert result.final_observation == 4


def test_acceptance_repeats_and_validates_the_semantic_trace() -> None:
    report = run_acceptance(goal=3, seed=0)

    assert report.passed is True
    assert report.failures == ()
    assert report.event_count == 11
    assert report.snapshot.status.value == "completed"
    assert len(report.checks) == 7


def test_acceptance_rejects_non_positive_goal() -> None:
    with pytest.raises(ValueError, match="goal"):
        run_acceptance(goal=0)


def test_cli_acceptance_returns_a_machine_readable_pass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["worldlab", "--acceptance", "--goal", "2", "--seed", "7"],
    )

    assert main() == 0

    output = capsys.readouterr().out
    assert "acceptance_passed=true" in output
    assert "event_count=8" in output
    assert "repeated run has the same result and semantic events" in output
