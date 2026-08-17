"""Release-level checks for the v0.3.4 package and acceptance surface."""

from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _project_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    assert match is not None, f"missing project version in {path}"
    return match.group(1)


def test_all_distributable_packages_share_the_release_version() -> None:
    project_files = (
        ROOT / "pyproject.toml",
        ROOT / "packages" / "worldlab-ui-panel" / "pyproject.toml",
        ROOT / "packages" / "worldlab-integrations-openpi" / "pyproject.toml",
    )

    assert {_project_version(path) for path in project_files} == {"0.3.4"}


def test_runtime_acceptance_and_optional_protocol_fixtures_are_present() -> None:
    assert (ROOT / "src" / "worldlab" / "acceptance.py").is_file()
    assert (ROOT / "packages" / "worldlab-integrations-openpi" / "tests" / "fake_server.py").is_file()
    assert (ROOT / "packages" / "worldlab-ui-panel" / "tests" / "test_panel_app.py").is_file()


def test_docker_service_keeps_a_stable_internal_port() -> None:
    compose = (ROOT / "docker-compose.openpi.yml").read_text(encoding="utf-8")

    assert '"${OPENPI_PORT:-9000}:8000"' in compose
    assert '"8000"' in compose
    assert "worldlab_openpi.openpi_serving.serve_pi05" in compose
