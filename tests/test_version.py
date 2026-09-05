"""Distribution identity and version consistency tests."""

from importlib.metadata import requires, version
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from click.testing import CliRunner

from rai import __version__
from rai.cli import cli


def test_distribution_and_runtime_versions_match() -> None:
    """Installed metadata and the importable package expose one version."""
    assert version("rich-ai") == __version__


def test_distribution_does_not_depend_on_foreign_rai_package() -> None:
    """Extras must reference this distribution rather than PyPI's unrelated rai."""
    dependency_names = {
        canonicalize_name(Requirement(requirement).name)
        for requirement in requires("rich-ai") or []
    }

    assert "rai" not in dependency_names


def test_distribution_has_no_direct_url_dependencies() -> None:
    """Public index metadata must not require packages from arbitrary URLs."""
    dependencies = [Requirement(requirement) for requirement in requires("rich-ai") or []]

    assert all(dependency.url is None for dependency in dependencies)


def test_semantic_release_only_publishes_python_distributions() -> None:
    """Release assets must not include the system packaging tree under dist/."""
    with Path("pyproject.toml").open("rb") as stream:
        release_config = tomllib.load(stream)["tool"]["semantic_release"]

    assert "--outdir build/pypi" in release_config["build_command"]
    assert release_config["publish"]["dist_glob_patterns"] == [
        "build/pypi/*.whl",
        "build/pypi/*.tar.gz",
    ]


def test_cli_reports_runtime_version() -> None:
    """The CLI must not contain an independently maintained version string."""
    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output
