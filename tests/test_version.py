"""Version consistency tests."""

from importlib.metadata import version

from click.testing import CliRunner

from rai import __version__
from rai.cli import cli


def test_distribution_and_runtime_versions_match() -> None:
    """Installed metadata and the importable package expose one version."""
    assert version("rai") == __version__


def test_cli_reports_runtime_version() -> None:
    """The CLI must not contain an independently maintained version string."""
    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert __version__ in result.output
