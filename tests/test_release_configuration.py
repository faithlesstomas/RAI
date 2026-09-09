"""Regression coverage for the semantic-release workflow."""

from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib


INSERTION_FLAG = "<!-- version list -->"


def _release_config() -> dict[str, object]:
    with Path("pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["tool"]["semantic_release"]


def test_changelog_update_mode_has_a_committed_insertion_marker() -> None:
    config = _release_config()
    changelog = config["changelog"]

    assert isinstance(changelog, dict)
    assert changelog["mode"] == "update"
    assert changelog["insertion_flag"] == INSERTION_FLAG
    assert "template_dir" not in changelog
    assert changelog["default_templates"]["changelog_file"] == "CHANGELOG.md"
    assert INSERTION_FLAG in Path("CHANGELOG.md").read_text(encoding="utf-8")


def test_release_channels_are_selected_by_branch_configuration() -> None:
    branches = _release_config()["branches"]

    assert branches == {
        "alpha": {
            "match": "^alpha$",
            "prerelease": True,
            "prerelease_token": "alpha",
        },
        "main": {"match": "^main$", "prerelease": False},
    }


def test_ci_has_one_version_producing_release_job() -> None:
    pipeline = Path(".gitlab-ci.yml").read_text(encoding="utf-8")

    assert "release_plan:" in pipeline
    assert "\nrelease:\n" in pipeline
    assert pipeline.count("semantic-release --strict version") == 1
    assert "semantic-release --strict version --skip-build" in pipeline
    assert "semantic-release --noop --strict version --print" in pipeline
    assert "semantic-release --strict publish --tag \"$CI_COMMIT_TAG\"" in pipeline
    assert "--as-prerelease" not in pipeline
    assert "release_guard.py" not in pipeline
    assert "python-semantic-release==$PYTHON_SEMANTIC_RELEASE_VERSION" in pipeline


def test_tag_publish_attaches_head_to_the_release_channel() -> None:
    pipeline = Path(".gitlab-ci.yml").read_text(encoding="utf-8")

    assert 'git checkout -B "$RELEASE_BRANCH" "$CI_COMMIT_SHA"' in pipeline
    assert "RELEASE_BRANCH: main" in pipeline
    assert "RELEASE_BRANCH: alpha" in pipeline
    assert pipeline.count("GIT_STRATEGY: clone") == 2


def test_release_artifacts_are_built_once_in_the_tag_pipeline() -> None:
    config = _release_config()
    pipeline = Path(".gitlab-ci.yml").read_text(encoding="utf-8")

    assert "build_command" not in config
    assert "uv build --out-dir build/pypi" in pipeline
    assert "semantic-release --strict version --skip-build" in pipeline
