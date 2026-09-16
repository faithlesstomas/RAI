"""Regression coverage for the semantic-release workflow."""

from pathlib import Path
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib


INSERTION_FLAG = "<!-- version list -->"
RELEASE_TAG_RULE = (
    r"$CI_COMMIT_TAG =~ /^v\d+\.\d+\.\d+"
    r"((a|b|rc)\d+|-(alpha|beta|rc)\.\d+)?$/"
)
RELEASE_COMMIT_WORKFLOW_RULE = (
    '$CI_PIPELINE_SOURCE == "push" '
    "&& $CI_COMMIT_BRANCH =~ /^(main|alpha)$/ "
    "&& $CI_COMMIT_AUTHOR =~ /^semantic-release-bot / "
    r"&& $CI_COMMIT_TITLE =~ /^chore\(release\): /"
)
QUALITY_JOB_COUNT = 7
RELEASE_TAG_RULE_COUNT = 5


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
    assert 'glab release upload "$CI_COMMIT_TAG"' in pipeline
    assert "--as-prerelease" not in pipeline
    assert "release_guard.py" not in pipeline
    assert "python-semantic-release==$PYTHON_SEMANTIC_RELEASE_VERSION" in pipeline
    assert "resource_group: release-$CI_COMMIT_REF_NAME" in pipeline


def test_release_commit_branch_pipeline_is_suppressed_without_suppressing_tag() -> None:
    pipeline = Path(".gitlab-ci.yml").read_text(encoding="utf-8")

    assert f"- if: '{RELEASE_COMMIT_WORKFLOW_RULE}'\n      when: never" in pipeline
    assert "    - when: always\n\n" in pipeline
    assert "$CI_COMMIT_BRANCH" in RELEASE_COMMIT_WORKFLOW_RULE


def test_full_quality_matrix_is_not_repeated_for_release_tags() -> None:
    pipeline = Path(".gitlab-ci.yml").read_text(encoding="utf-8")

    quality_template = (
        ".branch_quality_job:\n"
        "  rules:\n"
        "    - if: $CI_COMMIT_TAG\n"
        "      when: never\n"
        "    - when: on_success"
    )
    assert quality_template in pipeline
    assert pipeline.count("  extends: .branch_quality_job") == QUALITY_JOB_COUNT


def test_release_tag_runs_targeted_verification_and_package_build() -> None:
    pipeline = Path(".gitlab-ci.yml").read_text(encoding="utf-8")

    assert "release_verify:\n  stage: test" in pipeline
    assert (
        "uv run pytest --timeout=30 tests/test_version.py "
        "tests/test_release_configuration.py"
    ) in pipeline
    assert pipeline.count(f"- if: {RELEASE_TAG_RULE}") == RELEASE_TAG_RULE_COUNT
    assert "    - if: $CI_COMMIT_TAG\n      when: never\n    - when: on_success" in pipeline


def test_stable_release_tag_republishes_documentation() -> None:
    pipeline = Path(".gitlab-ci.yml").read_text(encoding="utf-8")

    assert "    - if: $CI_COMMIT_TAG =~ /^v\\d+\\.\\d+\\.\\d+$/" in pipeline
    assert "    - job: release_verify\n      optional: true" in pipeline


def test_tag_publish_uses_the_gitlab_package_registry() -> None:
    pipeline = Path(".gitlab-ci.yml").read_text(encoding="utf-8")

    assert "registry.gitlab.com/gitlab-org/cli:latest" in pipeline
    assert "--use-package-registry --package-name rich-ai" in pipeline
    assert "semantic-release --strict publish" not in pipeline
    assert "RELEASE_BRANCH" not in pipeline


def test_release_artifacts_are_built_once_in_the_tag_pipeline() -> None:
    config = _release_config()
    pipeline = Path(".gitlab-ci.yml").read_text(encoding="utf-8")

    assert "build_command" not in config
    assert "uv build --out-dir build/pypi" in pipeline
    assert "semantic-release --strict version --skip-build" in pipeline
