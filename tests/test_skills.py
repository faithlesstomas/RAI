"""Unit tests for the rai.skills framework."""

from __future__ import annotations

from pathlib import Path

from returns.result import Failure, Success

from rai.skills.defaults import create_default_skill_registry
from rai.skills.models import SkillDefinition, SkillRegistry


def test_skill_definition_formatting() -> None:
    skill = SkillDefinition(
        name="test-skill",
        description="A test skill for demonstration",
        capabilities=("application.list", "process.inspect"),
        when_to_use="When testing the skills framework.",
        instructions="1. Run tests.\n2. Verify assertions.",
    )
    formatted = skill.format_for_prompt()
    assert "### Skill: test-skill" in formatted
    assert "application.list, process.inspect" in formatted
    assert "When testing the skills framework." in formatted
    assert "1. Run tests." in formatted


def test_skill_registry_markdown_parser() -> None:
    markdown = """---
name: custom-eval
description: Evaluate arbitrary math expressions
capabilities:
  - eval.math
when_to_use: When mathematical reasoning is required
---
Execute eval.math with the given expression.
"""
    res = SkillRegistry.parse_markdown(markdown)
    assert isinstance(res, Success)
    skill = res.unwrap()
    assert skill.name == "custom-eval"
    assert skill.description == "Evaluate arbitrary math expressions"
    assert skill.capabilities == ("eval.math",)
    assert skill.when_to_use == "When mathematical reasoning is required"
    assert "Execute eval.math" in skill.instructions


def test_skill_registry_invalid_markdown() -> None:
    assert isinstance(SkillRegistry.parse_markdown("No frontmatter here"), Failure)
    assert isinstance(SkillRegistry.parse_markdown("---\nname: incomplete\n"), Failure)
    assert isinstance(SkillRegistry.parse_markdown("---\nfoo: bar\n---\nbody"), Failure)


def test_skill_registry_file_discovery(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills"
    skill_dir.mkdir()
    file_a = skill_dir / "skill_a.md"
    file_a.write_text(
        """---
name: skill-a
description: Skill A
capabilities:
  - cap.a
---
Instructions for A
""",
        encoding="utf-8",
    )
    file_b = skill_dir / "skill_b.md"
    file_b.write_text(
        """---
name: skill-b
description: Skill B
capabilities:
  - cap.b
---
Instructions for B
""",
        encoding="utf-8",
    )

    registry = SkillRegistry()
    discovered = registry.discover_directory(skill_dir)
    expected_discovered_count = 2
    assert len(discovered) == expected_discovered_count
    names = {s.name for s in discovered}
    assert names == {"skill-a", "skill-b"}

    prompt = registry.format_skills_instruction(available_capabilities=("cap.a",))
    assert "skill-a" in prompt
    assert "skill-b" not in prompt


def test_default_skill_registry() -> None:
    registry = create_default_skill_registry()
    skills = registry.all()
    min_core_skills = 3
    assert len(skills) >= min_core_skills
    assert registry.get("system-inspection") is not None
    assert registry.get("web-research") is not None
    assert registry.get("file-inspection") is not None
