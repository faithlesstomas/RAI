"""Data models and registry for discoverable agent skills."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from returns.result import Failure, Result, Success
import yaml


class SkillDefinition(BaseModel):
    """Declarative specification of an agent skill."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    description: str = Field(min_length=1)
    capabilities: tuple[str, ...] = Field(default_factory=tuple)
    when_to_use: str = Field(default="")
    instructions: str = Field(default="")
    source_path: Path | None = None

    def format_for_prompt(self) -> str:
        """Format skill into concise prompt guidance."""
        caps_str = ", ".join(self.capabilities) if self.capabilities else "none"
        body = f"### Skill: {self.name}\nDescription: {self.description}\nRequired capabilities: {caps_str}"
        if self.when_to_use:
            body += f"\nWhen to use: {self.when_to_use}"
        if self.instructions:
            body += f"\nInstructions:\n{self.instructions}"
        return body


class SkillRegistry:
    """In-memory registry and file loader for agent skills."""

    def __init__(self, skills: list[SkillDefinition] | None = None) -> None:
        self._skills: dict[str, SkillDefinition] = {s.name: s for s in (skills or [])}

    def register(self, skill: SkillDefinition) -> None:
        """Register or overwrite a skill definition."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> SkillDefinition | None:
        """Retrieve a skill by name."""
        return self._skills.get(name)

    def all(self) -> tuple[SkillDefinition, ...]:
        """Return all registered skills."""
        return tuple(self._skills.values())

    @classmethod
    def parse_markdown(  # noqa: PLR0911
        cls, text: str, source_path: Path | None = None
    ) -> Result[SkillDefinition, str]:
        """Parse a markdown skill file with YAML frontmatter."""
        text = text.strip()
        if not text.startswith("---"):
            return Failure("Skill markdown must start with '---' YAML frontmatter")

        min_parts = 3
        parts = text.split("---", 2)
        if len(parts) < min_parts:
            return Failure("Skill markdown missing closing '---' for YAML frontmatter")

        frontmatter_str = parts[1]
        body = parts[2].strip()

        try:
            frontmatter: dict[str, Any] = yaml.safe_load(frontmatter_str) or {}
        except Exception as e:
            return Failure(f"Invalid YAML frontmatter: {e}")

        name = frontmatter.get("name")
        if not name or not isinstance(name, str):
            return Failure("Frontmatter missing required string field 'name'")

        description = frontmatter.get("description", "")
        if not description:
            return Failure("Frontmatter missing required string field 'description'")

        caps_raw = frontmatter.get("capabilities", [])
        if isinstance(caps_raw, list):
            capabilities = tuple(str(c) for c in caps_raw)
        elif isinstance(caps_raw, str):
            capabilities = (caps_raw,)
        else:
            capabilities = ()

        when_to_use = str(frontmatter.get("when_to_use", ""))
        instructions = body

        try:
            skill = SkillDefinition(
                name=name,
                description=description,
                capabilities=capabilities,
                when_to_use=when_to_use,
                instructions=instructions,
                source_path=source_path,
            )
            return Success(skill)
        except Exception as e:
            return Failure(f"Validation failed for skill '{name}': {e}")

    def load_file(self, path: Path) -> Result[SkillDefinition, str]:
        """Read and parse a skill from a markdown file."""
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            return Failure(f"Could not read skill file '{path}': {e}")

        res = self.parse_markdown(content, source_path=path)
        if isinstance(res, Success):
            self.register(res.unwrap())
        return res

    def discover_directory(self, directory: Path) -> list[SkillDefinition]:
        """Discover all *.md and */SKILL.md skill files in a directory."""
        discovered: list[SkillDefinition] = []
        if not directory.is_dir():
            return discovered

        for md_file in directory.rglob("*.md"):
            if md_file.is_file():
                res = self.load_file(md_file)
                if isinstance(res, Success):
                    discovered.append(res.unwrap())
        return discovered

    def format_skills_instruction(self, available_capabilities: tuple[str, ...] | None = None) -> str:
        """Format matching skills as a prompt section for system instruction or planner."""
        matched: list[SkillDefinition] = []
        for skill in self._skills.values():
            if available_capabilities is None:
                matched.append(skill)
            elif not skill.capabilities or any(cap in available_capabilities for cap in skill.capabilities):
                matched.append(skill)

        if not matched:
            return ""

        sections = ["\nAvailable Skills & Task Recipes:"]
        for s in matched:
            sections.append(s.format_for_prompt())
        return "\n\n".join(sections)
