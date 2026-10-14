"""Skills framework for discoverable agent skills and task recipes."""

from __future__ import annotations

from .defaults import create_default_skill_registry
from .models import SkillDefinition, SkillRegistry

__all__ = [
    "SkillDefinition",
    "SkillRegistry",
    "create_default_skill_registry",
]
