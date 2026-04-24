"""boxBot skills package — filesystem-based skill loader.

Public API:
    SkillMeta          — frontmatter-only record of a skill
    discover_skills()  — scan the repo-level ``skills/`` directory
    get_skill_index()  — markdown block for system-prompt injection
    load_skill()       — read full body or a sub-file of a skill
"""

from boxbot.skills.loader import (
    SkillMeta,
    discover_skills,
    get_skill_index,
    load_skill,
)

__all__ = [
    "SkillMeta",
    "discover_skills",
    "get_skill_index",
    "load_skill",
]
