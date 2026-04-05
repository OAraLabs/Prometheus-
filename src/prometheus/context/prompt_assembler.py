# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/prompts/context.py
# License: MIT
# Modified: renamed imports to prometheus.*; restructured as a standalone
#           build_runtime_system_prompt() function with static/dynamic boundary

"""Runtime prompt assembler for Prometheus.

Combines the static system prompt (identity + environment + tool schemas) with
dynamic per-turn content (PROMETHEUS.md, memory pointers, user model, task
state, loaded skills) separated by the ``SYSTEM_PROMPT_DYNAMIC_BOUNDARY``
marker.
"""

from __future__ import annotations

from prometheus.context.environment import get_environment_info
from prometheus.context.prometheusmd import load_prometheus_md_prompt
from prometheus.context.system_prompt import (
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    build_system_prompt,
)


def build_runtime_system_prompt(
    *,
    cwd: str,
    config: dict | None = None,
    memory_content: str = "",
    skills: list | None = None,
    task_state: str = "",
    loaded_skill_content: str = "",
) -> str:
    """Assemble the full runtime system prompt.

    The result is structured as::

        <static section>
        --- SYSTEM_PROMPT_DYNAMIC_BOUNDARY ---
        <dynamic section>

    Parameters
    ----------
    cwd:
        Current working directory — used for environment detection and
        PROMETHEUS.md discovery.
    config:
        Optional configuration dict.  Recognised keys:

        - ``"system_prompt"`` — custom base prompt (replaces default identity).
        - ``"fast_mode"``     — if truthy, adds a fast-mode hint.
        - ``"effort"``        — reasoning effort level label.
        - ``"passes"``        — reasoning pass count label.
    memory_content:
        Pre-formatted memory content to inject (e.g. from MemoryPointer).
    skills:
        List of skill dicts with ``"name"`` and ``"description"`` keys.
    task_state:
        Serialised task-tracking state (e.g. todo list) to inject.
    loaded_skill_content:
        Content from a loaded skill that should appear in the dynamic section.
    """
    if skills is None:
        skills = []
    if config is None:
        config = {}

    # ------------------------------------------------------------------
    # Static section
    # ------------------------------------------------------------------
    custom_prompt = config.get("system_prompt")
    static_prompt = build_system_prompt(custom_prompt=custom_prompt, cwd=cwd)

    # Tool schema injection point — callers can embed schemas into the
    # static section by including them in the custom_prompt.  The base
    # build_system_prompt already appends environment info.

    static_sections = [static_prompt]

    # ------------------------------------------------------------------
    # Dynamic section
    # ------------------------------------------------------------------
    dynamic_sections: list[str] = []

    # Session mode hints
    if config.get("fast_mode"):
        dynamic_sections.append(
            "# Session Mode\n"
            "Fast mode is enabled. Prefer concise replies, minimal tool use, "
            "and quicker progress over exhaustive exploration."
        )

    effort = config.get("effort", "standard")
    passes = config.get("passes", 1)
    dynamic_sections.append(
        "# Reasoning Settings\n"
        f"- Effort: {effort}\n"
        f"- Passes: {passes}\n"
        "Adjust depth and iteration count to match these settings while "
        "still completing the task."
    )

    # Skills listing
    if skills:
        lines = [
            "# Available Skills",
            "",
            "The following skills are available via the `skill` tool. "
            "When a user's request matches a skill, invoke it with "
            '`skill(name="<skill_name>")` to load detailed instructions '
            "before proceeding.",
            "",
        ]
        for skill in skills:
            name = skill.get("name", "unknown")
            desc = skill.get("description", "")
            lines.append(f"- **{name}**: {desc}")
        dynamic_sections.append("\n".join(lines))

    # PROMETHEUS.md project instructions
    prometheus_md = load_prometheus_md_prompt(cwd)
    if prometheus_md:
        dynamic_sections.append(prometheus_md)

    # Memory content
    if memory_content:
        dynamic_sections.append(f"# Memory\n\n{memory_content}")

    # Task state
    if task_state:
        dynamic_sections.append(f"# Current Task State\n\n{task_state}")

    # Loaded skill content
    if loaded_skill_content:
        dynamic_sections.append(
            f"# Loaded Skill\n\n{loaded_skill_content}"
        )

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    static_part = "\n\n".join(s for s in static_sections if s.strip())
    dynamic_part = "\n\n".join(s for s in dynamic_sections if s.strip())

    return f"{static_part}\n\n{SYSTEM_PROMPT_DYNAMIC_BOUNDARY}\n\n{dynamic_part}"
