"""Shared command handlers for gateway adapters (Telegram, Slack).

Platform-agnostic command logic. Each function returns a string
that the adapter sends via its own transport.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prometheus.tools.base import ToolRegistry


def cmd_help() -> str:
    """Return help text listing available commands."""
    return (
        "Prometheus — Sovereign AI Agent\n"
        "\n"
        "Commands:\n"
        "/status    — Model, uptime, tools, memory, SENTINEL\n"
        "/model     — Current model name and provider\n"
        "/wiki      — Wiki stats and recent entries\n"
        "/sentinel  — SENTINEL subsystem status\n"
        "/benchmark — Run a quick smoke test\n"
        "/context   — Context window usage\n"
        "/skills    — List available skills\n"
        "/reset     — Clear conversation context\n"
        "/help      — This message\n"
        "\n"
        "Send any message to chat with the agent."
    )


def cmd_model(model_name: str, model_provider: str) -> str:
    """Return model info text."""
    name = model_name or "(unknown)"
    provider = model_provider or "(unknown)"
    return f"Model: {name}\nProvider: {provider}"


def cmd_status(
    model_name: str,
    model_provider: str,
    start_time: float,
    tool_registry: ToolRegistry,
) -> str:
    """Return full status text."""
    lines: list[str] = ["Prometheus Status\n"]

    lines.append(f"Model: {model_name or '(unknown)'}")
    lines.append(f"Provider: {model_provider or '(unknown)'}")

    if start_time:
        elapsed = int(time.monotonic() - start_time)
        h, remainder = divmod(elapsed, 3600)
        m, s = divmod(remainder, 60)
        lines.append(f"Uptime: {h}h {m}m {s}s")

    lines.append(f"Tools: {len(tool_registry.list_tools())}")

    # Memory stats
    try:
        from prometheus.tools.builtin.wiki_compile import _memory_store

        if _memory_store is not None:
            facts = _memory_store.get_all_memories(limit=10000)
            lines.append(f"Memory facts: {len(facts)}")
        else:
            lines.append("Memory: not initialized")
    except Exception:
        lines.append("Memory: unavailable")

    # SENTINEL state
    try:
        from prometheus.tools.builtin.sentinel_status import (
            _autodream,
            _observer,
        )

        if _observer is not None and _autodream is not None:
            state = (
                "dreaming"
                if _autodream.dreaming
                else ("active" if _observer.started else "idle")
            )
            lines.append(f"\nSENTINEL: {state}")
            lines.append(f"Dream cycles: {_autodream.cycle_count}")
            if _autodream.last_results:
                lines.append("Last dream results:")
                for r in _autodream.last_results:
                    status = "OK" if not r.error else f"FAIL: {r.error}"
                    lines.append(f"  {r.phase}: {status} ({r.duration_seconds:.1f}s)")
        else:
            lines.append("\nSENTINEL: not initialized")
    except Exception:
        lines.append("\nSENTINEL: unavailable")

    return "\n".join(lines)


def cmd_wiki() -> str:
    """Return wiki stats text."""
    wiki_index = Path.home() / ".prometheus" / "wiki" / "index.md"
    if not wiki_index.exists():
        return "Wiki: no index found at ~/.prometheus/wiki/index.md"

    try:
        content = wiki_index.read_text(encoding="utf-8")
        entries: list[str] = []
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("- ["):
                entries.append(stripped)

        lines = [f"Wiki: {len(entries)} pages"]

        mtime = wiki_index.stat().st_mtime
        from datetime import datetime, timezone

        updated = datetime.fromtimestamp(mtime, tz=timezone.utc)
        lines.append(f"Last updated: {updated.strftime('%Y-%m-%d %H:%M UTC')}")

        if entries:
            lines.append("\nRecent entries:")
            for entry in entries[-5:]:
                lines.append(f"  {entry}")

        return "\n".join(lines)
    except Exception as exc:
        return f"Wiki: error reading index — {exc}"


def cmd_sentinel() -> str:
    """Return SENTINEL subsystem status text."""
    try:
        from prometheus.sentinel.signals import SignalBus
        from prometheus.tools.builtin.sentinel_status import (
            _autodream,
            _observer,
            _signal_bus,
        )
    except ImportError:
        return "SENTINEL module not available."

    if _signal_bus is None or _observer is None or _autodream is None:
        return "SENTINEL not initialized. Is the daemon running with sentinel enabled?"

    lines: list[str] = ["SENTINEL Status\n"]

    idle_secs = int(time.time() - _observer.last_activity)
    lines.append("Observer:")
    lines.append(f"  Active: {_observer.started}")
    lines.append(f"  Last activity: {idle_secs}s ago")
    lines.append(f"  Pending nudges: {len(_observer.pending_nudges)}")

    lines.append("\nAutoDream Engine:")
    lines.append(f"  Dreaming: {_autodream.dreaming}")
    lines.append(f"  Cycles completed: {_autodream.cycle_count}")
    if _autodream.last_cycle_time:
        ago = int(time.time() - _autodream.last_cycle_time)
        lines.append(f"  Last cycle: {ago}s ago")

    lines.append("\nSignal Bus:")
    lines.append(f"  Total signals: {_signal_bus.signal_count}")
    lines.append(f"  Subscribers: {_signal_bus.subscriber_count}")

    recent = _signal_bus.recent(limit=10)
    if recent:
        lines.append("\nRecent Signals:")
        for sig in recent:
            ago = int(time.time() - sig.timestamp)
            lines.append(f"  [{sig.kind}] from {sig.source} ({ago}s ago)")

    if _autodream.last_results:
        lines.append("\nLast Dream Cycle:")
        for r in _autodream.last_results:
            status = "OK" if not r.error else f"FAIL: {r.error}"
            lines.append(f"  {r.phase}: {status} ({r.duration_seconds:.1f}s)")
            for k, v in r.summary.items():
                lines.append(f"    {k}: {v}")

    if _observer.pending_nudges:
        lines.append("\nPending Nudges:")
        for nudge in _observer.pending_nudges[:5]:
            lines.append(f"  [{nudge.nudge_type}] {nudge.message[:80]}")

    return "\n".join(lines)


def cmd_context(system_prompt: str, model_name: str) -> str:
    """Return context window usage text."""
    from prometheus.context.token_estimation import estimate_tokens

    try:
        from prometheus.context.budget import TokenBudget

        budget = TokenBudget.from_config(model=model_name)
        effective_limit = budget.effective_limit
        reserved_output = budget.reserved_output
    except Exception:
        effective_limit = 24000
        reserved_output = 2000

    prompt_tokens = estimate_tokens(system_prompt)
    available = effective_limit - reserved_output
    headroom = max(0, available - prompt_tokens)
    usage_pct = (prompt_tokens / available * 100) if available > 0 else 0

    lines = [
        "Context Window\n",
        f"Window size:    {effective_limit:,} tokens",
        f"Reserved output: {reserved_output:,} tokens",
        f"Available:       {available:,} tokens",
        "",
        f"System prompt:   {prompt_tokens:,} tokens ({usage_pct:.0f}%)",
        f"Headroom:        {headroom:,} tokens",
        "",
        f"Model: {model_name or '(unknown)'}",
    ]

    bar_len = 20
    filled = round(usage_pct / 100 * bar_len)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)
    lines.append(f"[{bar}] {usage_pct:.0f}% used")

    return "\n".join(lines)


def cmd_skills() -> str:
    """Return available skills list text."""
    try:
        from prometheus.skills.loader import load_skill_registry

        registry = load_skill_registry()
        skills = registry.list_skills()
    except Exception as exc:
        return f"Skills: error loading registry — {exc}"

    if not skills:
        return "No skills available."

    lines = [f"Skills ({len(skills)})\n"]
    for skill in skills:
        source_tag = f" [{skill.source}]" if skill.source else ""
        lines.append(f"  {skill.name}{source_tag}")
        if skill.description:
            lines.append(f"    {skill.description[:80]}")

    lines.append("\nUse the skill tool to load a skill by name.")
    return "\n".join(lines)
