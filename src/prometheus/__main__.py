"""Prometheus CLI entry point.

Provides interactive and one-shot modes for running the agent loop,
plus a ``daemon`` subcommand that delegates to ``scripts/daemon.py``.

Usage:
    prometheus                          # interactive REPL
    prometheus --once "List files"      # single query, then exit
    prometheus daemon                   # start always-on daemon
    prometheus daemon --telegram-only   # daemon with Telegram only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

import yaml

from prometheus import __version__
from prometheus.config.paths import get_config_dir, get_data_dir, get_logs_dir
from prometheus.engine.agent_loop import AgentLoop, RunResult, run_loop, LoopContext
from prometheus.engine.messages import ConversationMessage
from prometheus.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import ModelProvider

log = logging.getLogger("prometheus")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROMETHEUS_YAML = Path(__file__).resolve().parents[2] / "config" / "prometheus.yaml"


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load prometheus.yaml and return the parsed dict."""
    path = Path(config_path) if config_path else _PROMETHEUS_YAML
    if not path.exists():
        alt = get_config_dir() / "prometheus.yaml"
        if alt.exists():
            path = alt
        else:
            log.debug("No config file found — using defaults")
            return {}
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def _detect_model_or_fallback(base_url: str, config_model: str) -> str:
    """Query /v1/models to discover the loaded model; fall back to config value."""
    import httpx as _httpx
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        resp = _httpx.get(url, timeout=10.0)
        resp.raise_for_status()
        models = resp.json().get("data", [])
        if models:
            detected = models[0].get("id")
            if detected:
                log.info("Detected loaded model: %s", detected)
                return detected
    except Exception as exc:
        log.warning("Could not detect model from %s: %s", url, exc)
    log.info("Using model name from config: %s", config_model)
    return config_model


def create_provider(model_cfg: dict[str, Any]) -> tuple[ModelProvider, str]:
    """Instantiate the model provider from config.  Returns (provider, model_name)."""
    provider_name = model_cfg.get("provider", "llama_cpp")
    model_name = model_cfg.get("model", "qwen3.5-32b")
    base_url = model_cfg.get("base_url", "http://localhost:8080")

    if provider_name == "llama_cpp":
        from prometheus.providers.llama_cpp import LlamaCppProvider
        return LlamaCppProvider(base_url=base_url), model_name

    if provider_name == "ollama":
        from prometheus.providers.ollama import OllamaProvider
        url = model_cfg.get("fallback_url", "http://localhost:11434")
        return OllamaProvider(base_url=url), model_name

    if provider_name == "anthropic":
        from prometheus.providers.anthropic import AnthropicProvider
        return AnthropicProvider(), model_name

    # Fallback — treat as llama_cpp-compatible
    from prometheus.providers.llama_cpp import LlamaCppProvider
    log.warning("Unknown provider %r — falling back to llama_cpp", provider_name)
    return LlamaCppProvider(base_url=base_url), model_name


# ---------------------------------------------------------------------------
# Tool registry factory
# ---------------------------------------------------------------------------

def create_tool_registry(security_cfg: dict[str, Any]) -> Any:
    """Build the default tool registry with all builtin tools."""
    from prometheus.tools.base import ToolRegistry
    from prometheus.tools.builtin import (
        AskUserTool,
        BashTool,
        DashboardTool,
        FileEditTool,
        FileReadTool,
        FileWriteTool,
        GlobTool,
        GrepTool,
        LCMDescribeTool,
        LCMExpandTool,
        LCMGrepTool,
        MessageTool,
        NotebookEditTool,
        TTSTool,
        WebFetchTool,
        WebSearchTool,
    )
    from prometheus.tools.builtin.cron_create import CronCreateTool
    from prometheus.tools.builtin.cron_delete import CronDeleteTool
    from prometheus.tools.builtin.cron_list import CronListTool
    from prometheus.tools.builtin.lcm_expand_query import LCMExpandQueryTool

    workspace = security_cfg.get("workspace_root")
    registry = ToolRegistry()
    for tool in [
        BashTool(workspace=workspace),
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        GrepTool(),
        GlobTool(),
        CronCreateTool(),
        CronDeleteTool(),
        CronListTool(),
        LCMDescribeTool(),
        LCMExpandTool(),
        LCMGrepTool(),
        LCMExpandQueryTool(),
        # --- New tools (httpx/stdlib only) ---
        WebSearchTool(),
        WebFetchTool(),
        MessageTool(),
        TTSTool(),
        DashboardTool(),
        NotebookEditTool(),
        AskUserTool(),
    ]:
        registry.register(tool)

    # Optional tools — don't fail if deps missing
    try:
        from prometheus.tools.builtin.skill import SkillTool
        registry.register(SkillTool())
    except Exception:
        pass
    try:
        from prometheus.tools.builtin.todo_write import TodoWriteTool
        registry.register(TodoWriteTool())
    except Exception:
        pass

    # Browser — requires optional playwright dependency
    try:
        from prometheus.tools.builtin.browser import BrowserTool
        registry.register(BrowserTool())
    except Exception:
        pass

    # Session tools — require task manager
    try:
        from prometheus.tools.builtin.sessions_list import SessionsListTool
        from prometheus.tools.builtin.sessions_send import SessionsSendTool
        from prometheus.tools.builtin.sessions_spawn import SessionsSpawnTool
        registry.register(SessionsListTool())
        registry.register(SessionsSendTool())
        registry.register(SessionsSpawnTool())
    except Exception:
        pass

    return registry


# ---------------------------------------------------------------------------
# Adapter + Security
# ---------------------------------------------------------------------------

def create_adapter(model_cfg: dict[str, Any]):
    """Create the model adapter layer (Sprint 3)."""
    from prometheus.adapter import ModelAdapter
    from prometheus.adapter.formatter import QwenFormatter, AnthropicFormatter

    provider_name = model_cfg.get("provider", "llama_cpp")
    if provider_name in ("llama_cpp", "ollama"):
        return ModelAdapter(formatter=QwenFormatter(), strictness="MEDIUM")
    return ModelAdapter(formatter=AnthropicFormatter(), strictness="NONE")


def create_security_gate(security_cfg: dict[str, Any]):
    """Create the permission checker (Sprint 4)."""
    from prometheus.permissions.checker import SecurityGate
    return SecurityGate(
        mode=security_cfg.get("permission_mode", "default"),
        workspace_root=security_cfg.get("workspace_root"),
        denied_commands=security_cfg.get("denied_commands"),
        denied_paths=security_cfg.get("denied_paths"),
    )


# ---------------------------------------------------------------------------
# LCM wiring
# ---------------------------------------------------------------------------

def create_lcm_engine(provider: ModelProvider):
    """Create and wire the LCM engine + tools."""
    try:
        from prometheus.memory.lcm_engine import LCMEngine
        from prometheus.tools.builtin.lcm_grep import set_lcm_engine
        engine = LCMEngine(provider)
        set_lcm_engine(engine)
        return engine
    except Exception as exc:
        log.warning("LCM engine not available: %s", exc)
        return None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def build_system_prompt(config: dict[str, Any]) -> str:
    """Assemble the full system prompt."""
    try:
        from prometheus.context.prompt_assembler import build_runtime_system_prompt
        return build_runtime_system_prompt(cwd=str(Path.cwd()), config=config)
    except Exception:
        return config.get("gateway", {}).get(
            "system_prompt",
            "You are Prometheus, a sovereign AI agent. Be concise and helpful.",
        )


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

async def run_interactive(
    context: LoopContext,
    lcm_engine: Any | None,
    session_id: str,
) -> None:
    """Run an interactive conversation loop with streaming output."""
    messages: list[ConversationMessage] = []
    turn_index = 0

    print(f"Prometheus {__version__} — interactive mode")
    print(f"Model: {context.model} | Provider: type(provider)")
    print("Type your message (Ctrl+D or 'exit' to quit)\n")

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "/exit", "/quit"):
            print("Goodbye.")
            break

        # Ingest to LCM
        if lcm_engine:
            await lcm_engine.ingest(session_id, "user", user_input, turn_index=turn_index)

        messages.append(ConversationMessage.from_user_text(user_input))

        # Stream the response
        response_text = ""
        try:
            async for event, usage in run_loop(context, messages):
                if isinstance(event, AssistantTextDelta):
                    print(event.text, end="", flush=True)
                    response_text += event.text
                elif isinstance(event, ToolExecutionStarted):
                    print(f"\n[tool] {event.tool_name}...", flush=True)
                elif isinstance(event, ToolExecutionCompleted):
                    status = "error" if event.is_error else "ok"
                    output_preview = event.output[:200] if event.output else ""
                    print(f"[tool] {event.tool_name} -> {status}", flush=True)
                    if output_preview:
                        print(f"  {output_preview}", flush=True)
                elif isinstance(event, AssistantTurnComplete):
                    if event.message.text and not response_text:
                        print(event.message.text, end="", flush=True)
                        response_text = event.message.text
        except Exception as exc:
            print(f"\n[error] {exc}")
            # Remove the last user message so we can retry
            if messages and messages[-1].role == "user":
                messages.pop()
            continue

        print()  # newline after response

        # Ingest assistant response to LCM
        if lcm_engine and response_text:
            await lcm_engine.ingest(
                session_id, "assistant", response_text, turn_index=turn_index
            )
            await lcm_engine.maybe_compact(session_id)

        turn_index += 1


# ---------------------------------------------------------------------------
# One-shot mode
# ---------------------------------------------------------------------------

async def run_once(context: LoopContext, query: str) -> None:
    """Run a single query and print the result."""
    messages = [ConversationMessage.from_user_text(query)]
    response_text = ""

    async for event, usage in run_loop(context, messages):
        if isinstance(event, AssistantTextDelta):
            print(event.text, end="", flush=True)
            response_text += event.text
        elif isinstance(event, ToolExecutionStarted):
            print(f"\n[tool] {event.tool_name}...", end="", flush=True)
        elif isinstance(event, ToolExecutionCompleted):
            status = "error" if event.is_error else "ok"
            print(f" {status}", flush=True)
        elif isinstance(event, AssistantTurnComplete):
            if event.message.text and not response_text:
                print(event.message.text, end="", flush=True)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Prometheus CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="prometheus",
        description="Prometheus — sovereign AI agent harness",
    )
    parser.add_argument(
        "--version", action="version", version=f"Prometheus {__version__}"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to prometheus.yaml config file",
    )
    parser.add_argument(
        "--once", type=str, default=None, metavar="QUERY",
        help="Run a single query then exit (non-interactive)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override model name from config",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Override provider (llama_cpp, ollama, anthropic)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--setup", action="store_true",
        help="Run first-time setup wizard",
    )
    parser.add_argument(
        "--setup-gateway-only", action="store_true",
        help="Add or change gateway only (skip model provider setup)",
    )

    subparsers = parser.add_subparsers(dest="command")
    daemon_parser = subparsers.add_parser("daemon", help="Start always-on daemon")
    daemon_parser.add_argument(
        "--telegram-only", action="store_true",
        help="Only start Telegram adapter",
    )

    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    # Setup wizard — runs before anything else
    if args.setup or args.setup_gateway_only:
        from prometheus.setup_wizard import SetupWizard
        wizard = SetupWizard(gateway_only=args.setup_gateway_only)
        success = wizard.run()
        sys.exit(0 if success else 1)

    # Daemon subcommand — delegate to scripts/daemon.py
    if args.command == "daemon":
        from scripts.daemon import main as daemon_main
        # Re-inject args so daemon sees them
        sys.argv = ["prometheus-daemon"]
        if args.config:
            sys.argv.extend(["--config", args.config])
        if args.telegram_only:
            sys.argv.append("--telegram-only")
        if args.debug:
            sys.argv.append("--debug")
        daemon_main()
        return

    # Load config — hint about setup wizard if no config exists
    config = load_config(args.config)
    if not config and not _PROMETHEUS_YAML.exists() and not (get_config_dir() / "prometheus.yaml").exists():
        print("No configuration found. Run the setup wizard:\n")
        print("  python3 -m prometheus --setup\n")
        print("Or create config/prometheus.yaml manually.")
        sys.exit(1)
    model_cfg = config.get("model", {})
    security_cfg = config.get("security", {})

    # Apply CLI overrides
    if args.model:
        model_cfg["model"] = args.model
    if args.provider:
        model_cfg["provider"] = args.provider

    # Build components
    provider, model_name = create_provider(model_cfg)

    # Detect actual loaded model from the server (falls back to config)
    if model_cfg.get("provider", "llama_cpp") in ("llama_cpp",):
        model_name = _detect_model_or_fallback(
            model_cfg.get("base_url", "http://localhost:8080"), model_name,
        )
        model_cfg["model"] = model_name

    registry = create_tool_registry(security_cfg)
    adapter = create_adapter(model_cfg)
    security_gate = create_security_gate(security_cfg)
    lcm_engine = create_lcm_engine(provider)
    system_prompt = build_system_prompt(config)

    # Telemetry (optional)
    telemetry = None
    if config.get("infrastructure", {}).get("telemetry_enabled", True):
        try:
            from prometheus.telemetry.tracker import ToolCallTelemetry
            telemetry = ToolCallTelemetry()
        except Exception:
            pass

    context = LoopContext(
        provider=provider,
        model=model_name,
        system_prompt=system_prompt,
        max_tokens=4096,
        tool_registry=registry,
        permission_checker=security_gate,
        adapter=adapter,
        telemetry=telemetry,
    )

    # Generate session ID
    import uuid
    session_id = f"cli-{uuid.uuid4().hex[:8]}"

    if args.once:
        asyncio.run(run_once(context, args.once))
    else:
        asyncio.run(run_interactive(context, lcm_engine, session_id))

    # Cleanup
    if lcm_engine:
        lcm_engine.close()


if __name__ == "__main__":
    main()
