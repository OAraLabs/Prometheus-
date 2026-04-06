#!/usr/bin/env python3
"""Prometheus daemon — main entry point for always-on operation.

Source: Novel code for Prometheus Sprint 6.
Starts Telegram adapter, cron scheduler, heartbeat, and memory extractor.
Signal handling for graceful shutdown.

Usage:
    python -m prometheus.scripts.daemon --telegram-only --debug
    python scripts/daemon.py --config config/prometheus.yaml
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

from prometheus.config.paths import get_config_dir, get_logs_dir
from prometheus.engine.agent_loop import AgentLoop
from prometheus.gateway.archive_writer import ArchiveWriter
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.cron_scheduler import run_scheduler_loop
from prometheus.gateway.heartbeat import Heartbeat
from prometheus.gateway.telegram import TelegramAdapter
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.__main__ import create_tool_registry
from prometheus.tools.base import ToolRegistry

logger = logging.getLogger("prometheus.daemon")

DEFAULT_SYSTEM_PROMPT = (
    "You are Prometheus, a sovereign AI agent. You have access to tools for "
    "file operations, shell commands, and cron job management. Be concise and helpful."
)


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load prometheus.yaml configuration."""
    if config_path:
        path = Path(config_path)
    else:
        path = Path("config/prometheus.yaml")
        if not path.exists():
            path = get_config_dir() / "prometheus.yaml"

    if not path.exists():
        logger.warning("Config file not found at %s, using defaults", path)
        return {}

    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_tool_registry(security_cfg: dict[str, Any] | None = None) -> ToolRegistry:
    """Create the tool registry with all builtin tools (same as CLI).

    Reuses create_tool_registry() from __main__ so daemon and CLI
    always have the same tool set.
    """
    if security_cfg is None:
        security_cfg = {}
    registry = create_tool_registry(security_cfg)

    # Add wiki tools (daemon-specific, not in CLI)
    try:
        from prometheus.tools.builtin.wiki_compile import WikiCompileTool
        from prometheus.tools.builtin.wiki_query import WikiQueryTool
        registry.register(WikiCompileTool())
        registry.register(WikiQueryTool())
    except Exception:
        pass

    return registry


async def run_daemon(args: argparse.Namespace) -> None:
    """Main async entry point — start all subsystems."""
    config = load_config(args.config)
    model_config = config.get("model", {})
    gateway_config = config.get("gateway", {})
    security_config = config.get("security", {})

    # Archive writer
    archive = ArchiveWriter()
    archive.archive_event("daemon_start", {"args": vars(args)})

    # Model provider — use LlamaCppProvider from Sprint 3
    provider = LlamaCppProvider(
        base_url=model_config.get("base_url", "http://localhost:8080"),
    )

    # Detect actual loaded model from the server (falls back to config)
    config_model = model_config.get("model", "qwen3.5-32b")
    detected = await provider.detect_loaded_model()
    model_name = detected or config_model
    if detected:
        model_config["model"] = detected
    else:
        logger.info("Using model name from config: %s", config_model)

    # Tool registry — same tools as CLI mode
    registry = build_tool_registry(security_cfg=security_config)

    # Agent loop
    agent_loop = AgentLoop(
        provider=provider,
        model=model_name,
        tool_registry=registry,
    )

    # Collect async tasks to run
    tasks: list[asyncio.Task] = []
    shutdown_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    # Telegram adapter
    telegram: TelegramAdapter | None = None
    telegram_token = gateway_config.get("telegram_token", "") or os.environ.get("PROMETHEUS_TELEGRAM_TOKEN", "")
    if telegram_token and gateway_config.get("telegram_enabled", True):
        tg_config = PlatformConfig(
            platform=Platform.TELEGRAM,
            token=telegram_token,
            allowed_chat_ids=gateway_config.get("allowed_chat_ids", []),
            proxy_url=gateway_config.get("proxy_url"),
        )
        from prometheus.context.prompt_assembler import build_runtime_system_prompt
        system_prompt = build_runtime_system_prompt(
            cwd=str(Path.cwd()), config=config,
        )
        telegram = TelegramAdapter(
            config=tg_config,
            agent_loop=agent_loop,
            tool_registry=registry,
            system_prompt=system_prompt,
            model_name=model_name,
            model_provider=model_config.get("provider", "llama_cpp"),
        )
        await telegram.start()
        archive.archive_event("telegram_started")
        logger.info("Telegram adapter started")

    # Slack adapter
    slack_adapter = None
    slack_bot_token = gateway_config.get("slack_bot_token", "") or os.environ.get("PROMETHEUS_SLACK_BOT_TOKEN", "")
    slack_app_token = gateway_config.get("slack_app_token", "") or os.environ.get("PROMETHEUS_SLACK_APP_TOKEN", "")
    if slack_bot_token and slack_app_token and gateway_config.get("slack_enabled", False):
        try:
            from prometheus.gateway.slack import SlackAdapter
            slack_config = PlatformConfig(
                platform=Platform.SLACK,
                token=slack_bot_token,
                app_token=slack_app_token,
                allowed_channels=gateway_config.get("slack_channels", []),
            )
            if "system_prompt" not in dir():
                from prometheus.context.prompt_assembler import build_runtime_system_prompt
                system_prompt = build_runtime_system_prompt(
                    cwd=str(Path.cwd()), config=config,
                )
            slack_adapter = SlackAdapter(
                config=slack_config,
                agent_loop=agent_loop,
                tool_registry=registry,
                system_prompt=system_prompt,
                model_name=model_name,
                model_provider=model_config.get("provider", "llama_cpp"),
            )
            await slack_adapter.start()
            archive.archive_event("slack_started")
            logger.info("Slack adapter started (Socket Mode)")
        except ImportError:
            logger.warning(
                "Slack is enabled but slack-bolt is not installed. "
                "Install with: pip install 'prometheus[slack]'"
            )
        except Exception as exc:
            logger.error("Failed to start Slack adapter: %s", exc)

    # Heartbeat
    heartbeat = Heartbeat(gateway=telegram)
    heartbeat_task = asyncio.create_task(heartbeat.run_forever())
    tasks.append(heartbeat_task)

    # Cron scheduler (skip if --telegram-only)
    if not args.telegram_only:
        cron_task = asyncio.create_task(run_scheduler_loop())
        tasks.append(cron_task)
        logger.info("Cron scheduler started")

    # LCM engine (optional, from Sprint 7)
    lcm_engine = None
    try:
        from prometheus.memory.lcm_engine import LCMEngine
        from prometheus.tools.builtin.lcm_grep import set_lcm_engine
        lcm_engine = LCMEngine(provider)
        set_lcm_engine(lcm_engine)
        logger.info("LCM engine initialised")
    except Exception as exc:
        logger.warning("LCM engine not available: %s", exc)

    # Memory extractor (optional, from Sprint 5)
    try:
        from prometheus.memory.extractor import MemoryExtractor
        from prometheus.memory.store import MemoryStore
        from prometheus.memory.wiki_compiler import WikiCompiler
        from prometheus.tools.builtin.wiki_compile import set_wiki_compiler

        memory_store = MemoryStore()

        # Wiki compiler — auto-compiles after each extraction run
        wiki_compiler = WikiCompiler(store=memory_store)
        set_wiki_compiler(wiki_compiler, memory_store)
        logger.info("Wiki compiler initialised at %s", wiki_compiler.wiki_root)

        extractor = MemoryExtractor(
            store=memory_store,
            provider=provider,
            model=model_name,
            post_extract_callback=wiki_compiler.compile,
        )

        # Wire extractor into LCM for pre-compaction flush
        if lcm_engine is not None:
            lcm_engine.set_memory_extractor(extractor)
            logger.info("Memory extractor wired to LCM pre-compaction flush")

        extractor_task = asyncio.create_task(extractor.run_forever())
        tasks.append(extractor_task)
        logger.info("Memory extractor started")
    except Exception as exc:
        logger.warning("Memory extractor not available: %s", exc)

    # Learning loop — SkillCreator (auto-generate skills from successful tasks)
    try:
        from prometheus.learning.skill_creator import SkillCreator
        skill_creator = SkillCreator(provider, model=model_name)
        agent_loop.set_post_task_hook(skill_creator.maybe_create)
        logger.info("SkillCreator wired to agent loop post-task hook")
    except Exception as exc:
        logger.warning("SkillCreator not available: %s", exc)

    # SENTINEL proactive subsystem (Sprint 9)
    sentinel_config = config.get("sentinel", {})
    if sentinel_config.get("enabled", True):
        try:
            from prometheus.sentinel.signals import SignalBus
            from prometheus.sentinel.autodream import AutoDreamEngine
            from prometheus.sentinel.observer import ActivityObserver
            from prometheus.sentinel.wiki_lint import WikiLinter
            from prometheus.sentinel.memory_consolidator import MemoryConsolidator
            from prometheus.sentinel.telemetry_digest import TelemetryDigest
            from prometheus.sentinel.knowledge_synth import KnowledgeSynthesizer
            from prometheus.tools.builtin.sentinel_status import set_sentinel_components
            from prometheus.tools.builtin.wiki_lint_tool import (
                set_wiki_linter as set_lint_wiki_linter,
            )
            from prometheus.tools.builtin.sentinel_status import SentinelStatusTool
            from prometheus.tools.builtin.wiki_lint_tool import WikiLintTool

            signal_bus = SignalBus()

            # Leaf components
            wiki_linter = WikiLinter()
            set_lint_wiki_linter(wiki_linter)

            mem_consolidator = None
            if "memory_store" in dir():
                mem_consolidator = MemoryConsolidator(
                    memory_store,
                    stale_days=sentinel_config.get("stale_threshold_days", 90),
                    decay_rate=sentinel_config.get("confidence_decay_rate", 0.05),
                )

            tel_digest = None
            try:
                from prometheus.telemetry.tracker import ToolCallTelemetry
                telemetry = ToolCallTelemetry()
                tel_digest = TelemetryDigest(
                    telemetry,
                    period_hours=sentinel_config.get("digest_lookback_hours", 24),
                )
            except Exception:
                logger.debug("SENTINEL: telemetry digest not available")

            knowledge_synth = None
            if "memory_store" in dir() and sentinel_config.get("synthesis_enabled", True):
                knowledge_synth = KnowledgeSynthesizer(
                    store=memory_store,
                    provider=provider,
                    model=model_name,
                    budget_tokens=sentinel_config.get("dream_budget_tokens", 2000),
                )

            # Orchestrators
            autodream = AutoDreamEngine(
                signal_bus,
                wiki_linter=wiki_linter,
                memory_consolidator=mem_consolidator,
                telemetry_digest=tel_digest,
                knowledge_synth=knowledge_synth,
                config=sentinel_config,
            )
            observer = ActivityObserver(
                signal_bus,
                gateway=telegram,
                config=sentinel_config,
            )

            # Wire signal bus into existing subsystems
            heartbeat.signal_bus = signal_bus
            if "extractor" in dir():
                extractor.signal_bus = signal_bus

            # Start (signal-reactive, no separate tasks needed)
            await observer.start()
            await autodream.start()

            # Wire tool singletons and register
            set_sentinel_components(signal_bus, observer, autodream)
            registry.register(SentinelStatusTool())
            registry.register(WikiLintTool())

            logger.info("SENTINEL proactive subsystem started")
        except Exception as exc:
            logger.warning("SENTINEL not available: %s", exc)

    logger.info("Prometheus daemon running. Press Ctrl+C to stop.")

    # Wait for shutdown
    await shutdown_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    archive.archive_event("daemon_shutdown")

    if telegram:
        await telegram.stop()

    if slack_adapter:
        await slack_adapter.stop()

    heartbeat.stop()

    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Prometheus daemon stopped")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Prometheus daemon")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to prometheus.yaml"
    )
    parser.add_argument(
        "--telegram-only",
        action="store_true",
        help="Only start Telegram adapter (skip cron scheduler)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_dir = get_logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "daemon.log"),
        ],
    )

    asyncio.run(run_daemon(args))


if __name__ == "__main__":
    main()
