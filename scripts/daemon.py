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
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin import (
    BashTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    GlobTool,
    GrepTool,
)
from prometheus.tools.builtin.cron_create import CronCreateTool
from prometheus.tools.builtin.cron_delete import CronDeleteTool
from prometheus.tools.builtin.cron_list import CronListTool

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


def build_tool_registry(workspace: str | None = None) -> ToolRegistry:
    """Create the default tool registry with all builtin + cron tools."""
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
    ]:
        registry.register(tool)
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

    # Tool registry
    workspace = security_config.get("workspace_root")
    registry = build_tool_registry(workspace=workspace)

    # Agent loop
    agent_loop = AgentLoop(
        provider=provider,
        model=model_config.get("model", "qwen3.5-32b"),
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
    telegram_token = gateway_config.get("telegram_token", "")
    if telegram_token and gateway_config.get("telegram_enabled", True):
        tg_config = PlatformConfig(
            platform=Platform.TELEGRAM,
            token=telegram_token,
            allowed_chat_ids=gateway_config.get("allowed_chat_ids", []),
            proxy_url=gateway_config.get("proxy_url"),
        )
        telegram = TelegramAdapter(
            config=tg_config,
            agent_loop=agent_loop,
            tool_registry=registry,
            system_prompt=gateway_config.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        )
        await telegram.start()
        archive.archive_event("telegram_started")
        logger.info("Telegram adapter started")

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

        memory_store = MemoryStore()
        extractor = MemoryExtractor(
            store=memory_store,
            provider=provider,
            model=model_config.get("model", "qwen3.5-32b"),
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

    logger.info("Prometheus daemon running. Press Ctrl+C to stop.")

    # Wait for shutdown
    await shutdown_event.wait()

    # Graceful shutdown
    logger.info("Shutting down...")
    archive.archive_event("daemon_shutdown")

    if telegram:
        await telegram.stop()

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
