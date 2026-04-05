"""Telegram platform adapter — full bot using python-telegram-bot.

Source: Novel code for Prometheus Sprint 6 (architecture inspired by Hermes
gateway.platforms.telegram).

Receives messages via long-polling, dispatches to AgentLoop.run_async(),
sends responses back with MarkdownV2 formatting and message chunking.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from telegram import BotCommand, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.platform_base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

if TYPE_CHECKING:
    from prometheus.engine.agent_loop import AgentLoop
    from prometheus.tools.base import ToolRegistry

logger = logging.getLogger(__name__)

# MarkdownV2 special characters that must be escaped
_MARKDOWN_V2_ESCAPE = re.compile(r"([_\*\[\]\(\)~`>#+\-=|{}.!\\])")

# Telegram message length limit
MAX_MESSAGE_LENGTH = 4096


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2."""
    return _MARKDOWN_V2_ESCAPE.sub(r"\\\1", text)


def chunk_message(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Split a long message into chunks respecting Telegram's limit.

    Tries to split at newlines, then at spaces, then hard-truncates.
    """
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break

        # Try to split at a newline
        cut = text.rfind("\n", 0, max_length)
        if cut <= 0:
            # Try to split at a space
            cut = text.rfind(" ", 0, max_length)
        if cut <= 0:
            # Hard truncate
            cut = max_length

        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")

    return chunks


class TelegramAdapter(BasePlatformAdapter):
    """Telegram bot adapter — receives messages, routes to AgentLoop."""

    def __init__(
        self,
        config: PlatformConfig,
        agent_loop: AgentLoop,
        tool_registry: ToolRegistry,
        system_prompt: str = "You are Prometheus, a helpful AI assistant.",
        model_name: str = "",
        model_provider: str = "",
    ) -> None:
        super().__init__(config)
        self.agent_loop = agent_loop
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.model_provider = model_provider
        self._app: Application | None = None
        self._sessions: dict[str, list[dict[str, Any]]] = {}
        self._start_time: float = 0.0

    async def start(self) -> None:
        """Build the telegram Application and start long-polling."""
        if not self.config.token:
            raise ValueError("Telegram bot token is required")

        builder = Application.builder().token(self.config.token)

        # Apply network config if proxy is set
        if self.config.proxy_url:
            builder.proxy(self.config.proxy_url)

        self._app = builder.build()

        # Register handlers
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("clear", self._cmd_clear))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("help", self._cmd_help))
        self._app.add_handler(CommandHandler("reset", self._cmd_reset))
        self._app.add_handler(CommandHandler("model", self._cmd_model))
        self._app.add_handler(CommandHandler("wiki", self._cmd_wiki))
        self._app.add_handler(CommandHandler("sentinel", self._cmd_sentinel))
        self._app.add_handler(CommandHandler("benchmark", self._cmd_benchmark))
        self._app.add_handler(CommandHandler("context", self._cmd_context))
        self._app.add_handler(CommandHandler("skills", self._cmd_skills))
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        self._running = True
        self._start_time = time.monotonic()

        # Register command menu with Telegram
        try:
            await self._app.bot.set_my_commands([
                BotCommand("start", "Check if Prometheus is online"),
                BotCommand("status", "Model, uptime, tools, SENTINEL state"),
                BotCommand("help", "List commands and capabilities"),
                BotCommand("reset", "Clear conversation context"),
                BotCommand("model", "Show current model and provider"),
                BotCommand("wiki", "Wiki stats and recent entries"),
                BotCommand("sentinel", "SENTINEL subsystem status"),
                BotCommand("benchmark", "Run a quick smoke test"),
                BotCommand("context", "Context window usage"),
                BotCommand("skills", "List available skills"),
            ])
        except Exception as exc:
            logger.warning("Failed to register command menu: %s", exc)

        logger.info("Telegram adapter started (polling)")

    async def stop(self) -> None:
        """Graceful shutdown of the Telegram bot."""
        if self._app and self._running:
            self._running = False
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            logger.info("Telegram adapter stopped")

    async def send(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to: int | None = None,
        parse_mode: str | None = None,
    ) -> SendResult:
        """Send a message to a Telegram chat, chunking if needed."""
        if not self._app:
            return SendResult(success=False, error="Bot not initialized")

        chunks = chunk_message(text)
        last_message_id: int | None = None

        for i, chunk in enumerate(chunks):
            try:
                # Try with MarkdownV2 first, fall back to plain text
                effective_parse_mode = parse_mode or self.config.parse_mode
                try:
                    msg = await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=escape_markdown_v2(chunk) if effective_parse_mode == "MarkdownV2" else chunk,
                        parse_mode=effective_parse_mode,
                        reply_to_message_id=reply_to if i == 0 else None,
                    )
                except Exception:
                    # Fallback: send as plain text
                    msg = await self._app.bot.send_message(
                        chat_id=chat_id,
                        text=chunk,
                        reply_to_message_id=reply_to if i == 0 else None,
                    )
                last_message_id = msg.message_id
            except Exception as exc:
                logger.error("Failed to send message to chat %d: %s", chat_id, exc)
                return SendResult(success=False, error=str(exc))

        return SendResult(success=True, message_id=last_message_id)

    async def on_message(self, event: MessageEvent) -> None:
        """Handle an incoming message — dispatch to agent and reply."""
        if not self.config.chat_allowed(event.chat_id):
            logger.warning(
                "Ignoring message from unauthorized chat %d (user %d)",
                event.chat_id,
                event.user_id,
            )
            return

        await self._dispatch_to_agent(event)

    # ------------------------------------------------------------------
    # Internal handlers
    # ------------------------------------------------------------------

    async def _cmd_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /start command."""
        if update.effective_chat is None:
            return
        await self.send(
            update.effective_chat.id,
            "Prometheus is online. Send me a message and I'll help you.",
            parse_mode=None,
        )

    async def _cmd_clear(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /clear command — reset conversation history."""
        if update.effective_chat is None:
            return
        session_key = f"{Platform.TELEGRAM.value}:{update.effective_chat.id}"
        self._sessions.pop(session_key, None)
        await self.send(
            update.effective_chat.id,
            "Conversation cleared.",
            parse_mode=None,
        )

    async def _cmd_help(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /help command — list commands and capabilities."""
        if update.effective_chat is None:
            return
        text = (
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
        await self.send(update.effective_chat.id, text, parse_mode=None)

    async def _cmd_reset(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /reset command — clear conversation context."""
        if update.effective_chat is None:
            return
        session_key = f"{Platform.TELEGRAM.value}:{update.effective_chat.id}"
        self._sessions.pop(session_key, None)
        await self.send(
            update.effective_chat.id,
            "Conversation context reset.",
            parse_mode=None,
        )

    async def _cmd_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /model command — show current model and provider."""
        if update.effective_chat is None:
            return
        name = self.model_name or "(unknown)"
        provider = self.model_provider or "(unknown)"
        await self.send(
            update.effective_chat.id,
            f"Model: {name}\nProvider: {provider}",
            parse_mode=None,
        )

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /status command — model, uptime, tools, memory, SENTINEL."""
        if update.effective_chat is None:
            return

        lines: list[str] = ["Prometheus Status\n"]

        # Model
        lines.append(f"Model: {self.model_name or '(unknown)'}")
        lines.append(f"Provider: {self.model_provider or '(unknown)'}")

        # Uptime
        if self._start_time:
            elapsed = int(time.monotonic() - self._start_time)
            h, remainder = divmod(elapsed, 3600)
            m, s = divmod(remainder, 60)
            lines.append(f"Uptime: {h}h {m}m {s}s")

        # Tools
        lines.append(f"Tools: {len(self.tool_registry.list_tools())}")

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
                state = "dreaming" if _autodream.dreaming else (
                    "active" if _observer.started else "idle"
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

        await self.send(update.effective_chat.id, "\n".join(lines), parse_mode=None)

    async def _cmd_wiki(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /wiki command — wiki stats and recent entries."""
        if update.effective_chat is None:
            return

        wiki_index = Path.home() / ".prometheus" / "wiki" / "index.md"
        if not wiki_index.exists():
            await self.send(
                update.effective_chat.id,
                "Wiki: no index found at ~/.prometheus/wiki/index.md",
                parse_mode=None,
            )
            return

        try:
            content = wiki_index.read_text(encoding="utf-8")
            entries: list[str] = []
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("- ["):
                    entries.append(line)

            lines = [f"Wiki: {len(entries)} pages"]

            # Last modified
            mtime = wiki_index.stat().st_mtime
            from datetime import datetime, timezone
            updated = datetime.fromtimestamp(mtime, tz=timezone.utc)
            lines.append(f"Last updated: {updated.strftime('%Y-%m-%d %H:%M UTC')}")

            # Show last 5 entries
            if entries:
                lines.append("\nRecent entries:")
                for entry in entries[-5:]:
                    lines.append(f"  {entry}")

            await self.send(
                update.effective_chat.id, "\n".join(lines), parse_mode=None
            )
        except Exception as exc:
            await self.send(
                update.effective_chat.id,
                f"Wiki: error reading index — {exc}",
                parse_mode=None,
            )

    async def _cmd_sentinel(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /sentinel command — reuses SentinelStatusTool logic."""
        if update.effective_chat is None:
            return

        try:
            from prometheus.tools.builtin.sentinel_status import (
                _autodream,
                _observer,
                _signal_bus,
            )
        except ImportError:
            await self.send(
                update.effective_chat.id,
                "SENTINEL module not available.",
                parse_mode=None,
            )
            return

        if _signal_bus is None or _observer is None or _autodream is None:
            await self.send(
                update.effective_chat.id,
                "SENTINEL not initialized. Is the daemon running with sentinel enabled?",
                parse_mode=None,
            )
            return

        lines: list[str] = ["SENTINEL Status\n"]

        # Observer
        idle_secs = int(time.time() - _observer.last_activity)
        lines.append("Observer:")
        lines.append(f"  Active: {_observer.started}")
        lines.append(f"  Last activity: {idle_secs}s ago")
        lines.append(f"  Pending nudges: {len(_observer.pending_nudges)}")

        # AutoDream
        lines.append("\nAutoDream Engine:")
        lines.append(f"  Dreaming: {_autodream.dreaming}")
        lines.append(f"  Cycles completed: {_autodream.cycle_count}")
        if _autodream.last_cycle_time:
            ago = int(time.time() - _autodream.last_cycle_time)
            lines.append(f"  Last cycle: {ago}s ago")

        # Signal bus
        lines.append("\nSignal Bus:")
        lines.append(f"  Total signals: {_signal_bus.signal_count}")
        lines.append(f"  Subscribers: {_signal_bus.subscriber_count}")

        # Recent signals
        recent = _signal_bus.recent(limit=10)
        if recent:
            lines.append("\nRecent Signals:")
            for sig in recent:
                ago = int(time.time() - sig.timestamp)
                lines.append(f"  [{sig.kind}] from {sig.source} ({ago}s ago)")

        # Last dream results
        if _autodream.last_results:
            lines.append("\nLast Dream Cycle:")
            for r in _autodream.last_results:
                status = "OK" if not r.error else f"FAIL: {r.error}"
                lines.append(f"  {r.phase}: {status} ({r.duration_seconds:.1f}s)")
                for k, v in r.summary.items():
                    lines.append(f"    {k}: {v}")

        # Pending nudges
        if _observer.pending_nudges:
            lines.append("\nPending Nudges:")
            for nudge in _observer.pending_nudges[:5]:
                lines.append(f"  [{nudge.nudge_type}] {nudge.message[:80]}")

        await self.send(update.effective_chat.id, "\n".join(lines), parse_mode=None)

    async def _cmd_benchmark(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /benchmark command — run a quick smoke test."""
        if update.effective_chat is None:
            return

        chat_id = update.effective_chat.id
        await self.send(chat_id, "Running benchmark...", parse_mode=None)

        try:
            t0 = time.monotonic()
            result = await self.agent_loop.run_async(
                system_prompt="You are a helpful assistant. Be concise.",
                user_message="What is 2+2? Reply with just the number.",
                tools=[],
            )
            elapsed = time.monotonic() - t0

            response = (result.text or "").strip()
            passed = "4" in response

            lines = [
                f"Benchmark: {'PASS' if passed else 'FAIL'}",
                f"Latency: {elapsed:.2f}s",
                f"Response: {response[:100]}",
                f"Tokens: {result.usage.input_tokens} in / {result.usage.output_tokens} out",
            ]
            await self.send(chat_id, "\n".join(lines), parse_mode=None)
        except Exception as exc:
            await self.send(
                chat_id,
                f"Benchmark: FAIL\nError: {exc}",
                parse_mode=None,
            )

    async def _cmd_skills(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /skills command — list available skills."""
        if update.effective_chat is None:
            return

        try:
            from prometheus.skills.loader import load_skill_registry
            registry = load_skill_registry()
            skills = registry.list_skills()
        except Exception as exc:
            await self.send(
                update.effective_chat.id,
                f"Skills: error loading registry — {exc}",
                parse_mode=None,
            )
            return

        if not skills:
            await self.send(
                update.effective_chat.id,
                "No skills available.",
                parse_mode=None,
            )
            return

        lines = [f"Skills ({len(skills)})\n"]
        for skill in skills:
            source_tag = f" [{skill.source}]" if skill.source else ""
            lines.append(f"  {skill.name}{source_tag}")
            if skill.description:
                lines.append(f"    {skill.description[:80]}")

        lines.append("\nUse the skill tool to load a skill by name.")
        await self.send(update.effective_chat.id, "\n".join(lines), parse_mode=None)

    async def _cmd_context(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle /context command — show context window usage."""
        if update.effective_chat is None:
            return

        from prometheus.context.token_estimation import estimate_tokens

        # Read effective_limit from config (with model override)
        try:
            from prometheus.context.budget import TokenBudget
            budget = TokenBudget.from_config(model=self.model_name)
            effective_limit = budget.effective_limit
            reserved_output = budget.reserved_output
        except Exception:
            effective_limit = 24000
            reserved_output = 2000

        # Estimate system prompt cost
        prompt_tokens = estimate_tokens(self.system_prompt)

        # Available for conversation
        available = effective_limit - reserved_output
        headroom = max(0, available - prompt_tokens)
        usage_pct = (prompt_tokens / available * 100) if available > 0 else 0

        lines = [
            "Context Window\n",
            f"Window size:    {effective_limit:,} tokens",
            f"Reserved output: {reserved_output:,} tokens",
            f"Available:       {available:,} tokens",
            f"",
            f"System prompt:   {prompt_tokens:,} tokens ({usage_pct:.0f}%)",
            f"Headroom:        {headroom:,} tokens",
            f"",
            f"Model: {self.model_name or '(unknown)'}",
        ]

        # Show bar visualization
        bar_len = 20
        filled = round(usage_pct / 100 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        lines.append(f"[{bar}] {usage_pct:.0f}% used")

        await self.send(update.effective_chat.id, "\n".join(lines), parse_mode=None)

    async def _handle_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle incoming text messages."""
        if not update.message or not update.message.text or not update.effective_chat:
            return

        event = MessageEvent(
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id if update.effective_user else 0,
            text=update.message.text,
            message_id=update.message.message_id,
            platform=Platform.TELEGRAM,
            message_type=MessageType.TEXT,
            username=(
                update.effective_user.username if update.effective_user else None
            ),
        )
        await self.on_message(event)

    async def _dispatch_to_agent(self, event: MessageEvent) -> None:
        """Route a message through AgentLoop and send the response."""
        if self._app:
            try:
                await self._app.bot.send_chat_action(
                    chat_id=event.chat_id, action=ChatAction.TYPING
                )
            except Exception:
                pass  # typing indicator is best-effort

        try:
            result = await self.agent_loop.run_async(
                system_prompt=self.system_prompt,
                user_message=event.text,
                tools=self.tool_registry.list_schemas(),
            )
            response_text = result.text or "(no response)"
        except Exception as exc:
            logger.error("Agent error for chat %d: %s", event.chat_id, exc)
            response_text = f"Error: {exc}"

        await self.send(
            event.chat_id,
            response_text,
            reply_to=event.message_id,
        )
