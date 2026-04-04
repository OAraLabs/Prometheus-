"""Telegram platform adapter — full bot using python-telegram-bot.

Source: Novel code for Prometheus Sprint 6 (architecture inspired by Hermes
gateway.platforms.telegram).

Receives messages via long-polling, dispatches to AgentLoop.run_async(),
sends responses back with MarkdownV2 formatting and message chunking.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from telegram import Update
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
    ) -> None:
        super().__init__(config)
        self.agent_loop = agent_loop
        self.tool_registry = tool_registry
        self.system_prompt = system_prompt
        self._app: Application | None = None
        self._sessions: dict[str, list[dict[str, Any]]] = {}

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
        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_text)
        )

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        self._running = True
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
