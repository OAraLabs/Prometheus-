"""Gateway configuration — platform enum and config dataclasses.

Source: Novel code for Prometheus Sprint 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Platform(str, Enum):
    """Supported messaging platforms."""

    TELEGRAM = "telegram"
    CLI = "cli"
    API = "api"


@dataclass
class PlatformConfig:
    """Configuration for a single platform adapter."""

    platform: Platform
    token: str = ""
    webhook_url: str | None = None
    allowed_chat_ids: list[int] = field(default_factory=list)
    proxy_url: str | None = None
    max_message_length: int = 4096
    parse_mode: str = "MarkdownV2"
    connect_timeout: float = 30.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_restricted(self) -> bool:
        """True if only allowed_chat_ids may use this adapter."""
        return len(self.allowed_chat_ids) > 0

    def chat_allowed(self, chat_id: int) -> bool:
        """Return True if the chat is permitted (or no restrictions set)."""
        if not self.allowed_chat_ids:
            return True
        return chat_id in self.allowed_chat_ids
