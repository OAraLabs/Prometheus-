"""ContextCompressor — pruning-based context compression for Sprint 4.

Triggered when TokenBudget.is_approaching_limit() returns True.
Strategy: prune tool_result content from messages older than fresh_tail_count turns,
retaining the tool name so the model retains structural context.

Full LCM summarization is deferred to Sprint 7.

Usage:
    compressor = ContextCompressor(budget, fresh_tail_count=32)
    messages = compressor.maybe_compress(messages)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from prometheus.context.budget import TokenBudget

if TYPE_CHECKING:
    from prometheus.engine.messages import ConversationMessage

log = logging.getLogger(__name__)

_PRUNED_MARKER = "[content pruned — context compression]"


class ContextCompressor:
    """Prune old tool results when the context budget is approaching its limit.

    Args:
        budget:           TokenBudget to check before deciding to compress.
        fresh_tail_count: Number of most-recent *user* messages to preserve intact.
                          Tool results in older messages are pruned.
    """

    def __init__(
        self,
        budget: TokenBudget,
        fresh_tail_count: int = 32,
    ) -> None:
        self._budget = budget
        self._fresh_tail_count = fresh_tail_count

    @classmethod
    def from_config(
        cls,
        budget: TokenBudget,
        config_path: str | None = None,
    ) -> ContextCompressor:
        """Build from prometheus.yaml context.fresh_tail_count."""
        import yaml
        from pathlib import Path

        if config_path is None:
            from prometheus.config.defaults import DEFAULTS_PATH
            config_path = str(DEFAULTS_PATH)

        try:
            with open(Path(config_path).expanduser()) as fh:
                data = yaml.safe_load(fh)
            fresh_tail_count = data.get("context", {}).get("fresh_tail_count", 32)
        except (OSError, Exception):
            fresh_tail_count = 32

        return cls(budget=budget, fresh_tail_count=fresh_tail_count)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def maybe_compress(
        self,
        messages: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        """Compress *messages* if budget is approaching limit.

        Returns the (possibly pruned) message list.  Modifies nothing in place;
        returns a new list with pruned message objects.
        """
        if not self._budget.is_approaching_limit():
            return messages

        compressed, pruned_count = self._prune_old_tool_results(messages)
        if pruned_count:
            log.info(
                "ContextCompressor: pruned %d tool_result blocks from older turns "
                "(fresh_tail_count=%d, budget_used=%d/%d)",
                pruned_count,
                self._fresh_tail_count,
                self._budget.used,
                self._budget.effective_limit,
            )
        return compressed

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune_old_tool_results(
        self,
        messages: list[ConversationMessage],
    ) -> tuple[list[ConversationMessage], int]:
        """Return (pruned_messages, count_of_pruned_blocks).

        Identifies the most-recent `fresh_tail_count` user-role messages and
        prunes tool_result content from all earlier user messages.
        """
        from prometheus.engine.messages import ConversationMessage, ToolResultBlock

        # Collect indices of user messages (tool results come in user turns)
        user_indices = [i for i, m in enumerate(messages) if m.role == "user"]

        # The last fresh_tail_count user messages are protected
        protected = set(user_indices[-self._fresh_tail_count :])

        pruned_count = 0
        result: list[ConversationMessage] = []

        for idx, msg in enumerate(messages):
            if idx in protected or msg.role != "user":
                result.append(msg)
                continue

            # Check if any content block is a ToolResultBlock
            has_tool_results = any(
                isinstance(block, ToolResultBlock) for block in msg.content
            )
            if not has_tool_results:
                result.append(msg)
                continue

            # Rebuild message with pruned ToolResultBlocks
            new_content = []
            for block in msg.content:
                if isinstance(block, ToolResultBlock):
                    pruned_count += 1
                    new_content.append(
                        ToolResultBlock(
                            tool_use_id=block.tool_use_id,
                            content=_PRUNED_MARKER,
                            is_error=block.is_error,
                        )
                    )
                else:
                    new_content.append(block)

            result.append(
                ConversationMessage(role=msg.role, content=new_content)
            )

        return result, pruned_count
