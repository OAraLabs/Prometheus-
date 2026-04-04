"""adapter package — Model Adapter Layer (Sprint 3).

ModelAdapter bundles the validator, formatter, retry engine, and enforcer.
Pass an instance to AgentLoop to activate model-specific formatting,
tool-call validation, auto-repair, and structured output extraction.

Usage:
    from prometheus.adapter import ModelAdapter
    from prometheus.adapter.formatter import QwenFormatter

    adapter = ModelAdapter(formatter=QwenFormatter(), strictness="MEDIUM")
    loop = AgentLoop(provider=provider, tool_registry=registry, adapter=adapter)
"""

from __future__ import annotations

from typing import Any

from prometheus.adapter.enforcer import StructuredOutputEnforcer
from prometheus.adapter.formatter import (
    AnthropicFormatter,
    GemmaFormatter,
    ModelPromptFormatter,
    QwenFormatter,
)
from prometheus.adapter.retry import RetryAction, RetryEngine
from prometheus.adapter.validator import RepairResult, Strictness, ToolCallValidator, ValidationResult

__all__ = [
    "ModelAdapter",
    "AnthropicFormatter",
    "GemmaFormatter",
    "ModelPromptFormatter",
    "QwenFormatter",
    "RepairResult",
    "RetryAction",
    "RetryEngine",
    "Strictness",
    "StructuredOutputEnforcer",
    "ToolCallValidator",
    "ValidationResult",
]


class ModelAdapter:
    """High-level adapter that wires together all Sprint 3 components.

    Args:
        formatter:   Model-specific prompt formatter.
                     Defaults to AnthropicFormatter (passthrough).
        strictness:  Validation strictness: "NONE" | "MEDIUM" | "STRICT".
                     NONE skips all validation (safe for Claude API).
                     MEDIUM validates + auto-repairs (Qwen, Mistral).
                     STRICT validates + repairs + coerces aggressively.
        max_retries: Max tool-call retries before giving up.
    """

    def __init__(
        self,
        formatter: ModelPromptFormatter | None = None,
        strictness: str | Strictness = Strictness.NONE,
        max_retries: int = 3,
    ) -> None:
        self.formatter = formatter or AnthropicFormatter()
        self.validator = ToolCallValidator(strictness=strictness)
        self.retry = RetryEngine(max_retries=max_retries)
        self.enforcer = StructuredOutputEnforcer()

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def format_request(
        self,
        system_prompt: str,
        tools: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Format system prompt and tools for the current model.

        Returns (formatted_system_prompt, formatted_tools).
        """
        formatted_tools = self.formatter.format_tools(tools)
        formatted_system = self.formatter.format_system_prompt(
            system_prompt, tools  # pass original tools for description extraction
        )
        return formatted_system, formatted_tools

    # ------------------------------------------------------------------
    # Validation + repair
    # ------------------------------------------------------------------

    def validate_and_repair(
        self,
        tool_name: str,
        tool_input: Any,
        tool_registry: Any,
    ) -> tuple[str, dict[str, Any], list[str]]:
        """Validate a tool call and auto-repair if needed.

        Returns (final_tool_name, final_tool_input, repairs_made).
        Raises ValueError if validation fails and repair also fails.
        """
        result = self.validator.validate(tool_name, tool_input, tool_registry)
        if result.valid:
            return tool_name, tool_input, []

        repair = self.validator.repair(tool_name, tool_input, result.error, tool_registry)
        if repair.repaired:
            return repair.tool_name, repair.tool_input, repair.repairs_made

        raise ValueError(
            f"Tool call validation failed and could not be repaired: {result.error}"
        )

    # ------------------------------------------------------------------
    # Text-based tool call extraction
    # ------------------------------------------------------------------

    def extract_tool_calls(
        self,
        text: str,
        tool_registry: Any = None,
    ):
        """Extract tool calls from raw model text (for models that embed JSON in prose)."""
        return self.enforcer.extract_tool_calls(text, tool_registry)

    # ------------------------------------------------------------------
    # Retry
    # ------------------------------------------------------------------

    def handle_retry(
        self,
        tool_name: str,
        error: str,
        tool_registry: Any,
    ) -> tuple[RetryAction, str]:
        """Decide whether to retry and build the retry prompt."""
        return self.retry.handle_failure(tool_name, error, tool_registry)
