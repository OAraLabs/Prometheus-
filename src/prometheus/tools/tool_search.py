"""Deferred tool loading via search and select."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from prometheus.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# Inline Levenshtein (avoids circular imports from adapter/validator)
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Input model
# ---------------------------------------------------------------------------

class ToolSearchInput(BaseModel):
    """Arguments for the tool_search tool."""

    query: str = Field(description="Search query: tool name, description keywords, or exact name for select")
    action: Literal["search", "select"] = Field(
        default="search",
        description="'search' to fuzzy-match tools, 'select' to load a specific tool by exact name",
    )


# ---------------------------------------------------------------------------
# Tool implementation
# ---------------------------------------------------------------------------

class ToolSearchTool(BaseTool):
    """Search for available tools by name or description.

    Use 'search' to find tools matching a query, or 'select' to load a
    specific tool by exact name.
    """

    name = "tool_search"
    description = (
        "Search for available tools by name or description. "
        "Use 'search' to find tools matching a query, or 'select' to "
        "load a specific tool by exact name."
    )
    input_model = ToolSearchInput

    def __init__(self) -> None:
        self._registry: ToolRegistry | None = None

    def set_registry(self, registry: ToolRegistry) -> None:
        """Inject the tool registry (called after construction)."""
        self._registry = registry

    def is_read_only(self, arguments: BaseModel) -> bool:
        del arguments
        return True

    async def execute(
        self,
        arguments: ToolSearchInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        if self._registry is None:
            return ToolResult(
                output="ToolSearchTool has no registry configured. Call set_registry() first.",
                is_error=True,
            )

        if arguments.action == "select":
            return self._handle_select(arguments.query)
        return self._handle_search(arguments.query)

    # -- action handlers ----------------------------------------------------

    def _handle_select(self, name: str) -> ToolResult:
        """Exact-name lookup; return the full schema for one tool."""
        assert self._registry is not None
        tool = self._registry.get(name)
        if tool is None:
            available = sorted(t.name for t in self._registry.list_tools())
            return ToolResult(
                output=json.dumps(
                    {
                        "error": f"No tool named '{name}'.",
                        "available_tools": available,
                    },
                    indent=2,
                ),
                is_error=True,
            )
        return ToolResult(output=json.dumps(tool.to_api_schema(), indent=2))

    def _handle_search(self, query: str) -> ToolResult:
        """Fuzzy search across tool names and descriptions."""
        assert self._registry is not None
        tools = self._registry.list_tools()

        # Empty query: return all tool names
        if not query.strip():
            names = sorted(t.name for t in tools)
            return ToolResult(output=json.dumps({"tools": names}, indent=2))

        query_lower = query.lower()
        scored: list[tuple[float, BaseTool]] = []

        for tool in tools:
            score = self._score_tool(tool, query_lower)
            scored.append((score, tool))

        # Sort ascending by score (lower = better match)
        scored.sort(key=lambda pair: pair[0])

        top = scored[:5]
        results: list[dict[str, Any]] = []
        for score, tool in top:
            entry = tool.to_api_schema()
            entry["match_score"] = round(score, 3)
            results.append(entry)

        return ToolResult(output=json.dumps(results, indent=2))

    # -- scoring ------------------------------------------------------------

    @staticmethod
    def _score_tool(tool: BaseTool, query_lower: str) -> float:
        """Score a tool against a query (lower is better).

        Combines substring matching and Levenshtein distance for fuzzy
        matching.  Exact substring hits in the name get the best score.
        """
        name_lower = tool.name.lower()
        desc_lower = tool.description.lower()

        # Exact name match
        if query_lower == name_lower:
            return 0.0

        # Substring in name
        if query_lower in name_lower:
            return 1.0

        # Substring in description
        if query_lower in desc_lower:
            return 2.0

        # Check if any query word is a substring of name or description
        words = query_lower.split()
        word_hits = sum(
            1 for w in words if w in name_lower or w in desc_lower
        )
        if word_hits > 0:
            return 3.0 - (word_hits / max(len(words), 1))

        # Levenshtein against the tool name (normalized)
        dist = _levenshtein(query_lower, name_lower)
        max_len = max(len(query_lower), len(name_lower), 1)
        normalized = dist / max_len
        return 4.0 + normalized
