"""Custom DeepEval metrics for Prometheus evaluation.

Three metrics tailored for tool-heavy agent evaluation:
- TaskCompletionMetric: LLM-judged task completion (0-1)
- ToolUsageMetric: Deterministic tool usage check (no LLM)
- NoHallucinationMetric: LLM-judged groundedness check
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, TYPE_CHECKING

log = logging.getLogger(__name__)

# Graceful import — deepeval is optional
try:
    from deepeval.metrics import BaseMetric
    from deepeval.test_case import LLMTestCase

    _HAS_DEEPEVAL = True
except ImportError:
    _HAS_DEEPEVAL = False

    class BaseMetric:  # type: ignore[no-redef]
        """Stub when deepeval is not installed."""

        threshold: float = 0.0
        score: float | None = None
        reason: str | None = None
        success: bool | None = None

        def measure(self, test_case: Any) -> float:
            raise NotImplementedError("deepeval not installed")

        async def a_measure(self, test_case: Any) -> float:
            raise NotImplementedError("deepeval not installed")

        def is_successful(self) -> bool:
            return (self.score or 0) >= self.threshold

    class LLMTestCase:  # type: ignore[no-redef]
        """Stub when deepeval is not installed."""

        pass


if TYPE_CHECKING:
    from prometheus.evals.judge import PrometheusJudge


class TaskCompletionMetric(BaseMetric):
    """Measures whether the agent completed the requested task.

    Uses G-Eval (chain-of-thought) for more reliable scoring with local
    models. The judge reasons through criteria before producing a score,
    which avoids brittle JSON parsing.
    """

    _CRITERIA = [
        "Did the agent attempt the requested task?",
        "Did the agent use appropriate tools or methods to accomplish the task?",
        "Does the agent's output address the user's request?",
        "Is the task fully completed, or only partially done?",
        "Are there any errors or omissions in the result?",
    ]

    def __init__(self, judge: PrometheusJudge, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self.judge = judge
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None

    @property
    def __name__(self) -> str:
        return "Task Completion"

    def measure(self, test_case: LLMTestCase) -> float:
        """Sync measurement — wraps a_measure."""
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async measurement via G-Eval chain-of-thought."""
        meta = getattr(test_case, "additional_metadata", None) or {}
        tool_trace = meta.get("tool_trace", [])

        tools_summary = ""
        if tool_trace:
            tools_summary = "\nTools called: " + ", ".join(
                t.get("tool_name", "?") for t in tool_trace if isinstance(t, dict)
            )

        context = (
            f"Task: {test_case.input}\n\n"
            f"Expected behavior: {test_case.expected_output or 'N/A'}\n\n"
            f"Agent output:\n{(test_case.actual_output or '')[:3000]}"
            f"{tools_summary}"
        )

        verdict = await self.judge.evaluate_geval(
            criteria=self._CRITERIA,
            context=context,
        )
        self.score = verdict.score
        self.reason = verdict.reasoning
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return (self.score or 0) >= self.threshold


class ToolUsageMetric(BaseMetric):
    """Measures whether the agent used appropriate tools.

    Deterministic metric — no LLM required. Checks that expected tools
    were called by reading the tool trace from additional_metadata.
    Score = (expected tools found) / (total expected tools).
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None

    @property
    def __name__(self) -> str:
        return "Tool Usage"

    def measure(self, test_case: LLMTestCase) -> float:
        """Sync measurement — deterministic, no LLM needed."""
        meta = getattr(test_case, "additional_metadata", None) or {}
        tool_trace = meta.get("tool_trace", [])
        tools_used = {t.get("tool_name", "") for t in tool_trace if isinstance(t, dict)}

        # Extract expected tools from context
        expected = set()
        context = getattr(test_case, "context", None) or []
        for ctx in context:
            if isinstance(ctx, str) and ctx.startswith("Tools expected:"):
                expected = {
                    t.strip()
                    for t in ctx.replace("Tools expected:", "").split(",")
                    if t.strip()
                }

        if not expected:
            self.score = 1.0
            self.reason = "No specific tools expected"
            self.success = True
            return self.score

        # Score based on overlap (any expected tool found counts)
        overlap = len(tools_used & expected)
        self.score = overlap / len(expected)
        self.reason = f"Used {tools_used or 'none'}, expected {expected}"
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async wrapper — delegates to sync (no I/O needed)."""
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return (self.score or 0) >= self.threshold


class NoHallucinationMetric(BaseMetric):
    """Measures whether the agent's output is grounded in tool results.

    Uses G-Eval (chain-of-thought) to check for fabricated information.
    The judge reasons through groundedness criteria before scoring,
    which produces more consistent results than JSON-only prompting.
    """

    _CRITERIA = [
        "Does the agent's output only contain information from tool results it actually received?",
        "Did the agent claim to perform actions it didn't actually take (no tool call evidence)?",
        "Did the agent fabricate specific data, numbers, or details not present in any tool output?",
        "If the agent couldn't complete the task, did it honestly say so rather than making up a result?",
    ]

    def __init__(self, judge: PrometheusJudge, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.judge = judge
        self.score: float | None = None
        self.reason: str | None = None
        self.success: bool | None = None

    @property
    def __name__(self) -> str:
        return "No Hallucination"

    def measure(self, test_case: LLMTestCase) -> float:
        """Sync measurement — wraps a_measure."""
        return asyncio.run(self.a_measure(test_case))

    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async measurement via G-Eval chain-of-thought."""
        meta = getattr(test_case, "additional_metadata", None) or {}
        tool_trace = meta.get("tool_trace", [])

        tool_results = "\n".join(
            f"- {t.get('tool_name', '?')}: {t.get('result', '')[:200]}"
            for t in tool_trace
            if isinstance(t, dict)
        ) or "(no tool calls recorded)"

        context = (
            f"Task: {test_case.input}\n\n"
            f"Agent output:\n{(test_case.actual_output or '')[:2000]}\n\n"
            f"Tool results the agent had access to:\n{tool_results}"
        )

        verdict = await self.judge.evaluate_geval(
            criteria=self._CRITERIA,
            context=context,
        )
        self.score = verdict.score
        self.reason = verdict.reasoning
        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return (self.score or 0) >= self.threshold
