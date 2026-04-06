"""PrometheusJudge — LLM-as-judge using the local llama.cpp endpoint.

Evaluates agent outputs against expected behavior descriptions.
Uses the same OpenAI-compatible /v1/chat/completions API as the main model.

Supports two evaluation modes:
- evaluate(): JSON-based scoring (original)
- evaluate_geval(): G-Eval chain-of-thought scoring (more reliable with local models)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge for an AI agent. You will be given:
1. The task the agent was asked to perform
2. A description of the expected behavior
3. The agent's actual output
4. Optionally, the tools the agent called

Rate how well the agent completed the task on a scale from 0.0 to 1.0:
- 1.0 = Task fully completed as expected
- 0.7 = Task mostly completed with minor issues
- 0.5 = Task partially completed
- 0.3 = Task attempted but largely failed
- 0.0 = Task not attempted or completely wrong

Respond with ONLY a JSON object: {"score": <float>, "reasoning": "<brief explanation>"}
"""

_GEVAL_SYSTEM_PROMPT = """\
You are an evaluation judge. You will evaluate an AI agent's output \
by reasoning through specific criteria step by step.

For each criterion, explain your assessment briefly. \
After evaluating all criteria, provide your final score.

IMPORTANT: Your very last line MUST be exactly: SCORE: <number>
where <number> is between 0.0 and 1.0.
"""


@dataclass
class JudgeVerdict:
    """Result of an LLM judge evaluation."""

    score: float
    reasoning: str
    raw_response: str


class PrometheusJudge:
    """Evaluate agent outputs using a local LLM as judge.

    Uses raw httpx calls to /v1/chat/completions — independent of the
    ModelProvider abstraction to avoid circular dependencies.
    """

    def __init__(
        self,
        base_url: str = "http://GPU_HOST:8080",
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    async def _detect_model(self) -> str:
        """Query /v1/models to find the loaded model."""
        if self._model:
            return self._model
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"{self._base_url}/v1/models")
                resp.raise_for_status()
                models = resp.json().get("data", [])
                if models:
                    detected = models[0].get("id", "unknown")
                    log.debug("Judge detected model: %s", detected)
                    return detected
        except Exception as exc:
            log.warning("Could not detect judge model: %s", exc)
        return "unknown"

    async def _call_llm(self, system: str, user: str, max_tokens: int = 1024) -> str:
        """Send a chat completion request and return the response text."""
        model = await self._detect_model()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # JSON-based evaluation (original)
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        task_input: str,
        agent_output: str,
        expected_behavior: str,
        tool_trace: list[dict[str, Any]] | None = None,
    ) -> JudgeVerdict:
        """Judge an agent's output against expected behavior (JSON mode).

        Returns a JudgeVerdict with score (0.0-1.0) and reasoning.
        """
        user_prompt = f"Task: {task_input}\n\nExpected behavior: {expected_behavior}\n\n"
        user_prompt += f"Agent output:\n{agent_output[:3000]}\n"

        if tool_trace:
            tools_summary = ", ".join(
                t.get("tool_name", "unknown") for t in tool_trace
            )
            user_prompt += f"\nTools called: {tools_summary}"

        raw = await self._call_llm(_JUDGE_SYSTEM_PROMPT, user_prompt, max_tokens=512)
        return self._parse_verdict(raw)

    # ------------------------------------------------------------------
    # G-Eval: chain-of-thought evaluation (better for local models)
    # ------------------------------------------------------------------

    async def evaluate_geval(
        self,
        criteria: list[str],
        context: str,
    ) -> JudgeVerdict:
        """G-Eval style evaluation with chain-of-thought reasoning.

        The model reasons through each criterion step by step, then
        produces a final score. More reliable than JSON-only prompting
        with local models (Qwen, Gemma) because the model can think
        before scoring.

        Args:
            criteria: Numbered evaluation criteria the model reasons through.
            context: The full evaluation context (task, output, evidence).
        """
        criteria_text = "\n".join(
            f"{i}. {c}" for i, c in enumerate(criteria, 1)
        )

        user_prompt = f"""{context}

---

Evaluate step by step using these criteria:
{criteria_text}

Think through each criterion, then write your final score.
Your last line MUST be: SCORE: <number from 0.0 to 1.0>"""

        raw = await self._call_llm(_GEVAL_SYSTEM_PROMPT, user_prompt, max_tokens=1024)
        return self._parse_geval_verdict(raw)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_verdict(self, raw: str) -> JudgeVerdict:
        """Parse the judge's JSON response into a JudgeVerdict."""
        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            parsed = json.loads(raw[start:end])
            return JudgeVerdict(
                score=max(0.0, min(1.0, float(parsed.get("score", 0.0)))),
                reasoning=str(parsed.get("reasoning", "")),
                raw_response=raw,
            )
        except (json.JSONDecodeError, ValueError, KeyError):
            log.warning("Could not parse judge response as JSON: %s", raw[:200])
            return self._fallback_parse(raw)

    def _parse_geval_verdict(self, raw: str) -> JudgeVerdict:
        """Parse G-Eval chain-of-thought response.

        Looks for 'SCORE: X.X' at the end of the response. The reasoning
        is everything before the SCORE line.
        """
        # Look for SCORE: pattern (case-insensitive, anywhere in response)
        match = re.search(r"SCORE:\s*(\d+\.?\d*)", raw, re.IGNORECASE)
        if match:
            score = max(0.0, min(1.0, float(match.group(1))))
            # Everything before the SCORE line is reasoning
            reasoning_end = match.start()
            reasoning = raw[:reasoning_end].strip()
            # Take last ~500 chars of reasoning (the most relevant part)
            if len(reasoning) > 500:
                reasoning = "..." + reasoning[-500:]
            return JudgeVerdict(
                score=score,
                reasoning=reasoning,
                raw_response=raw,
            )

        log.warning("No SCORE: found in G-Eval response: %s", raw[:200])
        return self._fallback_parse(raw)

    def _fallback_parse(self, raw: str) -> JudgeVerdict:
        """Last-resort score extraction from unstructured text."""
        match = re.search(r"(\d+\.?\d*)", raw)
        score = float(match.group(1)) if match else 0.0
        score = max(0.0, min(1.0, score))
        return JudgeVerdict(
            score=score,
            reasoning=f"Parse fallback: {raw[:200]}",
            raw_response=raw,
        )
