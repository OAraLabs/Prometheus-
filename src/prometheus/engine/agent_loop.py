# Source: OpenHarness (HKUDS/OpenHarness)
# Original: src/openharness/engine/query.py
# License: MIT
# Modified: decoupled from Anthropic API — replaced SupportsStreamingMessages + openharness.api.client
#           with abstract ModelProvider from prometheus.providers.base;
#           renamed all imports (openharness → prometheus);
#           removed auto-compact (Sprint 4 concern — openharness.services.compact not yet ported);
#           wrapped run_query() async generator into AgentLoop class with run() sync entry point;
#           ToolRegistry / PermissionChecker are optional (stubs used when not provided)

"""Core tool-aware agent loop."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from prometheus.engine.messages import ConversationMessage, ToolResultBlock
from prometheus.engine.stream_events import (
    AssistantTextDelta,
    AssistantTurnComplete,
    StreamEvent,
    ToolExecutionCompleted,
    ToolExecutionStarted,
)
from prometheus.engine.usage import UsageSnapshot
from prometheus.providers.base import (
    ApiMessageCompleteEvent,
    ApiMessageRequest,
    ApiTextDeltaEvent,
    ModelProvider,
)

log = logging.getLogger(__name__)

PermissionPrompt = Callable[[str, str], Awaitable[bool]]
AskUserPrompt = Callable[[str], Awaitable[str]]


@dataclass
class RunResult:
    """The outcome of a completed agent run."""

    text: str
    messages: list[ConversationMessage]
    usage: UsageSnapshot = field(default_factory=UsageSnapshot)
    turns: int = 0


@dataclass
class LoopContext:
    """Context shared across a loop run."""

    provider: ModelProvider
    model: str
    system_prompt: str
    max_tokens: int
    tool_registry: object | None = None       # ToolRegistry — wired in Sprint 2
    permission_checker: object | None = None  # PermissionChecker — wired in Sprint 4
    hook_executor: object | None = None       # HookExecutor — wired in Sprint 2
    adapter: object | None = None             # ModelAdapter — wired in Sprint 3
    telemetry: object | None = None           # ToolCallTelemetry — wired in Sprint 3
    cwd: Path = field(default_factory=Path.cwd)
    max_turns: int = 200
    permission_prompt: PermissionPrompt | None = None
    ask_user_prompt: AskUserPrompt | None = None
    tool_metadata: dict[str, object] | None = None


async def run_loop(
    context: LoopContext,
    messages: list[ConversationMessage],
) -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]:
    """Run the conversation loop until the model stops requesting tools.

    Yields (StreamEvent, UsageSnapshot | None) tuples. The loop exits when
    the assistant returns a response with no tool_uses, or after max_turns.
    """
    tool_schema: list[dict] = []
    if context.tool_registry is not None and hasattr(context.tool_registry, "to_api_schema"):
        tool_schema = context.tool_registry.to_api_schema()

    # Sprint 3: format tools + system prompt for the target model
    active_system_prompt = context.system_prompt
    active_tools = tool_schema
    if context.adapter is not None and hasattr(context.adapter, "format_request"):
        active_system_prompt, active_tools = context.adapter.format_request(
            context.system_prompt, tool_schema
        )

    for turn in range(context.max_turns):
        final_message: ConversationMessage | None = None
        usage = UsageSnapshot()

        async for event in context.provider.stream_message(
            ApiMessageRequest(
                model=context.model,
                messages=messages,
                system_prompt=active_system_prompt,
                max_tokens=context.max_tokens,
                tools=active_tools,
            )
        ):
            if isinstance(event, ApiTextDeltaEvent):
                yield AssistantTextDelta(text=event.text), None
                continue

            if isinstance(event, ApiMessageCompleteEvent):
                final_message = event.message
                usage = event.usage

        if final_message is None:
            raise RuntimeError("Model stream finished without a final message")

        # Sprint 3: try to extract tool calls from text when none came back structured
        if (
            not final_message.tool_uses
            and final_message.text
            and context.adapter is not None
        ):
            extracted = context.adapter.extract_tool_calls(
                final_message.text, context.tool_registry
            )
            if extracted:
                from prometheus.engine.messages import TextBlock
                final_message = ConversationMessage(
                    role="assistant",
                    content=extracted,
                )

        messages.append(final_message)
        yield AssistantTurnComplete(message=final_message, usage=usage), usage

        if not final_message.tool_uses:
            return

        tool_calls = final_message.tool_uses

        if len(tool_calls) == 1:
            tc = tool_calls[0]
            yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input), None
            result = await _execute_tool_call(context, tc.name, tc.id, tc.input)
            yield ToolExecutionCompleted(
                tool_name=tc.name,
                output=result.content,
                is_error=result.is_error,
            ), None
            tool_results = [result]
        else:
            for tc in tool_calls:
                yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input), None

            async def _run(tc):
                return await _execute_tool_call(context, tc.name, tc.id, tc.input)

            results = await asyncio.gather(*[_run(tc) for tc in tool_calls])
            tool_results = list(results)

            for tc, result in zip(tool_calls, tool_results):
                yield ToolExecutionCompleted(
                    tool_name=tc.name,
                    output=result.content,
                    is_error=result.is_error,
                ), None

        messages.append(ConversationMessage(role="user", content=tool_results))

    raise RuntimeError(f"Exceeded maximum turn limit ({context.max_turns})")


async def _execute_tool_call(
    context: LoopContext,
    tool_name: str,
    tool_use_id: str,
    tool_input: dict[str, object],
) -> ToolResultBlock:
    """Execute a single tool call, running hooks if configured."""
    # Pre-tool hook (Sprint 2)
    if context.hook_executor is not None:
        from prometheus.hooks import HookEvent
        pre = await context.hook_executor.execute(
            HookEvent.PRE_TOOL_USE,
            {"tool_name": tool_name, "tool_input": tool_input, "event": HookEvent.PRE_TOOL_USE.value},
        )
        if pre.blocked:
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=pre.reason or f"pre_tool_use hook blocked {tool_name}",
                is_error=True,
            )

    if context.tool_registry is None:
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"No tool registry configured — cannot execute {tool_name}",
            is_error=True,
        )

    # Sprint 3: validate + auto-repair the tool call before execution
    retries_used = 0
    repair_log: list[str] = []
    if context.adapter is not None:
        try:
            tool_name, tool_input, repair_log = context.adapter.validate_and_repair(
                tool_name, tool_input, context.tool_registry
            )
        except ValueError as exc:
            # Validation failed and repair failed — ask retry engine
            action, retry_prompt = context.adapter.handle_retry(
                tool_name, str(exc), context.tool_registry
            )
            retries_used = 1
            if context.telemetry is not None:
                context.telemetry.record(
                    model=context.model,
                    tool_name=tool_name,
                    success=False,
                    retries=retries_used,
                    latency_ms=0.0,
                    error_type="validation_failed",
                    error_detail=str(exc),
                )
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=retry_prompt,
                is_error=True,
            )

    tool = context.tool_registry.get(tool_name)
    if tool is None:
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Unknown tool: {tool_name}",
            is_error=True,
        )

    try:
        parsed_input = tool.input_model.model_validate(tool_input)
    except Exception as exc:
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Invalid input for {tool_name}: {exc}",
            is_error=True,
        )

    # Permission check (Sprint 4)
    if context.permission_checker is not None:
        _file_path = str(tool_input.get("file_path", "")) or None
        _command = str(tool_input.get("command", "")) or None
        decision = context.permission_checker.evaluate(
            tool_name,
            is_read_only=tool.is_read_only(parsed_input),
            file_path=_file_path,
            command=_command,
        )
        if not decision.allowed:
            if decision.requires_confirmation and context.permission_prompt is not None:
                confirmed = await context.permission_prompt(tool_name, decision.reason)
                if not confirmed:
                    return ToolResultBlock(
                        tool_use_id=tool_use_id,
                        content=f"Permission denied for {tool_name}",
                        is_error=True,
                    )
            else:
                return ToolResultBlock(
                    tool_use_id=tool_use_id,
                    content=decision.reason or f"Permission denied for {tool_name}",
                    is_error=True,
                )

    from prometheus.tools.base import ToolExecutionContext
    _t0 = time.monotonic()
    result = await tool.execute(
        parsed_input,
        ToolExecutionContext(
            cwd=context.cwd,
            metadata={
                "tool_registry": context.tool_registry,
                "ask_user_prompt": context.ask_user_prompt,
                **(context.tool_metadata or {}),
            },
        ),
    )
    _latency_ms = (time.monotonic() - _t0) * 1000.0
    tool_result = ToolResultBlock(
        tool_use_id=tool_use_id,
        content=result.output,
        is_error=result.is_error,
    )

    # Sprint 3: record telemetry
    if context.telemetry is not None:
        context.telemetry.record(
            model=context.model,
            tool_name=tool_name,
            success=not result.is_error,
            retries=retries_used,
            latency_ms=_latency_ms,
            error_type="tool_error" if result.is_error else None,
        )

    # Post-tool hook (Sprint 2)
    if context.hook_executor is not None:
        from prometheus.hooks import HookEvent
        await context.hook_executor.execute(
            HookEvent.POST_TOOL_USE,
            {
                "tool_name": tool_name,
                "tool_input": tool_input,
                "tool_output": tool_result.content,
                "tool_is_error": tool_result.is_error,
                "event": HookEvent.POST_TOOL_USE.value,
            },
        )

    return tool_result


class AgentLoop:
    """High-level agent loop that wraps run_loop().

    Usage:
        provider = StubProvider(base_url="http://localhost:8080")
        loop = AgentLoop(provider=provider)
        result = loop.run(
            system_prompt="You are a helpful assistant.",
            user_message="What is 2+2?",
        )
        print(result.text)
    """

    def __init__(
        self,
        provider: ModelProvider,
        model: str = "qwen3.5-32b",
        max_tokens: int = 4096,
        max_turns: int = 200,
        tool_registry=None,
        hook_executor=None,
        permission_checker=None,
        adapter=None,
        telemetry=None,
        cwd: Path | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._tool_registry = tool_registry
        self._hook_executor = hook_executor
        self._permission_checker = permission_checker
        self._adapter = adapter
        self._telemetry = telemetry
        self._cwd = cwd or Path.cwd()

    async def run_async(
        self,
        system_prompt: str,
        user_message: str,
        tools: list | None = None,
    ) -> RunResult:
        """Run the agent loop asynchronously, return a RunResult."""
        messages = [ConversationMessage.from_user_text(user_message)]

        context = LoopContext(
            provider=self._provider,
            model=self._model,
            system_prompt=system_prompt,
            max_tokens=self._max_tokens,
            max_turns=self._max_turns,
            tool_registry=self._tool_registry,
            hook_executor=self._hook_executor,
            permission_checker=self._permission_checker,
            adapter=self._adapter,
            telemetry=self._telemetry,
            cwd=self._cwd,
        )

        last_text = ""
        last_usage = UsageSnapshot()
        turns = 0

        async for event, usage in run_loop(context, messages):
            if isinstance(event, AssistantTurnComplete):
                last_text = event.message.text
                last_usage = event.usage
                turns += 1
            elif isinstance(event, AssistantTextDelta):
                pass  # streaming deltas — consumed silently here

        return RunResult(
            text=last_text,
            messages=messages,
            usage=last_usage,
            turns=turns,
        )

    def run(
        self,
        system_prompt: str,
        user_message: str,
        tools: list | None = None,
    ) -> RunResult:
        """Synchronous entry point — wraps run_async() via asyncio.run()."""
        return asyncio.run(self.run_async(system_prompt, user_message, tools))
