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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Sprint 10: Model Router + Divergence Detector
    # Lazy-imported at runtime to avoid circular import
    # (coordinator.__init__ → subagent → engine.agent_loop)
    from prometheus.adapter.router import ModelRouter, ProviderConfig
    from prometheus.coordinator.divergence import DivergenceDetector, CheckpointStore

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
    # Sprint 10: Model Router + Divergence Detector
    model_router: object | None = None
    divergence_detector: object | None = None


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

    # Sprint 10/15: route the first user message through ModelRouter
    if context.model_router is not None and messages:
        first_user = next(
            (m.text for m in messages if m.role == "user" and m.text), None
        )
        if first_user:
            try:
                route = context.model_router.route(first_user)
                log.debug(
                    "ModelRouter: %s → %s/%s (%s)",
                    first_user[:60], route.provider, route.model, route.reason,
                )
            except Exception:
                log.debug("ModelRouter: classification failed", exc_info=True)

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
        tool_results = await _dispatch_tool_calls(context, tool_calls)

        for tc, result in zip(tool_calls, tool_results):
            yield ToolExecutionStarted(tool_name=tc.name, tool_input=tc.input), None
            yield ToolExecutionCompleted(
                tool_name=tc.name,
                output=result.content,
                is_error=result.is_error,
            ), None

        messages.append(ConversationMessage(role="user", content=tool_results))

        # Sprint 10: checkpoint + divergence evaluation after tool dispatch
        if context.divergence_detector is not None:
            dd = context.divergence_detector
            # Maybe create a checkpoint
            msg_dicts = [
                {"role": m.role, "content": m.text or ""}
                for m in messages
                if hasattr(m, "role")
            ]
            dd.maybe_checkpoint(msg_dicts)

            # Evaluate divergence (only after 3+ steps to gather signal)
            if dd.step_count > 3:
                tool_result_dicts = [
                    {"result": tr.content, "success": not tr.is_error}
                    for tr in tool_results
                ]
                div_result = dd.evaluate(msg_dicts, tool_result_dicts)
                if div_result.should_rollback and div_result.checkpoint:
                    trust = 1  # default to non-autonomous
                    rolled_back, restored = dd.rollback(div_result.checkpoint, trust)
                    if rolled_back:
                        log.warning(
                            "Divergence rollback: restoring %d messages",
                            len(restored),
                        )

    raise RuntimeError(f"Exceeded maximum turn limit ({context.max_turns})")


async def _dispatch_tool_calls(
    context: LoopContext,
    tool_calls: list,
) -> list[ToolResultBlock]:
    """Dispatch tool calls with parallel execution for read-only tools.

    Read-only tools are executed simultaneously via ``asyncio.gather``.
    Mutating tools are executed sequentially afterwards to preserve order.
    Single tool calls skip partitioning entirely.
    """
    if len(tool_calls) == 1:
        tc = tool_calls[0]
        return [await _execute_tool_call(context, tc.name, tc.id, tc.input)]

    # Partition into read-only and mutating based on tool.is_read_only()
    read_only: list[tuple[int, object]] = []   # (original_index, tool_call)
    mutating: list[tuple[int, object]] = []

    for i, tc in enumerate(tool_calls):
        tool = context.tool_registry.get(tc.name) if context.tool_registry else None
        if tool is not None and _is_tool_read_only(tool, tc.input):
            read_only.append((i, tc))
        else:
            mutating.append((i, tc))

    results: list[tuple[int, ToolResultBlock]] = []

    # Run all read-only tools in parallel
    if read_only:
        async def _run_ro(idx, tc):
            r = await _execute_tool_call(context, tc.name, tc.id, tc.input)
            return idx, r

        parallel = await asyncio.gather(
            *[_run_ro(idx, tc) for idx, tc in read_only],
            return_exceptions=True,
        )
        for item in parallel:
            if isinstance(item, Exception):
                log.error("Parallel tool execution failed: %s", item)
                if context.telemetry is not None:
                    context.telemetry.record(
                        model=context.model,
                        tool_name="unknown_parallel",
                        success=False,
                        error_type="parallel_exception",
                        error_detail=str(item),
                    )
                # We lost the index — append a generic error
                results.append((-1, ToolResultBlock(
                    tool_use_id="error",
                    content=f"Parallel execution error: {item}",
                    is_error=True,
                )))
            else:
                results.append(item)

    # Run mutating tools sequentially (order matters)
    for idx, tc in mutating:
        result = await _execute_tool_call(context, tc.name, tc.id, tc.input)
        results.append((idx, result))

    # Restore original order
    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


def _is_tool_read_only(tool: object, tool_input: dict) -> bool:
    """Check if a tool call is read-only, handling both method and attribute patterns."""
    if callable(getattr(tool, "is_read_only", None)):
        try:
            parsed = tool.input_model.model_validate(tool_input)
            return tool.is_read_only(parsed)
        except Exception:
            return False
    return getattr(tool, "is_read_only", False)


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
            if context.telemetry is not None:
                context.telemetry.record(
                    model=context.model,
                    tool_name=tool_name,
                    success=False,
                    error_type="hook_blocked",
                    error_detail=pre.reason or f"pre_tool_use hook blocked {tool_name}",
                )
            return ToolResultBlock(
                tool_use_id=tool_use_id,
                content=pre.reason or f"pre_tool_use hook blocked {tool_name}",
                is_error=True,
            )

    if context.tool_registry is None:
        if context.telemetry is not None:
            context.telemetry.record(
                model=context.model,
                tool_name=tool_name,
                success=False,
                error_type="no_registry",
                error_detail="No tool registry configured",
            )
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
        if context.telemetry is not None:
            context.telemetry.record(
                model=context.model,
                tool_name=tool_name,
                success=False,
                error_type="unknown_tool",
                error_detail=f"Unknown tool: {tool_name}",
            )
        return ToolResultBlock(
            tool_use_id=tool_use_id,
            content=f"Unknown tool: {tool_name}",
            is_error=True,
        )

    try:
        parsed_input = tool.input_model.model_validate(tool_input)
    except Exception as exc:
        if context.telemetry is not None:
            context.telemetry.record(
                model=context.model,
                tool_name=tool_name,
                success=False,
                error_type="input_validation",
                error_detail=str(exc),
            )
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
                    if context.telemetry is not None:
                        context.telemetry.record(
                            model=context.model,
                            tool_name=tool_name,
                            success=False,
                            error_type="permission_denied",
                            error_detail=f"User denied permission for {tool_name}",
                        )
                    return ToolResultBlock(
                        tool_use_id=tool_use_id,
                        content=f"Permission denied for {tool_name}",
                        is_error=True,
                    )
            else:
                if context.telemetry is not None:
                    context.telemetry.record(
                        model=context.model,
                        tool_name=tool_name,
                        success=False,
                        error_type="permission_denied",
                        error_detail=decision.reason or f"Permission denied for {tool_name}",
                    )
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

    # Sprint 10: record tool call for divergence detection
    if context.divergence_detector is not None:
        context.divergence_detector.record_tool_call(
            tool_name=tool_name,
            args=tool_input,
            result=tool_result.content,
            success=not tool_result.is_error,
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
        model_router: object | None = None,
        divergence_detector: object | None = None,
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
        self._post_task_hook: Callable | None = None
        self._tool_trace: list[dict] = []
        # Sprint 10
        self._model_router = model_router
        self._divergence_detector = divergence_detector

    def set_post_task_hook(self, hook: Callable) -> None:
        """Register a callback invoked after each completed task.

        The hook is called with ``(task_description, tool_trace)`` and
        should return a coroutine (e.g. ``SkillCreator.maybe_create``).
        """
        self._post_task_hook = hook

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
            model_router=self._model_router,
            divergence_detector=self._divergence_detector,
        )

        last_text = ""
        last_usage = UsageSnapshot()
        turns = 0
        self._tool_trace = []

        async for event, usage in run_loop(context, messages):
            if isinstance(event, AssistantTurnComplete):
                last_text = event.message.text
                last_usage = event.usage
                turns += 1
            elif isinstance(event, ToolExecutionCompleted):
                self._tool_trace.append({
                    "tool_name": event.tool_name,
                    "result": (event.output or "")[:200],
                    "is_error": event.is_error,
                })
            elif isinstance(event, AssistantTextDelta):
                pass  # streaming deltas — consumed silently here

        result = RunResult(
            text=last_text,
            messages=messages,
            usage=last_usage,
            turns=turns,
        )

        # Post-task learning hook — auto-generate skills from traces
        if self._post_task_hook and self._tool_trace:
            try:
                await self._post_task_hook(user_message, self._tool_trace)
            except Exception:
                log.debug("Post-task hook failed", exc_info=True)
            self._tool_trace = []

        return result

    def run(
        self,
        system_prompt: str,
        user_message: str,
        tools: list | None = None,
    ) -> RunResult:
        """Synchronous entry point — wraps run_async() via asyncio.run()."""
        return asyncio.run(self.run_async(system_prompt, user_message, tools))
