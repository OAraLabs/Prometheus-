# PROMETHEUS.md — Agent Instructions

This is Prometheus, a sovereign AI agent harness. It runs local LLMs (llama.cpp, Ollama)
via an abstract Model Adapter Layer — no Anthropic API dependency in the agent loop.

## Architecture

```
prometheus/
  engine/       AgentLoop — the main turn loop (Sprint 1) ✓
  adapter/      Model Adapter Layer — provider abstraction (Sprint 3)
  tools/        Tool registry + builtin tools (Sprint 2)
  hooks/        PreToolUse / PostToolUse hooks (Sprint 2)
  permissions/  Security gate (Sprint 4)
  context/      Context management + compression (Sprint 4)
  providers/    ModelProvider ABC + StubProvider (Sprint 1) ✓
  gateway/      Telegram / messaging interface (Sprint 6)
  learning/     Learning loop + skill creation (Sprint 7)
  tasks/        Task persistence (Sprint 5)
  memory/       LCM + persistent memory (Sprint 5)
  skills/       Skill loading from .md files (Sprint 5)
  coordinator/  Multi-agent coordination (Sprint 8)
  telemetry/    Tool call tracking (Sprint 3)
  config/       Settings + path management ✓
```

## Key Conventions

- All extracted donor code has a provenance header: Source, Original path, License, Modified
- Imports use `from prometheus.` not `from openharness.` or `from hermes.`
- Config is loaded from `config/prometheus.yaml` via `prometheus.config`
- Paths resolve through `prometheus.config.paths` (adapted from OpenHarness)
- Python 3.11+, managed with `uv`

---

## Sprint 0: Skeleton ✓

### `prometheus.config.paths`
`src/prometheus/config/paths.py` — adapted from OpenHarness (MIT)

```python
get_config_dir() -> Path          # ~/.prometheus/ (or $PROMETHEUS_CONFIG_DIR)
get_data_dir() -> Path            # ~/.prometheus/data/
get_logs_dir() -> Path            # ~/.prometheus/logs/
get_sessions_dir() -> Path        # ~/.prometheus/data/sessions/
get_tasks_dir() -> Path           # ~/.prometheus/data/tasks/
get_cron_registry_path() -> Path  # ~/.prometheus/data/cron_jobs.json
get_project_config_dir(cwd) -> Path  # <cwd>/.prometheus/
get_workspace_dir() -> Path       # ~/.prometheus/workspace/
```

### `prometheus.config.defaults`
`src/prometheus/config/defaults.py` — constants: `DEFAULT_MODEL_PROVIDER`, `DEFAULT_CONTEXT_LIMIT`, etc.

### `config/prometheus.yaml`
Root config file. Keys: `system`, `model`, `context`, `security`, `infrastructure`, `gateway`, `learning`.

---

## Sprint 1: Agent Loop ✓

### `prometheus.engine.messages`
`src/prometheus/engine/messages.py` — adapted from OpenHarness (MIT)

```python
class TextBlock(BaseModel):         # type="text", text: str
class ToolUseBlock(BaseModel):      # type="tool_use", id, name, input: dict
class ToolResultBlock(BaseModel):   # type="tool_result", tool_use_id, content, is_error
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock

class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ContentBlock]
    @classmethod from_user_text(text) -> ConversationMessage
    @property text -> str           # concatenated TextBlocks
    @property tool_uses -> list[ToolUseBlock]
    def to_api_param() -> dict      # OpenAI wire format
```

### `prometheus.engine.usage`
`src/prometheus/engine/usage.py`

```python
class UsageSnapshot(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    @property total_tokens -> int
```

### `prometheus.engine.cost_tracker`
`src/prometheus/engine/cost_tracker.py`

```python
class CostTracker:
    def add(usage: UsageSnapshot) -> None
    @property total -> UsageSnapshot
```

### `prometheus.engine.stream_events`
`src/prometheus/engine/stream_events.py`

```python
@dataclass AssistantTextDelta:      # text: str
@dataclass AssistantTurnComplete:   # message: ConversationMessage, usage: UsageSnapshot
@dataclass ToolExecutionStarted:    # tool_name: str, tool_input: dict
@dataclass ToolExecutionCompleted:  # tool_name: str, output: str, is_error: bool
StreamEvent = AssistantTextDelta | AssistantTurnComplete | ToolExecutionStarted | ToolExecutionCompleted
```

### `prometheus.engine.agent_loop`
`src/prometheus/engine/agent_loop.py` — adapted from OpenHarness `query.py` (MIT)

```python
@dataclass RunResult:
    text: str
    messages: list[ConversationMessage]
    usage: UsageSnapshot
    turns: int

@dataclass LoopContext:
    provider: ModelProvider         # required
    model: str
    system_prompt: str
    max_tokens: int
    tool_registry: object | None    # wired in Sprint 2
    permission_checker: object | None  # wired in Sprint 4
    hook_executor: object | None    # wired in Sprint 2
    cwd: Path
    max_turns: int = 200

async def run_loop(context: LoopContext, messages: list[ConversationMessage])
    -> AsyncIterator[tuple[StreamEvent, UsageSnapshot | None]]

class AgentLoop:
    def __init__(provider, model="qwen3.5-32b", max_tokens=4096, max_turns=200,
                 tool_registry=None, hook_executor=None, permission_checker=None, cwd=None)
    async def run_async(system_prompt, user_message, tools=None) -> RunResult
    def run(system_prompt, user_message, tools=None) -> RunResult  # asyncio.run() wrapper
```

### `prometheus.providers.base`
`src/prometheus/providers/base.py` — replaces OpenHarness `SupportsStreamingMessages` Protocol

```python
@dataclass ApiMessageRequest:
    model: str
    messages: list[ConversationMessage]
    system_prompt: str | None
    max_tokens: int = 4096
    tools: list[dict] = []

@dataclass ApiTextDeltaEvent:       # text: str
@dataclass ApiMessageCompleteEvent: # message, usage, stop_reason

class ModelProvider(ABC):
    @abstractmethod
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
```

### `prometheus.providers.stub`
`src/prometheus/providers/stub.py` — new code, implements `ModelProvider`

```python
class StubProvider(ModelProvider):
    def __init__(base_url="http://localhost:8080", timeout=120.0)
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
    # Connects to /v1/chat/completions (OpenAI-compatible)
    # Handles SSE streaming, text + tool_calls, retry on 429/500/502/503
```

### Wiring

```python
from prometheus.engine import AgentLoop
from prometheus.providers.stub import StubProvider

provider = StubProvider(base_url="http://localhost:8080")
loop = AgentLoop(provider=provider)
result = loop.run(system_prompt="You are helpful.", user_message="Hello")
print(result.text)
```

---

## Sprint 2: Tools + Hooks ✓

### `prometheus.tools.base`
`src/prometheus/tools/base.py` — adapted from OpenHarness (MIT)

```python
@dataclass ToolExecutionContext:
    cwd: Path
    metadata: dict[str, Any]          # carries tool_registry, ask_user_prompt, etc.

@dataclass(frozen=True) ToolResult:
    output: str
    is_error: bool = False
    metadata: dict[str, Any]

class BaseTool(ABC):
    name: str
    description: str
    input_model: type[BaseModel]
    async def execute(arguments, context) -> ToolResult
    def is_read_only(arguments) -> bool
    def to_api_schema() -> dict        # Anthropic format: {name, description, input_schema}
    def to_openai_schema() -> dict     # OpenAI format: {type: "function", function: {...}}

class ToolRegistry:
    def register(tool: BaseTool) -> None
    def get(name: str) -> BaseTool | None
    def list_tools() -> list[BaseTool]
    def to_api_schema() -> list[dict]          # Anthropic format
    def list_schemas() -> list[dict]           # alias for to_api_schema()
    def to_openai_schemas() -> list[dict]      # OpenAI function-calling format
    def list_schemas_for_task(task_description: str) -> list[dict]
    # keyword-matches task against tool name+description; falls back to all schemas
```

### `prometheus.tools.builtin`
`src/prometheus/tools/builtin/` — adapted from OpenHarness (MIT)

```python
# All import from prometheus.tools.builtin

BashTool(workspace=None, max_output=10_000)
    # name="bash" — runs /bin/bash -lc <command>
    # workspace locking: raises error if cwd resolves outside workspace root
    # configurable timeout (default 30s, max 600s)
    # output truncation at max_output chars

FileReadTool()
    # name="read_file" — reads text files with line numbers
    # supports offset + limit for windowed reads (default 200 lines)
    # rejects binary files (null bytes)

FileWriteTool()
    # name="write_file" — creates or overwrites files
    # auto-creates parent directories

FileEditTool()
    # name="edit_file" — str_replace semantics (old_str → new_str)
    # replace_all=True for global replacement

GrepTool()
    # name="grep" — pure-Python regex search across file tree
    # file_glob filter, case_sensitive flag, limit (default 200 matches)

GlobTool()
    # name="glob" — glob pattern file listing
    # optional root override, limit (default 200 paths)
```

### `prometheus.hooks`
`src/prometheus/hooks/` — adapted from OpenHarness (MIT)

```python
# events.py
class HookEvent(str, Enum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"

# schemas.py — Pydantic hook definitions
CommandHookDefinition(command, timeout_seconds=30, matcher=None, block_on_failure=False)
HttpHookDefinition(url, headers={}, timeout_seconds=30, matcher=None, block_on_failure=False)
PromptHookDefinition(prompt, model=None, timeout_seconds=30, matcher=None, block_on_failure=True)
AgentHookDefinition(prompt, model=None, timeout_seconds=60, matcher=None, block_on_failure=True)
# matcher: fnmatch against tool_name — None matches all tools

# types.py
@dataclass(frozen=True) HookResult:
    hook_type: str; success: bool; output: str; blocked: bool; reason: str; metadata: dict

@dataclass(frozen=True) AggregatedHookResult:
    results: list[HookResult]
    @property blocked -> bool     # True if any result has blocked=True
    @property reason -> str       # first blocking reason

# registry.py — in-memory registry (loader.py deferred to Sprint 5)
class HookRegistry:
    def add(event: HookEvent, hook: HookDefinition) -> None
    def get(event: HookEvent) -> list[HookDefinition]
    def clear(event: HookEvent | None = None) -> None

# executor.py
@dataclass HookExecutionContext:
    cwd: Path
    provider: ModelProvider       # used by prompt/agent hooks
    default_model: str

class HookExecutor:
    def __init__(registry: HookRegistry, context: HookExecutionContext)
    async def execute(event: HookEvent, payload: dict) -> AggregatedHookResult
    def update_registry(registry: HookRegistry) -> None
    # Runs all hooks registered for event whose matcher matches payload["tool_name"]
    # Command hooks: /bin/bash -lc; sets PROMETHEUS_HOOK_EVENT + PROMETHEUS_HOOK_PAYLOAD env vars
    # HTTP hooks: POST {event, payload} JSON
    # Prompt/agent hooks: calls provider.stream_message(), parses {"ok": bool} JSON response
```

### Wiring into AgentLoop

`agent_loop._execute_tool_call()` already wires both hooks:

```python
# Before tool dispatch:
pre = await hook_executor.execute(HookEvent.PRE_TOOL_USE, {tool_name, tool_input, event})
if pre.blocked:
    return ToolResultBlock(is_error=True, content=pre.reason)

# Dispatch tool via registry.get(tool_name).execute(parsed_input, ToolExecutionContext(...))

# After tool dispatch:
await hook_executor.execute(HookEvent.POST_TOOL_USE, {tool_name, tool_input, tool_output, ...})
```

Full wiring example:

```python
from prometheus.engine import AgentLoop
from prometheus.providers.stub import StubProvider
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin import BashTool, FileReadTool, FileWriteTool
from prometheus.hooks import HookExecutor, HookExecutionContext, HookRegistry, HookEvent
from prometheus.hooks.schemas import CommandHookDefinition

registry = ToolRegistry()
registry.register(BashTool(workspace="/tmp/prometheus-test"))
registry.register(FileReadTool())
registry.register(FileWriteTool())

hook_reg = HookRegistry()
hook_reg.add(HookEvent.PRE_TOOL_USE, CommandHookDefinition(command="echo pre-hook ok"))

provider = StubProvider(base_url="http://localhost:8080")
hook_executor = HookExecutor(
    hook_reg,
    HookExecutionContext(cwd=Path.cwd(), provider=provider, default_model="qwen3.5-32b"),
)

loop = AgentLoop(provider=provider, tool_registry=registry, hook_executor=hook_executor)
result = loop.run(
    system_prompt="You are a coding assistant with access to tools.",
    user_message='Create hello.py with print("hello world"), then run it.',
    tools=registry.list_schemas(),
)
print(result.text)
```

---

## Sprint Status

- [x] Sprint 0: Skeleton
- [x] Sprint 1: Agent loop
- [x] Sprint 2: Tools + hooks (extract OpenHarness tools/ + hooks/)
- [ ] Sprint 3: Model Adapter Layer (novel code)
- [ ] Sprint 4: Security + context management
- [ ] Sprint 5: Skills + memory
- [ ] Sprint 6: Gateway (Telegram)
- [ ] Sprint 7: Learning loop + LCM
- [ ] Sprint 8: Multi-agent + benchmarks

---

## Running

```bash
uv sync
uv run prometheus
# or
./scripts/start.sh
```
