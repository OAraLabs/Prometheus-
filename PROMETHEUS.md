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
  learning/     Learning loop + skill creation (Sprint 7) ✓
  tasks/        Task persistence (Sprint 5)
  memory/       LCM + persistent memory (Sprint 5)
  skills/       Skill loading from .md files (Sprint 5)
  coordinator/  Multi-agent coordination (Sprint 8) ✓
  benchmarks/   Benchmark suite + runner (Sprint 8) ✓
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

## Sprint 3: Model Adapter Layer ✓

### `prometheus.adapter` — ModelAdapter
`src/prometheus/adapter/__init__.py` — novel code

```python
class ModelAdapter:
    def __init__(formatter=None, strictness="NONE", max_retries=3)
    # formatter  defaults to AnthropicFormatter (passthrough)
    # strictness "NONE" | "MEDIUM" | "STRICT" — controls validation depth

    def format_request(system_prompt, tools) -> (str, list[dict])
    # Calls formatter.format_system_prompt() + formatter.format_tools()
    # Returns (formatted_system_prompt, formatted_tools)

    def validate_and_repair(tool_name, tool_input, tool_registry) -> (str, dict, list[str])
    # Returns (final_name, final_input, repairs_made) or raises ValueError

    def extract_tool_calls(text, tool_registry=None) -> list[ToolUseBlock]
    # Delegates to StructuredOutputEnforcer

    def handle_retry(tool_name, error, tool_registry) -> (RetryAction, str)
    # Delegates to RetryEngine
```

### `prometheus.adapter.validator` — ToolCallValidator
`src/prometheus/adapter/validator.py` — novel code

```python
class Strictness(str, Enum):
    NONE = "NONE"    # skip all validation (Claude API)
    MEDIUM = "MEDIUM"  # validate + auto-repair (Qwen, Mistral)
    STRICT = "STRICT"  # validate + repair + coerce aggressively

@dataclass ValidationResult:
    valid: bool; error: str; error_type: str
    # error_type: unknown_tool | invalid_json | missing_param | wrong_type | extra_param

@dataclass RepairResult:
    repaired: bool; tool_name: str; tool_input: dict
    repairs_made: list[str]; error: str

class ToolCallValidator:
    def __init__(strictness: Strictness | str = Strictness.NONE)
    def validate(tool_name, tool_input, tool_registry) -> ValidationResult
    def repair(tool_name, tool_input, error, tool_registry) -> RepairResult
    # Repair strategies: fuzzy Levenshtein name match, JSON extraction from markdown,
    # type coercion (str "5" → int 5), strip unknown params
```

### `prometheus.adapter.formatter` — Model-specific prompt formatting
`src/prometheus/adapter/formatter.py` — novel code

```python
class ModelPromptFormatter(ABC):
    def format_tools(tools: list[dict]) -> list[dict]
    def format_system_prompt(base_prompt, tools, context=None) -> str
    def parse_tool_calls(raw_response: str) -> list[ToolUseBlock]

class AnthropicFormatter(ModelPromptFormatter)   # passthrough — Anthropic handles natively
class QwenFormatter(ModelPromptFormatter)        # OpenAI format + explicit examples in system prompt
class GemmaFormatter(ModelPromptFormatter)       # <tool_call>...</tool_call> native format
```

### `prometheus.adapter.retry` — RetryEngine
`src/prometheus/adapter/retry.py` — novel code

```python
class RetryAction(str, Enum):
    RETRY = "RETRY"   # send retry prompt to model
    ABORT = "ABORT"   # max retries exceeded

class RetryEngine:
    def __init__(max_retries: int = 3)
    def handle_failure(tool_name, error, tool_registry, session_key=None) -> (RetryAction, str)
    def build_retry_prompt(tool_name, error, tool_registry) -> str
    def retry_count(session_key: str) -> int
    def reset(session_key: str | None = None) -> None
```

### `prometheus.adapter.enforcer` — StructuredOutputEnforcer
`src/prometheus/adapter/enforcer.py` — novel code

```python
class StructuredOutputEnforcer:
    def extract_tool_calls(raw_response, tool_registry=None) -> list[ToolUseBlock]
    # Handles: clean JSON, JSON in ```json...``` blocks, mixed prose+JSON,
    # multiple calls, partial/truncated JSON (best-effort repair)
    # Optionally filters against tool_registry

    def generate_grammar(tool_schemas: list[dict]) -> str
    # Returns GBNF grammar string for llama.cpp constrained decoding
    # Pass to LlamaCppProvider(grammar=...) to constrain model output
```

### `prometheus.telemetry.tracker` — ToolCallTelemetry
`src/prometheus/telemetry/tracker.py` — novel code

```python
class ToolCallTelemetry:
    def __init__(db_path: str | Path = "~/.prometheus/telemetry.db")
    # SQLite storage — auto-creates parent dirs

    def record(model, tool_name, success, retries=0, latency_ms=0.0,
               error_type=None, error_detail=None) -> None

    def report() -> dict
    # Returns {
    #   "models": {"<model>": {"<tool>": {calls, successes, failures,
    #                                      success_rate, avg_retries, avg_latency_ms}}},
    #   "tools":  {"<tool>": {calls, success_rate, avg_retries, avg_latency_ms, error_types}},
    #   "total_calls": int,
    #   "overall_success_rate": float,
    # }

    def close() -> None
```

### Real Provider Implementations

#### `prometheus.providers.llama_cpp` — LlamaCppProvider
`src/prometheus/providers/llama_cpp.py` — novel code

```python
class LlamaCppProvider(ModelProvider):
    def __init__(base_url="http://localhost:8080", timeout=120.0, grammar=None)
    # grammar: optional GBNF grammar string from StructuredOutputEnforcer.generate_grammar()
    # Passes grammar to llama-server via the `grammar` request field
    def set_grammar(grammar: str | None) -> None
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
```

#### `prometheus.providers.ollama` — OllamaProvider
`src/prometheus/providers/ollama.py` — novel code

```python
class OllamaProvider(ModelProvider):
    def __init__(base_url="http://localhost:11434", timeout=120.0, force_json=False)
    # force_json=True adds format="json" to requests for structured output
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
```

#### `prometheus.providers.anthropic` — AnthropicProvider
`src/prometheus/providers/anthropic.py` — novel code

```python
class AnthropicProvider(ModelProvider):
    def __init__(api_key=None, model="claude-sonnet-4-6", timeout=120.0,
                 prompt_caching=False, base_url=...)
    # api_key: reads ANTHROPIC_API_KEY env var if not provided
    # prompt_caching=True adds cache_control headers on long system prompts (≥1024 chars)
    async def stream_message(request: ApiMessageRequest) -> AsyncIterator[ApiStreamEvent]
```

### Updated AgentLoop signature (Sprint 3 additions)

```python
class AgentLoop:
    def __init__(provider, model="qwen3.5-32b", max_tokens=4096, max_turns=200,
                 tool_registry=None, hook_executor=None, permission_checker=None,
                 adapter=None,    # NEW: ModelAdapter
                 telemetry=None,  # NEW: ToolCallTelemetry
                 cwd=None)
```

### LoopContext additions

```python
@dataclass LoopContext:
    ...
    adapter: object | None = None    # ModelAdapter — wired in Sprint 3
    telemetry: object | None = None  # ToolCallTelemetry — wired in Sprint 3
```

### Sprint 3 wiring in the agent loop

```
Before LLM call:
  adapter.format_request(system_prompt, tools)  → formatted_system, formatted_tools

After LLM response:
  if no tool_uses but text contains JSON:
    adapter.extract_tool_calls(text, registry)  → inject as ToolUseBlocks

Before tool execution:
  adapter.validate_and_repair(name, input, registry)  → repaired name + input
  on failure: adapter.handle_retry(...)  → RetryAction.RETRY + retry_prompt (returned to model)

After tool execution:
  telemetry.record(model, tool_name, success, retries, latency_ms, error_type)
```

### Sprint 3 full wiring example

```python
from prometheus.engine import AgentLoop
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin import BashTool, FileReadTool, FileWriteTool, GrepTool, GlobTool
from prometheus.adapter import ModelAdapter
from prometheus.adapter.formatter import QwenFormatter
from prometheus.telemetry.tracker import ToolCallTelemetry

registry = ToolRegistry()
for tool in [BashTool('/tmp/prom-test'), FileReadTool(), FileWriteTool(), GrepTool(), GlobTool()]:
    registry.register(tool)

adapter = ModelAdapter(formatter=QwenFormatter(), strictness='MEDIUM')
telemetry = ToolCallTelemetry('~/.prometheus/telemetry.db')
provider = LlamaCppProvider(base_url='http://localhost:8080')

loop = AgentLoop(
    provider=provider,
    tool_registry=registry,
    adapter=adapter,
    telemetry=telemetry,
)
result = loop.run(
    system_prompt='You are a coding assistant.',
    user_message='List all Python files in /tmp/prom-test, then create test.py.',
    tools=registry.list_schemas(),
)
print(result.text)
print(telemetry.report())
```

---

## Sprint 4: Security + Context Management ✓

### `prometheus.permissions.modes`
`src/prometheus/permissions/modes.py` — new code

```python
class TrustLevel(IntEnum):
    BLOCKED = 0     # always deny
    APPROVE = 1     # requires user confirmation
    AUTO = 2        # allow automatically
    AUTONOMOUS = 3  # allow (background ops)

class PermissionMode(str, Enum):
    DEFAULT = "default"       # destructive ops require approval
    STRICT = "strict"         # file writes + network require approval
    AUTONOMOUS = "autonomous" # no user confirmations
```

### `prometheus.permissions.checker`
`src/prometheus/permissions/checker.py` — new code

```python
@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool              # False for DENY and APPROVE; True for ALLOW
    requires_confirmation: bool  # True only for APPROVE
    reason: str
    action: str          # "ALLOW" | "DENY" | "APPROVE"
    trust_level: TrustLevel

    @classmethod allow(reason="", level=AUTO) -> PermissionDecision   # allowed=True
    @classmethod approve(reason="") -> PermissionDecision             # allowed=False, requires_confirmation=True
    @classmethod deny(reason) -> PermissionDecision                   # allowed=False

class SecurityGate:
    def __init__(denied_commands=None, denied_paths=None,
                 workspace_root=None, mode=PermissionMode.DEFAULT)
    @classmethod from_config(config_path=None) -> SecurityGate
    # Loads security section from prometheus.yaml

    def evaluate(tool_name, *, is_read_only=False,
                 file_path=None, command=None) -> PermissionDecision
    # Used by agent_loop.py permission_checker slot

    def pre_tool_use(tool_name, tool_input: dict, context: dict) -> PermissionDecision
    # Acceptance-test interface: gate.pre_tool_use('bash', {'command': 'rm -rf /'}, {})
```

Trust rules applied in order:
1. AUTONOMOUS mode → ALLOW all except always-blocked patterns
2. Always-blocked regex (rm -rf /, mkfs, dd to /dev/, fork bomb …) → DENY
3. denied_commands list from config → DENY
4. denied_paths list from config → DENY
5. write_file/edit_file in STRICT mode → APPROVE
6. write_file/edit_file outside workspace_root → APPROVE
7. bash with git push / curl / wget / ssh / pip install → APPROVE
8. Everything else → ALLOW

### `prometheus.permissions.sandbox`
`src/prometheus/permissions/sandbox.py` — new code

```python
class SandboxedExecution:
    def __init__(workspace: str | Path, timeout=30, max_output=10_000,
                 strip_env_keys: list[str] | None = None)
    async def run(command: str, env_override=None) -> ToolResult
    @property workspace -> Path
    # Strips API keys / tokens / secrets from subprocess env automatically
    # Enforces timeout; truncates output at max_output chars
```

### `prometheus.context.token_estimation`
`src/prometheus/context/token_estimation.py` — new code

```python
def estimate_tokens(text: str) -> int
# len(text) // 4 — 4-chars-per-token heuristic; fast, no dependencies
```

### `prometheus.context.budget`
`src/prometheus/context/budget.py` — new code

```python
@dataclass
class TokenBudget:
    effective_limit: int
    reserved_output: int = 2000
    model_overrides: dict[str, int]

    @classmethod from_config(model=None, config_path=None) -> TokenBudget
    # Reads context section from prometheus.yaml; applies model_overrides

    def add(category: str, text: str) -> None   # accumulates by category
    def reset() -> None
    @property used -> int                        # sum across all categories
    def usage_by_category() -> dict[str, int]
    def headroom() -> int                        # effective_limit - reserved_output - used
    def is_approaching_limit(threshold=0.75) -> bool
```

### `prometheus.context.truncation`
`src/prometheus/context/truncation.py` — new code

```python
class ToolResultTruncator:
    def __init__(max_tokens=4000)
    @classmethod from_config(config_path=None) -> ToolResultTruncator
    def truncate(tool_name: str, output: str) -> str
    def __call__(tool_name, output) -> str   # callable interface

    # Strategies:
    # bash      → keep last 100 lines
    # read_file → first 50 + last 50 lines + "[... N lines truncated ...]"
    # grep      → top 20 results
    # default   → hard-truncate at max_tokens*4 chars + "[truncated at N tokens]"
```

### `prometheus.context.compression`
`src/prometheus/context/compression.py` — new code

```python
class ContextCompressor:
    def __init__(budget: TokenBudget, fresh_tail_count=32)
    @classmethod from_config(budget, config_path=None) -> ContextCompressor

    def maybe_compress(messages: list[ConversationMessage]) -> list[ConversationMessage]
    # No-op if budget.is_approaching_limit() is False
    # Prunes ToolResultBlock.content from messages older than fresh_tail_count user turns
    # Replaces pruned content with "[content pruned — context compression]"
    # Full LCM summarization deferred to Sprint 7
```

### `prometheus.context.dynamic_tools`
`src/prometheus/context/dynamic_tools.py` — new code

```python
CORE_TOOLS: frozenset[str]   # {"bash", "read_file", "write_file"}

class DynamicToolLoader:
    def __init__(registry: ToolRegistry)
    def active_schemas(task_description: str | None = None) -> list[dict]
    # None → all tools; otherwise CORE_TOOLS + keyword-matched extras
    # Keywords: grep/search → grep; find/list/files → glob;
    #           edit/modify/replace/patch → edit_file
    def on_demand(tool_name: str) -> dict | None
    # Returns schema if tool registered, else None
    def all_schemas() -> list[dict]
```

### Sprint 4 wiring into AgentLoop

```python
from prometheus.permissions import SecurityGate
from prometheus.context import TokenBudget, ToolResultTruncator, ContextCompressor, DynamicToolLoader

# Permission gate (slots into existing permission_checker= param)
gate = SecurityGate.from_config()   # reads config/prometheus.yaml security section

# Context management
budget = TokenBudget.from_config(model="qwen3.5-32b")
truncator = ToolResultTruncator.from_config()
compressor = ContextCompressor.from_config(budget)

loop = AgentLoop(
    provider=provider,
    tool_registry=registry,
    permission_checker=gate,   # evaluate() called before every tool execution
    # APPROVE decisions (allowed=False, requires_confirmation=True) trigger
    # permission_prompt callback in agent_loop — if no prompt is wired, they block.
)

# Truncation + compression used outside AgentLoop (Sprint 5 will wire fully):
output = truncator.truncate(tool_name, raw_output)
messages = compressor.maybe_compress(messages)
```

---

## Sprint Status

- [x] Sprint 0: Skeleton
- [x] Sprint 1: Agent loop
- [x] Sprint 2: Tools + hooks (extract OpenHarness tools/ + hooks/)
- [x] Sprint 3: Model Adapter Layer (novel code)
- [x] Sprint 4: Security + context management
- [x] Sprint 5: Skills + memory + tasks
- [x] Sprint 6: Gateway (Telegram) + Cron + Daemon
- [ ] Sprint 7: Learning loop + LCM
- [ ] Sprint 8: Multi-agent + benchmarks

---

## Sprint 5: Skills + Memory + Tasks

### `prometheus.skills`

**`SkillDefinition`** *(frozen dataclass)* — `src/prometheus/skills/types.py`
```python
SkillDefinition(name: str, description: str, content: str, source: str, path: str | None)
```

**`SkillRegistry`** — `src/prometheus/skills/registry.py`
```python
SkillRegistry()
  .register(skill: SkillDefinition) -> None
  .get(name: str) -> SkillDefinition | None          # case-insensitive fallback
  .list_skills() -> list[SkillDefinition]             # sorted by name
```

**`loader`** — `src/prometheus/skills/loader.py`
```python
get_builtin_skills() -> list[SkillDefinition]         # reads skills/builtin/*.md
load_user_skills() -> list[SkillDefinition]           # reads ~/.prometheus/skills/*.md
load_skill_registry(cwd=None) -> SkillRegistry        # builtin + user merged
```

**Builtin skills** — `src/prometheus/skills/builtin/`: `commit.md`, `debug.md`, `plan.md`

**Wires into**: `SkillTool` (`tools/builtin/skill.py`) exposes registry to agent via `skill` tool name.

---

### `prometheus.memory`

**`MemoryStore`** *(SQLite + FTS5)* — `src/prometheus/memory/store.py`
```python
MemoryStore(db_path: str | Path | None = None)        # default: ~/.prometheus/memory.db
  # Messages
  .add_message(session_id, role, content, *, message_id, compressed) -> str
  .get_messages(session_id, *, since, compressed, limit) -> list[dict]
  # Memories
  .persist_memory(entity_type, entity_name, fact, confidence, *, relationship,
                  source_event_ids, tags, memory_id) -> str   # deduplicates on name+fact
  .search_memories(*, query, entity, entity_type, min_confidence, limit) -> list[dict]
  .get_memory(memory_id: str) -> dict | None
  # Summaries
  .add_summary(summary_text, source_message_ids, *, level, summary_id) -> str
  .get_summaries(*, level, limit) -> list[dict]
  .close() / context manager
```

Tables: `messages` (FTS5), `memories` (FTS5), `summaries`.

**`MemoryPointer`** — `src/prometheus/memory/pointer.py`
```python
MemoryPointer(pointer_path: str | Path | None = None, max_chars: int = 8000)
  .add_pointer(text: str) -> None
  .remove_pointer(text: str) -> bool
  .replace_pointer(old_text, new_text) -> bool
  .get_all() -> list[str]
  .format_for_prompt() -> str                         # "## Memory Pointers\n- ..."
  .clear() -> None
```
Backed by `~/.prometheus/MEMORY.md` with `fcntl` exclusive locking and char-limit pruning.

**`FileMemoryStore`** — `src/prometheus/memory/hermes_memory_tool.py`
```python
FileMemoryStore(path: Path, max_chars: int)
  .add(entry: str) -> str
  .replace(old_text, new_text) -> str
  .remove(text: str) -> str
  .list_entries() -> list[str]
  .format_for_prompt(header: str) -> str

get_memory_store() -> FileMemoryStore       # MEMORY.md, 12 000 char cap
get_user_store() -> FileMemoryStore         # USER.md, 8 000 char cap
format_memory_for_prompt() -> str           # both sections combined
```

**`MemoryTool`** *(BaseTool)* — `src/prometheus/memory/hermes_memory_tool.py`
```
name="memory"  inputs: operation (add|replace|remove|list), target (memory|user),
               entry, old_entry
```

**`MemoryExtractor`** — `src/prometheus/memory/extractor.py`
```python
MemoryExtractor(store: MemoryStore, provider: ModelProvider, *, model, obsidian_writer, batch_size)
  .run_once(session_id=None) -> int                   # returns # memories persisted
  .run_forever(interval=1800, session_id=None) -> None # 30-min background loop
```
Reads `messages` table → batches of 15 → LLM extraction prompt → writes to `memories` + optional Obsidian vault.

**`ObsidianWriter`** — `src/prometheus/memory/extractor.py`
```python
ObsidianWriter(vault_path: str | Path)
  .write_fact(fact: dict) -> None   # appends to vault/Memory/<entity_name>.md
```

---

### `prometheus.tasks`

**`TaskType`** — `"local_bash" | "local_agent" | "remote_agent" | "in_process_teammate"`
**`TaskStatus`** — `"pending" | "running" | "completed" | "failed" | "killed"`

**`TaskRecord`** *(dataclass)* — `src/prometheus/tasks/types.py`
```python
TaskRecord(id, type, status, description, cwd, output_file, command, prompt,
           created_at, started_at, ended_at, return_code, metadata)
```

**`BackgroundTaskManager`** — `src/prometheus/tasks/manager.py`
```python
BackgroundTaskManager()
  # Async create
  await .create_shell_task(*, command, description, cwd, task_type) -> TaskRecord
  await .create_agent_task(*, prompt, description, cwd, task_type, model, api_key, command) -> TaskRecord
  # Read
  .get_task(task_id: str) -> TaskRecord | None
  .list_tasks(*, status: TaskStatus | None) -> list[TaskRecord]
  .read_task_output(task_id, *, max_bytes=12000) -> str
  # Mutate
  .update_task(task_id, *, description, progress, status_note) -> TaskRecord
  await .stop_task(task_id) -> TaskRecord
  await .write_to_task(task_id, data: str) -> None

get_task_manager() -> BackgroundTaskManager   # process-wide singleton
```
Output streamed to `~/.prometheus/data/tasks/<id>.log`. Agent tasks restart on broken pipe.

---

### New tools (`prometheus.tools.builtin`)

| File | Tool name | Input fields |
|------|-----------|-------------|
| `skill.py` | `skill` | `name` |
| `task_create.py` | `task_create` | `type`, `description`, `command`, `prompt`, `model` |
| `task_get.py` | `task_get` | `task_id` |
| `task_list.py` | `task_list` | `status?` |
| `task_update.py` | `task_update` | `task_id`, `description?`, `progress?`, `status_note?` |
| `task_stop.py` | `task_stop` | `task_id` |
| `task_output.py` | `task_output` | `task_id`, `max_bytes?` |
| `todo_write.py` | `todo_write` | `item`, `checked?`, `path?` |

All extend `BaseTool` from `prometheus.tools.base`.

### Wiki Maintenance

Transforms extracted memory facts into a persistent, cross-linked Markdown
wiki at `~/.prometheus/wiki/`. Inspired by the LLM Wiki pattern — knowledge
is compiled once and kept current, not re-derived on every query.

```
~/.prometheus/wiki/
├── index.md            # Auto-generated, organized by category
├── log.md              # Append-only chronological compile log
├── .last_compile_ts    # Watermark for incremental compilation
├── people/             # person entities
├── clients/            # organization entities
├── projects/           # task + tool entities
├── topics/             # concept, place, preference entities
└── queries/            # Auto-filed query results (compounding loop)
```

**`WikiCompiler`** — `src/prometheus/memory/wiki_compiler.py`
```python
WikiCompiler(store: MemoryStore, wiki_root: Path | None = None)  # default: ~/.prometheus/wiki/
  .compile(new_facts: list[dict]) -> None   # thread-safe (threading.Lock)
  .get_watermark() -> float                 # last compile timestamp from .last_compile_ts
  .wiki_root -> Path                        # property
```

Entity type mapping: `person→people/`, `organization→clients/`,
`task|tool→projects/`, `concept|place|preference→topics/`.

Page creation requires 2+ `mention_count` in `MemoryStore`. Pages use YAML
frontmatter (`type`, `first_seen`, `last_updated`, `source_count`) and
`[[wiki-links]]` for cross-references detected by substring match (3+ char
entity names). Regenerates `index.md` and appends to `log.md` after each pass.

**Wiring into `MemoryExtractor`** (Sprint 5):
```python
# extractor.py — new parameter
MemoryExtractor(..., post_extract_callback: Callable[[list[dict]], None] | None = None)
  .run_once(session_id) -> tuple[int, list[dict]]   # was: -> int
  # run_forever() calls post_extract_callback(facts) after each pass
```
`daemon.py` passes `WikiCompiler.compile` as `post_extract_callback`.
`lcm_engine.py:238` updated to unpack the new tuple return.

**`WikiCompileTool`** *(BaseTool)* — `src/prometheus/tools/builtin/wiki_compile.py`
```
name="wiki_compile"  inputs: entity_name? (str | None)
```
Module-level `set_wiki_compiler(compiler, store)` setter (same pattern as
`lcm_grep.py`). Reads `.last_compile_ts` watermark, queries `MemoryStore`
for newer facts, calls `compile()`.

**`WikiQueryTool`** *(BaseTool)* — `src/prometheus/tools/builtin/wiki_query.py`
```
name="wiki_query"  inputs: query (str)
```
Reads `index.md`, scores entries by keyword overlap, reads top-5 pages,
returns concatenated content. **Compounding loop:** files result to
`wiki/queries/` when it spans 2+ pages *and* exceeds 200 chars — prevents
trivial lookups from becoming wiki pages.

Both tools registered in `build_tool_registry()` (`scripts/daemon.py`) and
exported from `prometheus.tools.builtin.__init__`.

---

## Sprint 6: Gateway (Telegram) + Cron + Daemon

### `prometheus.gateway.config`
`src/prometheus/gateway/config.py` — novel code

```python
class Platform(str, Enum):           # TELEGRAM, CLI, API

@dataclass
class PlatformConfig:
    platform: Platform
    token: str = ""
    webhook_url: str | None = None
    allowed_chat_ids: list[int] = []
    proxy_url: str | None = None
    max_message_length: int = 4096
    parse_mode: str = "MarkdownV2"
    connect_timeout: float = 30.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    extra: dict[str, Any] = {}

    @property is_restricted -> bool
    def chat_allowed(chat_id: int) -> bool
```

### `prometheus.gateway.platform_base`
`src/prometheus/gateway/platform_base.py` — novel code (architecture inspired by Hermes)

```python
class MessageType(str, Enum):        # TEXT, COMMAND, CALLBACK, EDITED, PHOTO, DOCUMENT, VOICE

@dataclass
class MessageEvent:
    chat_id: int
    user_id: int
    text: str
    message_id: int
    platform: Platform
    message_type: MessageType = TEXT
    username: str | None = None
    timestamp: datetime = now(utc)
    raw: dict[str, Any] = {}
    def session_key() -> str         # "{platform}:{chat_id}"

@dataclass(frozen=True)
class SendResult:
    success: bool
    message_id: int | None = None
    error: str | None = None

class BasePlatformAdapter(ABC):
    def __init__(config: PlatformConfig)
    @property platform -> Platform
    @property running -> bool
    async def start() -> None
    async def stop() -> None
    async def send(chat_id, text, *, reply_to=None, parse_mode=None) -> SendResult
    async def on_message(event: MessageEvent) -> None
```

### `prometheus.gateway.telegram`
`src/prometheus/gateway/telegram.py` — novel code (architecture inspired by Hermes)

```python
def escape_markdown_v2(text: str) -> str
def chunk_message(text: str, max_length=4096) -> list[str]

class TelegramAdapter(BasePlatformAdapter):
    def __init__(config: PlatformConfig, agent_loop: AgentLoop,
                 tool_registry: ToolRegistry,
                 system_prompt="You are Prometheus, a helpful AI assistant.")
    async def start() -> None           # builds Application, registers handlers, starts polling
    async def stop() -> None            # graceful shutdown
    async def send(chat_id, text, *, reply_to=None, parse_mode=None) -> SendResult
    async def on_message(event: MessageEvent) -> None
    # Internal:
    async def _cmd_start(update, context) -> None       # /start handler
    async def _cmd_clear(update, context) -> None       # /clear handler
    async def _handle_text(update, context) -> None     # text message handler
    async def _dispatch_to_agent(event: MessageEvent) -> None
    # Dispatches to agent_loop.run_async(system_prompt, event.text, tools)
    # Sends result.text back via self.send()
```

### `prometheus.gateway.telegram_network`
`src/prometheus/gateway/telegram_network.py` — novel code

```python
@dataclass
class TelegramNetworkConfig:
    proxy_url: str | None = None
    base_url: str | None = None
    base_file_url: str | None = None
    connect_timeout: float = 30.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    pool_timeout: float = 10.0
    def to_bot_kwargs() -> dict
    def to_request_kwargs() -> dict
```

### `prometheus.gateway.cron_service`
`src/prometheus/gateway/cron_service.py` — adapted from OpenHarness (MIT)

```python
def load_cron_jobs() -> list[dict]
def save_cron_jobs(jobs: list[dict]) -> None
def validate_cron_expression(expression: str) -> bool
def next_run_time(expression: str, base: datetime | None) -> datetime
def upsert_cron_job(job: dict) -> None
def delete_cron_job(name: str) -> bool
def get_cron_job(name: str) -> dict | None
def set_job_enabled(name: str, enabled: bool) -> bool
def mark_job_run(name: str, *, success: bool) -> None
# JSON registry at ~/.prometheus/data/cron_jobs.json via get_cron_registry_path()
```

### `prometheus.gateway.cron_scheduler`
`src/prometheus/gateway/cron_scheduler.py` — adapted from OpenHarness (MIT)

```python
TICK_INTERVAL_SECONDS = 30

# History
def append_history(entry: dict) -> None         # JSONL at data/cron_history.jsonl
def load_history(*, limit=50, job_name=None) -> list[dict]

# PID management
def read_pid() -> int | None
def write_pid() -> None
def remove_pid() -> None
def is_scheduler_running() -> bool
def stop_scheduler() -> bool

# Job execution
async def execute_job(job: dict) -> dict        # /bin/bash -lc, 300s timeout
async def run_scheduler_loop(*, once=False) -> None   # main loop, signal-aware
def scheduler_status() -> dict
```

### `prometheus.gateway.heartbeat`
`src/prometheus/gateway/heartbeat.py` — novel code

```python
class Heartbeat:
    def __init__(*, interval=30, gateway: BasePlatformAdapter | None = None,
                 task_manager: BackgroundTaskManager | None = None)
    async def check() -> dict           # cron_jobs_due, gateway_running, tasks_running
    async def run_forever() -> None     # loop until stop()
    def stop() -> None
```

### `prometheus.gateway.archive_writer`
`src/prometheus/gateway/archive_writer.py` — novel code (inspired by OpenClaw)

```python
class ArchiveWriter:
    def __init__(path: str | Path | None = None)   # default: ~/.prometheus/data/archive.jsonl
    def archive_event(event_type: str, data: dict | None = None) -> None
    def read_events(*, event_type=None, limit=100) -> list[dict]
```

### New tools (`prometheus.tools.builtin`)

| File | Tool name | Input fields |
|------|-----------|-------------|
| `cron_create.py` | `cron_create` | `name`, `schedule`, `command`, `cwd?`, `enabled?` |
| `cron_delete.py` | `cron_delete` | `name` |
| `cron_list.py` | `cron_list` | *(none)* — read-only |

All adapted from OpenHarness (MIT), imports changed to `prometheus.*`.

### `scripts/daemon.py` — Main daemon entry point
```python
def load_config(config_path=None) -> dict
def build_tool_registry(workspace=None) -> ToolRegistry
async def run_daemon(args) -> None
def main() -> None

# Wires: LlamaCppProvider → ToolRegistry (builtin + cron tools) → AgentLoop
#        → TelegramAdapter + Heartbeat + CronScheduler + MemoryExtractor
# All run via asyncio.gather with SIGTERM/SIGINT graceful shutdown
```

Usage:
```bash
python scripts/daemon.py --telegram-only --debug
python scripts/daemon.py --config config/prometheus.yaml
```

### `scripts/health_check.sh`
Checks cron scheduler PID, daemon log freshness, cron registry, archive events.

### `scripts/prometheus.service`
Systemd unit file for running daemon as a service.

### Sprint 6 wiring

```python
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.tools.base import ToolRegistry
from prometheus.tools.builtin import BashTool, FileReadTool, FileWriteTool, CronCreateTool, ...
from prometheus.engine import AgentLoop
from prometheus.gateway import TelegramAdapter, PlatformConfig, Platform, Heartbeat
from prometheus.gateway.cron_scheduler import run_scheduler_loop

provider = LlamaCppProvider(base_url="http://localhost:8080")
registry = ToolRegistry()
# register all builtin + cron tools ...
agent_loop = AgentLoop(provider=provider, tool_registry=registry)

tg_config = PlatformConfig(platform=Platform.TELEGRAM, token="BOT_TOKEN")
telegram = TelegramAdapter(config=tg_config, agent_loop=agent_loop,
                           tool_registry=registry)

await asyncio.gather(
    telegram.start(),
    run_scheduler_loop(),
    Heartbeat(gateway=telegram).run_forever(),
)
```

### Dependencies added
- `python-telegram-bot>=21.0` — Telegram Bot API
- `croniter>=2.0` — cron expression parsing

---

## Sprint 7: Learning Loop + LCM ✓

### File tree additions

```
learning/
  __init__.py         PeriodicNudge, SkillCreator, SkillRefiner
  nudge.py            Self-evaluation injection every N turns
  skill_creator.py    Auto-generate SKILL.md from tool traces
  skill_refiner.py    Refine skills based on execution deviations

memory/
  lcm_types.py        MessagePart, SummaryNode, CompactionConfig, AssemblyResult, CompactionResult, LCMStats
  lcm_fts5.py         sanitize_fts5_query(), tokenize_for_fts5()
  lcm_conversation_store.py  SQLite message store with FTS5
  lcm_summary_store.py       SQLite DAG summary store with FTS5
  lcm_compaction.py    Incremental DAG compaction engine
  lcm_assembler.py     Context assembly from summaries + fresh tail
  lcm_summarize.py     LLM summarization with circuit breaker
  lcm_engine.py        Top-level LCM orchestrator

tools/builtin/
  lcm_grep.py          FTS5 search over messages + summaries
  lcm_expand.py        Expand summary back to source messages
  lcm_describe.py      Inspect summary metadata / LCM stats

context/
  system_prompt.py     Prometheus identity + SYSTEM_PROMPT_DYNAMIC_BOUNDARY
  prompt_assembler.py  Runtime system prompt assembly (static + dynamic)
  prometheusmd.py      PROMETHEUS.md discovery and loading
```

### `prometheus.learning.nudge.PeriodicNudge`
`src/prometheus/learning/nudge.py`

```python
@dataclass
class PeriodicNudge:
    interval: int = 15          # turns between nudges
    prompt: str = _NUDGE_PROMPT # < 200 tokens
    enabled: bool = True

    @classmethod
    def from_config(cls, config_path: str | None = None) -> PeriodicNudge
    def maybe_inject(self, turn_count: int) -> dict | None  # returns nudge msg or None
    def reset(self) -> None
```

### `prometheus.learning.skill_creator.SkillCreator`
`src/prometheus/learning/skill_creator.py`

```python
class SkillCreator:
    def __init__(self, provider: ModelProvider, *, model: str = "default",
                 min_tool_calls: int = 3, auto_dir: Path | None = None) -> None

    @classmethod
    def from_config(cls, provider: ModelProvider, config_path: str | None = None) -> SkillCreator
    async def maybe_create(self, task_description: str,
                           tool_trace: list[dict]) -> Path | None
    # Writes to ~/.prometheus/skills/auto/<slug>.md
```

### `prometheus.learning.skill_refiner.SkillRefiner`
`src/prometheus/learning/skill_refiner.py`

```python
class SkillRefiner:
    def __init__(self, provider: ModelProvider, *, model: str = "default") -> None

    async def maybe_refine(self, skill_path: Path,
                           tool_trace: list[dict], outcome: str) -> bool
    # Creates .bak backup before updating
```

### `prometheus.memory.lcm_types`
`src/prometheus/memory/lcm_types.py`

```python
@dataclass
class MessagePart:
    role: str; content: str; timestamp: float; message_id: str
    session_id: str = ""; turn_index: int = 0; token_count: int = 0

@dataclass
class SummaryNode:
    id: str; parent_ids: list[str]; source_message_ids: list[str]
    summary_text: str; depth: int = 0; token_count: int = 0
    created_at: float; is_leaf: bool = True

@dataclass
class CompactionConfig:
    context_threshold: int = 18_000; fresh_tail_count: int = 32
    summary_model: str = "default"; max_summary_depth: int = 5
    compaction_batch_size: int = 10

@dataclass
class AssemblyResult:
    summaries: list[SummaryNode]; fresh_messages: list[MessagePart]
    total_tokens: int = 0; compression_ratio: float = 1.0

@dataclass
class CompactionResult:
    summaries_created: int = 0; messages_compacted: int = 0
    new_depth: int = 0; tokens_saved: int = 0

@dataclass
class LCMStats:
    total_messages: int = 0; total_summaries: int = 0; max_depth: int = 0
    total_compactions: int = 0; last_compaction_at: float | None = None
```

### `prometheus.memory.lcm_conversation_store.LCMConversationStore`
`src/prometheus/memory/lcm_conversation_store.py` — SQLite + FTS5, WAL mode, shared `lcm.db`

```python
class LCMConversationStore:
    def __init__(self, db_path: Path | None = None) -> None

    def insert_message(self, msg: MessagePart) -> str
    def get_messages(self, session_id: str, *, since_turn: int | None = None,
                     limit: int = 500) -> list[MessagePart]
    def get_fresh_tail(self, session_id: str, count: int) -> list[MessagePart]
    def mark_compacted(self, message_ids: list[str]) -> int
    def search(self, query: str, *, session_id: str | None = None,
               limit: int = 20) -> list[MessagePart]
    def count_uncompacted(self, session_id: str) -> int
    def close(self) -> None
```

### `prometheus.memory.lcm_summary_store.LCMSummaryStore`
`src/prometheus/memory/lcm_summary_store.py` — DAG summary storage, shared `lcm.db`

```python
class LCMSummaryStore:
    def __init__(self, db_path: Path | None = None) -> None

    def insert_summary(self, node: SummaryNode) -> str  # marks parents non-leaf
    def get_by_id(self, summary_id: str) -> SummaryNode | None
    def get_leaves(self, *, max_depth: int | None = None) -> list[SummaryNode]
    def get_by_depth(self, depth: int, *, limit: int = 100) -> list[SummaryNode]
    def get_roots(self) -> list[SummaryNode]
    def get_children(self, parent_id: str) -> list[SummaryNode]
    def get_ancestors(self, node_id: str) -> list[SummaryNode]  # BFS walk up DAG
    def search(self, query: str, *, limit: int = 20) -> list[SummaryNode]
    def get_stats(self) -> dict
    def close(self) -> None
```

### `prometheus.memory.lcm_compaction.LCMCompactor`
`src/prometheus/memory/lcm_compaction.py` — the heart of LCM

```python
class LCMCompactor:
    def __init__(self, conversation_store: LCMConversationStore,
                 summary_store: LCMSummaryStore,
                 summarizer: LCMSummarizer,
                 config: CompactionConfig) -> None

    async def compact(self, session_id: str) -> CompactionResult
    def should_compact(self, session_id: str) -> bool
    # Internal: _summarize_messages, _cascade_summaries (6+ leaves → merge up),
    #           _summarize_nodes
```

### `prometheus.memory.lcm_assembler.LCMAssembler`
`src/prometheus/memory/lcm_assembler.py`

```python
class LCMAssembler:
    def __init__(self, conversation_store: LCMConversationStore,
                 summary_store: LCMSummaryStore,
                 config: CompactionConfig) -> None

    def assemble(self, session_id: str, token_budget: int) -> AssemblyResult
    def format_summary_preamble(self, summaries: list[SummaryNode]) -> str
```

### `prometheus.memory.lcm_summarize.LCMSummarizer`
`src/prometheus/memory/lcm_summarize.py`

```python
class LCMCircuitBreakerOpen(Exception): ...

class LCMSummarizer:
    def __init__(self, provider: ModelProvider, *, model: str = "default",
                 max_retries: int = 2) -> None

    async def summarize_messages(self, messages: list[MessagePart]) -> str
    async def summarize_summaries(self, summaries: list[SummaryNode]) -> str
    def reset(self) -> None
    @property
    def circuit_open(self) -> bool
    # Circuit breaker: 3 consecutive failures → LCMCircuitBreakerOpen
```

### `prometheus.memory.lcm_engine.LCMEngine`
`src/prometheus/memory/lcm_engine.py` — top-level orchestrator

```python
class LCMEngine:
    def __init__(self, provider: ModelProvider, *,
                 config: CompactionConfig | None = None,
                 db_path: Path | None = None) -> None

    async def ingest(self, session_id: str, role: str, content: str,
                     *, turn_index: int = 0) -> str
    def assemble(self, session_id: str, token_budget: int) -> AssemblyResult
    async def compact(self, session_id: str) -> CompactionResult
    async def maybe_compact(self, session_id: str) -> CompactionResult | None
    async def pre_compaction_flush(self, session_id: str) -> None
    def get_stats(self, session_id: str) -> LCMStats
    def close(self) -> None
    # Context manager support: __enter__, __exit__
```

### LCM Agent Tools
`src/prometheus/tools/builtin/lcm_grep.py`, `lcm_expand.py`, `lcm_describe.py`

```python
# lcm_grep — FTS5 search over messages + summaries
class LCMGrepTool(BaseTool):
    name = "lcm_grep"
    # Input: query, session_id?, search_target ("messages"|"summaries"|"both"), limit

# lcm_expand — expand a summary back to originals
class LCMExpandTool(BaseTool):
    name = "lcm_expand"
    # Input: summary_id, depth?

# lcm_describe — inspect summary metadata or overall stats
class LCMDescribeTool(BaseTool):
    name = "lcm_describe"
    # Input: summary_id?, session_id?

# Engine wiring:
set_lcm_engine(engine: LCMEngine) -> None  # module-level setter for tool access
```

### `prometheus.context.system_prompt`
`src/prometheus/context/system_prompt.py`

```python
SYSTEM_PROMPT_DYNAMIC_BOUNDARY: str = "--- SYSTEM_PROMPT_DYNAMIC_BOUNDARY ---"

def build_system_prompt(custom_prompt: str | None = None,
                        env: EnvironmentInfo | None = None,
                        cwd: str | None = None) -> str
```

### `prometheus.context.prompt_assembler`
`src/prometheus/context/prompt_assembler.py`

```python
def build_runtime_system_prompt(
    *, cwd: str, config: dict | None = None,
    memory_content: str = "", skills: list | None = None,
    task_state: str = "", loaded_skill_content: str = "",
) -> str
# Assembles: STATIC (base prompt + env) + DYNAMIC_BOUNDARY + DYNAMIC (PROMETHEUS.md, memory, skills, tasks)
```

### `prometheus.context.prometheusmd`
`src/prometheus/context/prometheusmd.py`

```python
def discover_prometheus_md_files(cwd: str | Path) -> list[Path]
def load_prometheus_md_prompt(cwd: str | Path, *, max_chars_per_file: int = 12000) -> str | None
```

### Sprint 7 wiring

```python
from prometheus.providers.llama_cpp import LlamaCppProvider
from prometheus.memory.lcm_engine import LCMEngine
from prometheus.learning import PeriodicNudge, SkillCreator, SkillRefiner
from prometheus.tools.builtin.lcm_grep import set_lcm_engine
from prometheus.context.prompt_assembler import build_runtime_system_prompt

provider = LlamaCppProvider(base_url="http://localhost:8080")

# LCM engine — manages conversation persistence + DAG compaction
lcm = LCMEngine(provider=provider)
set_lcm_engine(lcm)  # wire into lcm_grep/expand/describe tools

# Learning loop
nudge = PeriodicNudge.from_config()
skill_creator = SkillCreator(provider)
skill_refiner = SkillRefiner(provider)

# Agent loop integration:
# 1. On each message: await lcm.ingest(session_id, role, content, turn_index=n)
# 2. Before LLM call: result = lcm.assemble(session_id, token_budget=24000)
# 3. Every N turns: nudge_msg = nudge.maybe_inject(turn_count)
# 4. After task: skill_path = await skill_creator.maybe_create(desc, trace)
# 5. After skill-guided task: await skill_refiner.maybe_refine(path, trace, outcome)
# 6. Periodically: await lcm.maybe_compact(session_id)
```

---

## Running

```bash
uv sync
uv run prometheus
# or
./scripts/start.sh

# Daemon mode (Sprint 6):
python scripts/daemon.py --telegram-only --debug

# Benchmarks (Sprint 8):
python -m prometheus.benchmarks.runner --model qwen3.5-32b --tier 1
python -m prometheus.benchmarks.runner --tier 2 --json
python -m prometheus.benchmarks.runner --case t1_bash_echo --verbose
```

---

## Sprint 8: Multi-Agent + Benchmark Suite ✓

### `prometheus.coordinator.agent_definitions` — AgentDefinition
`src/prometheus/coordinator/agent_definitions.py` — adapted from OpenHarness (MIT)

```python
@dataclass
class AgentDefinition:
    def __init__(name, description, system_prompt="", tools=[], model="",
                 read_only=False, max_turns=50, metadata={})

# Built-in agents: general-purpose, explorer, planner, worker, verification

get_all_agent_definitions() -> dict[str, AgentDefinition]
get_agent_definition(name) -> AgentDefinition | None
register_agent_definition(defn: AgentDefinition) -> None
```

### `prometheus.coordinator.coordinator_mode` — TeamRegistry
`src/prometheus/coordinator/coordinator_mode.py` — adapted from OpenHarness (MIT)

```python
@dataclass
class TeamRecord:
    name: str; description: str; agents: list[str]; metadata: dict

class TeamRegistry:
    def create_team(name, description="", agents=None) -> TeamRecord
    def get_team(name) -> TeamRecord | None
    def list_teams() -> list[TeamRecord]
    def add_agent_to_team(team_name, agent_name) -> bool
    def remove_agent_from_team(team_name, agent_name) -> bool

get_team_registry() -> TeamRegistry   # module-level singleton
is_coordinator_mode(agent_count) -> bool
get_coordinator_system_prompt(team=None) -> str
```

### `prometheus.coordinator.subagent` — SubagentSpawner
`src/prometheus/coordinator/subagent.py` — novel code

```python
@dataclass(frozen=True)
class SubagentResult:
    agent_id: str; agent_type: str; text: str; turns: int
    success: bool; error: str | None; metadata: dict

class SubagentSpawner:
    def __init__(provider, *, parent_tool_registry=None, model="qwen3.5-32b",
                 max_tokens=4096, cwd=None, adapter=None, telemetry=None)

    async def spawn(task, *, agent_type="general-purpose", tools_subset=None,
                    model=None, system_prompt=None, max_turns=None) -> SubagentResult
    # Spawns a fresh AgentLoop with isolated messages[]. Result returned
    # without polluting the parent conversation.

    async def spawn_parallel(tasks: list[dict]) -> list[SubagentResult]
    # Runs multiple subagents concurrently via asyncio.gather()
```

**Wiring**: `SubagentSpawner` creates a new `AgentLoop` (Sprint 1) per subagent, optionally filtering the parent `ToolRegistry` (Sprint 2) to a tool subset. Agent definitions resolve via `get_agent_definition()`.

### `prometheus.tools.builtin.agent` — AgentTool
`src/prometheus/tools/builtin/agent.py` — adapted from OpenHarness agent_tool.py (MIT)

```python
class AgentToolInput(BaseModel):
    description: str; prompt: str
    subagent_type: str = "general-purpose"
    model: str | None = None

class AgentTool(BaseTool):
    name = "Agent"
    # Reads SubagentSpawner from context.metadata["subagent_spawner"]
    async def execute(arguments: AgentToolInput, context: ToolExecutionContext) -> ToolResult
```

**Wiring**: Registered in `tools/builtin/__init__.py`. Uses `BaseTool` interface (Sprint 2). Requires `subagent_spawner` key in `ToolExecutionContext.metadata`.

### `prometheus.coordinator.health` — HealthMonitor
`src/prometheus/coordinator/health.py` — novel code

```python
class HealthState(str, Enum): HEALTHY, DEGRADED, CRITICAL

@dataclass
class ComponentHealth:
    name: str; healthy: bool; detail: str; latency_ms: float

@dataclass
class HealthStatus:
    state: HealthState; components: list[ComponentHealth]; timestamp: float
    @property degraded_components -> list[ComponentHealth]
    def summary() -> str

# Individual check functions:
check_llama_cpp(base_url="http://127.0.0.1:8080") -> ComponentHealth
check_sqlite(db_path=None) -> ComponentHealth
check_tailscale() -> ComponentHealth
check_gpu_memory() -> ComponentHealth
check_disk(path="/") -> ComponentHealth

class HealthMonitor:
    def __init__(*, interval=60, llama_url="...", db_path=None, disk_path="/",
                 alert_callback=None, checks=["llama_cpp","sqlite","tailscale","gpu_memory","disk"])

    async def check() -> HealthStatus
    async def run_forever() -> None  # alerts via callback on state transition to degraded
    def stop() -> None
```

**Wiring**: Extends Sprint 6's `Heartbeat` (gateway) with infrastructure checks. `alert_callback` can post to Telegram via the gateway adapter.

### `prometheus.benchmarks.suite` — BenchmarkSuite
`src/prometheus/benchmarks/suite.py` — novel code

```python
class TestTier(IntEnum): TIER_1 = 1, TIER_2 = 2

@dataclass
class TestCase:
    id: str; name: str; tier: int; prompt: str
    expected_tools: list[str]; expected_output_contains: list[str]
    expected_output_not_contains: list[str]; expected_file_exists: list[str]
    expected_file_contains: dict[str, str]; max_turns: int
    setup_commands: list[str]; teardown_commands: list[str]; tags: list[str]

class BenchmarkSuite:
    def filter_tier(tier) -> list[TestCase]
    def filter_tags(tags) -> list[TestCase]
    def get(case_id) -> TestCase | None
    def to_yaml() -> str
    @classmethod from_yaml(yaml_str) -> BenchmarkSuite
    @classmethod from_file(path) -> BenchmarkSuite

load_suite(tier=None) -> BenchmarkSuite
# Built-in: 22 Tier 1 atomic tests + 5 Tier 2 multi-step tests
```

### `prometheus.benchmarks.runner` — BenchmarkRunner
`src/prometheus/benchmarks/runner.py` — novel code

```python
class Score(str, Enum): SUCCESS, PARTIAL, RETRY_SUCCESS, FAIL, CRASH

@dataclass
class ScoreResult:
    case_id: str; case_name: str; score: Score
    turns: int; latency_ms: float; details: str; tool_calls: list[str]

class BenchmarkRunner:
    def __init__(provider, tool_registry, *, model="qwen3.5-32b",
                 max_tokens=4096, cwd=None, adapter=None)

    async def run_case(case: TestCase) -> ScoreResult
    async def run_suite(suite, *, concurrency=1) -> list[ScoreResult]

# CLI: python -m prometheus.benchmarks.runner --model MODEL --tier 1|2
#      --json --verbose --case CASE_ID --concurrency N
```

**Wiring**: `BenchmarkRunner` creates an `AgentLoop` (Sprint 1) per test case with a `ToolRegistry` (Sprint 2). Scoring evaluates tool calls from `ConversationMessage.tool_uses`, output text, and file system assertions.

### Sprint 8 File Tree

```
coordinator/
  __init__.py             — package exports
  agent_definitions.py    — AgentDefinition + builtin agents (adapted from OpenHarness)
  coordinator_mode.py     — TeamRecord, TeamRegistry, coordinator system prompt
  subagent.py             — SubagentSpawner, SubagentResult (novel)
  health.py               — HealthMonitor + 5 check functions (novel)
tools/builtin/
  agent.py                — AgentTool (BaseTool adapter for SubagentSpawner)
benchmarks/
  __init__.py             — package exports
  __main__.py             — CLI entry point
  suite.py                — TestCase, BenchmarkSuite, 27 built-in test cases
  runner.py               — BenchmarkRunner, Score, ScoreResult, CLI
tests/
  test_coordinator.py     — 37 tests (definitions, teams, subagent, health)
  test_benchmarks.py      — 31 tests (suite, scoring, runner, agent tool)
```

---

## Sprint 9 — SENTINEL: Proactive Daemon + AutoDream

Transforms Prometheus from reactive (waits for messages) to proactive (observes, acts, consolidates). Two subsystems: **Activity Observer** watches signals and sends Telegram nudges, **AutoDream Engine** uses idle time for wiki maintenance, memory consolidation, telemetry analysis, and knowledge synthesis.

### Architecture

```
SENTINEL LAYER (signal-driven)
├── SignalBus              — async pub/sub for ActivitySignal events
├── Activity Observer      — pattern detection → Telegram nudges
│   ├── Extraction spike   — high fact count in one pass
│   ├── Error streak       — consecutive tool failures
│   └── Nudge cooldown     — per-type cooldown prevents spam
│
└── AutoDream Engine       — triggered by idle_start signal
    ├── Phase 1: WikiLint             — orphans, broken links, stale pages (no LLM)
    ├── Phase 2: MemoryConsolidation  — dedup, decay, tombstone (no LLM)
    ├── Phase 3: TelemetryDigest      — anomaly detection (no LLM)
    └── Phase 4: KnowledgeSynthesis   — cross-entity insights (LLM, budget-capped)
```

### New Classes

**`sentinel/signals.py`** — Signal bus

```python
@dataclass
class ActivitySignal:
    kind: str               # "idle_start", "idle_end", "extraction_complete", etc.
    timestamp: float
    payload: dict[str, Any]
    source: str

class SignalBus:
    def subscribe(kind: str, callback: SignalCallback) -> None   # "*" for wildcard
    async def emit(signal: ActivitySignal) -> None               # broadcast, exceptions caught
    def recent(kind: str | None, *, limit: int) -> list[ActivitySignal]
```

**`sentinel/observer.py`** — Activity observer

```python
class ActivityObserver:
    def __init__(bus: SignalBus, gateway: BasePlatformAdapter | None, *, config: dict)
    async def start() -> None              # subscribes to "*" on bus
    async def _send_nudge(nudge_type: str, message: str) -> None  # respects cooldown
    # Properties: started, last_activity, pending_nudges, nudge_history
```

**`sentinel/autodream.py`** — AutoDream engine

```python
@dataclass
class DreamResult:
    phase: str
    duration_seconds: float
    summary: dict[str, Any]
    error: str | None

class AutoDreamEngine:
    def __init__(bus: SignalBus, *, wiki_linter, memory_consolidator,
                 telemetry_digest, knowledge_synth, config: dict)
    async def start() -> None              # subscribes to idle_start/idle_end
    async def run_cycle() -> list[DreamResult]  # runs all 4 phases
    # Properties: dreaming, cycle_count, last_cycle_time, last_results
```

**`sentinel/wiki_lint.py`** — Wiki linter

```python
@dataclass
class LintIssue:
    severity: str    # "error" | "warning" | "info"
    category: str    # "orphan" | "broken_link" | "stale" | "duplicate" | "missing_crossref" | "imbalance"
    page: str
    detail: str
    fixable: bool

class WikiLinter:
    def __init__(wiki_root: Path | None)
    def lint() -> LintResult              # scans for 6 issue types
    def auto_fix(result: LintResult) -> int  # fixes safe issues
    def summary(issues: list[LintIssue]) -> str
```

**`sentinel/memory_consolidator.py`** — Memory cleaner

```python
class MemoryConsolidator:
    def __init__(store: MemoryStore, *, decay_rate=0.05, min_confidence=0.1,
                 stale_days=90, similarity_threshold=0.80)
    def consolidate() -> ConsolidationResult  # dedup + decay + tombstone
```

**`sentinel/telemetry_digest.py`** — Telemetry health report

```python
class TelemetryDigest:
    def __init__(telemetry: ToolCallTelemetry, *, period_hours=24, baseline_hours=168)
    def generate() -> DigestResult        # compares current vs baseline
    # Flags: success_rate_drop >5%, retry_increase >10%, latency_spike >50%
```

**`sentinel/knowledge_synth.py`** — LLM-powered insight generation

```python
class KnowledgeSynthesizer:
    def __init__(store: MemoryStore, provider: ModelProvider, *, model, budget_tokens=2000)
    async def synthesize(budget_tokens: int | None) -> list[SynthInsight]
    # Writes to wiki/queries/insight-{date}-{topic}.md
```

### New Tools

**`tools/builtin/sentinel_status.py`** — `sentinel_status` tool

```python
class SentinelStatusTool(BaseTool):
    # Returns: observer state, dream state, signal bus stats, pending nudges
    # Input: SentinelStatusInput(verbose: bool = False)
```

**`tools/builtin/wiki_lint_tool.py`** — `wiki_lint` tool

```python
class WikiLintTool(BaseTool):
    # Triggers wiki lint on demand, returns issue list
    # Input: WikiLintInput(severity: str | None, auto_fix: bool = False)
```

### Modified Files

| File | Change |
|------|--------|
| `gateway/heartbeat.py` | Added `signal_bus` property + setter, idle detection via `_check_idle()`, emits `idle_start`/`idle_end` |
| `memory/extractor.py` | Added `signal_bus` property + setter, emits `extraction_complete` after `run_once()` |
| `memory/store.py` | Added `update_memory(id, **fields)`, `delete_memory(id)`, `get_all_memories(min_confidence, limit)` |
| `telemetry/tracker.py` | Added `since: float | None` param to `report()` for time-windowed queries |
| `tools/builtin/__init__.py` | Exports `SentinelStatusTool`, `WikiLintTool` |
| `scripts/daemon.py` | Creates SignalBus + all SENTINEL components, wires into heartbeat/extractor, registers tools |
| `config/prometheus.yaml` | Added `sentinel:` section with 10 config keys |

### Daemon Wiring (scripts/daemon.py)

SENTINEL is wired after the memory extractor block. Signal bus is passed to heartbeat and extractor via property setters to avoid restructuring existing init order.

```python
signal_bus = SignalBus()
heartbeat.signal_bus = signal_bus       # enables idle detection
extractor.signal_bus = signal_bus       # emits extraction_complete
observer = ActivityObserver(signal_bus, gateway=telegram, config=sentinel_config)
autodream = AutoDreamEngine(signal_bus, wiki_linter=..., ...)
await observer.start()                  # subscribes to "*"
await autodream.start()                 # subscribes to idle_start/idle_end
```

### Config (`prometheus.yaml`)

```yaml
sentinel:
  enabled: true
  idle_threshold_minutes: 15
  nudge_cooldown_minutes: 60
  dream_interval_minutes: 30
  dream_budget_tokens: 2000
  stale_threshold_days: 90
  confidence_decay_rate: 0.05
  digest_lookback_hours: 24
  auto_fix_wiki: true
  synthesis_enabled: true
```

### Sprint 9 File Tree

```
sentinel/
  __init__.py               — package exports (SignalBus, ActivitySignal)
  signals.py                — ActivitySignal dataclass, SignalBus async pub/sub
  observer.py               — ActivityObserver, PendingNudge
  autodream.py              — AutoDreamEngine, DreamResult
  wiki_lint.py              — WikiLinter, LintIssue, LintResult
  memory_consolidator.py    — MemoryConsolidator, ConsolidationResult
  telemetry_digest.py       — TelemetryDigest, DigestAnomaly, DigestResult
  knowledge_synth.py        — KnowledgeSynthesizer, SynthInsight
tools/builtin/
  sentinel_status.py          — SentinelStatusTool + set_sentinel_components()
  wiki_lint_tool.py         — WikiLintTool + set_wiki_linter()
tests/
  test_sentinel.py            — 33 tests (signals, observer, autodream, linter, consolidator, digest, tools)
```

---

## Parallel Tool Dispatch

When the model returns multiple tool calls in one response, the agent loop partitions them by `is_read_only()` and executes read-only tools in parallel via `asyncio.gather`, then mutating tools sequentially. Each tool still goes through the full pipeline individually (pre-hooks → validate → permission check → execute → telemetry → post-hooks).

### Dispatch Flow

```
Model returns N tool calls
    ↓
Partition by tool.is_read_only(parsed_input)
    ↓
Read-only tools → asyncio.gather (parallel)
Mutating tools  → sequential loop (order preserved)
    ↓
Results re-sorted to match original call order
```

Single tool calls skip partitioning entirely.

### Key Functions (`engine/agent_loop.py`)

```python
async def _dispatch_tool_calls(context: LoopContext, tool_calls: list) -> list[ToolResultBlock]
    # Partitions into read-only vs mutating, dispatches accordingly

def _is_tool_read_only(tool: object, tool_input: dict) -> bool
    # Calls tool.is_read_only(parsed_input); handles both method and attribute patterns
```

### Read-Only Tools

Tools returning `True` from `is_read_only()` (safe for parallel execution):
`file_read`, `grep`, `glob`, `lcm_grep`, `lcm_describe`, `lcm_expand`, `lcm_expand_query`, `wiki_query`, `sentinel_status`, `cron_list`, `task_list`, `task_get`, `task_output`

### Error Isolation

`asyncio.gather(return_exceptions=True)` ensures one failed read-only tool does not block others. Failed tools return an error `ToolResultBlock`.

### Tests (`tests/test_parallel_dispatch.py`)

12 tests covering: `_is_tool_read_only` helper, single call passthrough, parallel timing verification, original-order preservation, mixed dispatch (read-only first then mutating), sequential mutating order, error isolation, unknown tools, and pre-hook deny per-tool in parallel.
