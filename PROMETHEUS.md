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

## Sprint Status

- [x] Sprint 0: Skeleton
- [x] Sprint 1: Agent loop
- [ ] Sprint 2: Tools + hooks (extract OpenHarness tools/ + hooks/)
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
