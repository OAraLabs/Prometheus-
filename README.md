# Prometheus

**A sovereign AI agent harness for open models.** Run any model locally — Qwen, Gemma, Llama, Mistral, whatever llama.cpp can serve — with Claude Code-quality orchestration, tool calling, persistent memory, and a compounding knowledge base. No API subscriptions. No provider lock-in. Your hardware, your models, your data.

```bash
git clone https://github.com/whieber1/Prometheus-.git
cd Prometheus-
pip install -e .
python3 -m prometheus --setup
```

Four questions. Five minutes. Working agent.

---

## Why Prometheus?

Every existing agent harness assumes Claude or GPT will get tool calls right. Open models don't — they hallucinate tool names, malform JSON, miss required parameters. Prometheus has a **Model Adapter Layer** that sits between the agent loop and the LLM, validating every tool call, auto-repairing common errors, enforcing structured output via GBNF grammar constraints, and retrying with specific error context. This is the layer that makes open models actually work in an agent loop.

Built from the best organs of five proven projects — assembled as a new codebase, not a fork:

| Donor | What Prometheus Takes |
|-------|----------------------|
| **OpenHarness** | Agent loop, tool registry, hooks, permissions — clean-room Python reimplementation of the Claude Code harness architecture |
| **Hermes** (NousResearch) | Telegram gateway, cron scheduler, skill creation, memory patterns |
| **OpenClaw** | Battle-tested integration patterns — heartbeat, Archive Bridge, Memory Extractor (running in production for months) |
| **Lossless-Claw** | 11,600 lines of TypeScript ported to Python — DAG-based lossless context management |
| **Karpathy's LLM Wiki** | Compounding knowledge base concept, adapted into the memory pipeline |

Plus two novel systems that don't exist anywhere else: the Model Adapter Layer and SENTINEL (a proactive daemon layer inspired by Claude Code's unreleased KAIROS feature, built from scratch).

No leaked source code is in this project. Everything is MIT-licensed donor code, clean-room reimplementation, or novel code written from scratch.

---

## Features

### Model Independence
- Runs **any model** llama.cpp or Ollama can serve — Qwen, Gemma, Llama, Mistral, Phi, DeepSeek, Command-R
- Optimized formatters for Qwen and Gemma, default formatter works with everything else
- Auto-detects whatever model is loaded — swap the GGUF, restart, done
- 4 providers: llama.cpp (primary), Ollama (fallback), Anthropic API (fallback), stub (testing)

### Model Adapter Layer (The Novel Contribution)
- **Tool Call Validator** — catches malformed tool calls before execution, auto-repairs common errors (fuzzy name matching, JSON extraction from markdown, type coercion)
- **GBNF Grammar Enforcer** — generates per-tool grammars for llama.cpp constrained decoding. Forces valid JSON at the token level
- **Multi-Strategy Output Extraction** — 4 cascading strategies for handling messy model output (JSON fenced blocks, generic fenced, JSON-on-line, greedy) + truncated JSON repair
- **Retry Engine** — specific error feedback with schema, not generic "try again"
- **Telemetry** — tracks success rate per model per tool. Know exactly where your model struggles
- Configurable strictness: NONE (Claude), MEDIUM (Qwen), STRICT (weaker models)

### 25 Builtin Tools
bash, file_read, file_write, file_edit, grep, glob, cron_create, cron_delete, cron_list, task_create, task_get, task_list, task_update, task_stop, task_output, todo_write, skill, agent (subagent spawning), lcm_grep, lcm_expand, lcm_describe, lcm_expand_query, wiki_compile, wiki_query, sentinel_status, wiki_lint

### Lossless Context Management (LCM)
Ported from Lossless-Claw (11,600 lines TypeScript → Python). DAG-based compression that never loses information:
- Every message persisted to SQLite — nothing is ever deleted
- Old messages summarized into DAG nodes when context fills up
- Summaries link back to originals — agent can `lcm_expand` any summary to recover detail
- FTS5 full-text search over compressed history via `lcm_grep`
- Works within 32K context windows without losing the plot

### Wiki Knowledge System
Compounding knowledge base inspired by Karpathy's LLM Wiki concept:
- Memory Extractor runs every 30 minutes, extracts structured facts from conversations
- WikiCompiler builds entity pages with `[[wiki-links]]` cross-references
- WikiQueryTool searches the wiki — query results file back as new pages
- The wiki gets smarter every time you use it
- Obsidian-compatible markdown (point it at a vault and the graph view lights up)

### SENTINEL — Proactive Background Intelligence
Transforms Prometheus from reactive to proactive:
- **Activity Observer** — watches telemetry and memory patterns, sends nudges via Telegram when something's interesting
- **AutoDream Engine** — runs during idle time:
  - Wiki Lint: finds orphan pages, broken links, missing cross-references
  - Memory Consolidation: deduplicates facts, decays stale confidence
  - Telemetry Digest: flags tool performance anomalies
  - Knowledge Synthesis: discovers cross-entity patterns (LLM-powered, budget-capped)
- Never auto-executes above AUTONOMOUS trust level — nudges, doesn't act

### Security
- 4-level trust model (BLOCKED → APPROVE → AUTO → AUTONOMOUS)
- 33+ always-blocked regex patterns
- Workspace boundary enforcement
- Bash command intent analysis
- Memory security scanning (anti prompt injection)
- PROMETHEUS.md discovery with upward directory traversal

### Parallel Tool Execution
- Read-only tools run concurrently via `asyncio.gather`
- Mutating tools run sequentially — no race conditions
- Security hooks still run on every tool, even in parallel
- Saves 1-2 seconds per skipped LLM round-trip

### Always-On
- Telegram gateway (Slack coming soon)
- Cron scheduler for recurring tasks
- Heartbeat monitoring
- Slash commands: `/start`, `/status`, `/help`, `/reset`, `/model`, `/wiki`, `/sentinel`
- systemd service for background operation

### Learning Loop
- Periodic nudges — agent self-evaluates at intervals
- Autonomous skill creation from successful task completions
- Skill refinement — skills improve on reuse
- Bounded memory management (MEMORY.md + USER.md)
- System prompt dynamic boundary (cache-aware static/dynamic split)

---

## Quick Start

### Prerequisites
- Python 3.11+
- llama.cpp or Ollama running with any model loaded
- A Telegram bot token (from [@BotFather](https://t.me/BotFather)) — optional, CLI works without it

### Install

```bash
git clone https://github.com/whieber1/Prometheus-.git
cd Prometheus-
pip install -e .
python3 -m prometheus --setup
```

The setup wizard walks you through:
1. Where is your LLM? (llama.cpp URL or Ollama URL)
2. Which gateway? (Telegram / CLI only)
3. Bot token (if Telegram)
4. Creates directories, writes config, runs smoke test

### Run

```bash
# Interactive CLI
python3 -m prometheus

# One-shot query
python3 -m prometheus --once "List all Python files in this directory"

# Daemon mode (Telegram + cron + heartbeat + SENTINEL)
python3 -m prometheus daemon

# As a systemd service (Linux)
systemctl --user enable --now prometheus
```

### Multi-Machine Setup

Prometheus is designed for split-brain deployments. Run the agent on a storage machine (SQLite, wiki, memory), point it at a GPU machine for inference:

```yaml
# config/prometheus.yaml
model:
  provider: "llama_cpp"
  base_url: "http://gpu-machine:8080"      # llama.cpp on GPU
  fallback_url: "http://gpu-machine:11434"  # Ollama fallback
```

Connect machines via Tailscale, WireGuard, or any network. Prometheus talks to llama.cpp over HTTP — localhost or remote, it doesn't care.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INTERFACE LAYER                        │
│  Telegram │ Slack (soon) │ CLI │ Voice (planned)         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                  ALWAYS-ON LAYER                          │
│  Heartbeat │ Cron │ SENTINEL │ Memory Extractor          │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                 ORCHESTRATION LAYER                       │
│  Agent Loop → Model Adapter → Tool Dispatch              │
│  ┌─────────────────────────────────────────────────┐     │
│  │  MODEL ADAPTER LAYER                             │     │
│  │  Validator │ Formatter │ Enforcer │ Retry │ Telem│     │
│  └─────────────────────────────────────────────────┘     │
│  Tools │ Hooks │ Permissions │ Skills │ Planning         │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                 KNOWLEDGE LAYER                           │
│  LCM (DAG compression) │ Wiki │ MEMORY.md │ USER.md      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────┐
│                 MODEL PROVIDER LAYER                      │
│  llama.cpp │ Ollama │ Anthropic API │ Any OpenAI-compat   │
└─────────────────────────────────────────────────────────┘
```

---

## Configuration

All config lives in `config/prometheus.yaml`. Key sections:

```yaml
model:
  provider: "llama_cpp"
  base_url: "http://localhost:8080"
  # model name auto-detected from llama.cpp on startup

context:
  effective_limit: 24000
  compression_trigger: 0.75
  tool_result_max: 4000

security:
  permission_mode: "default"
  workspace_root: "~/.prometheus/workspace"

gateway:
  telegram_enabled: true
  telegram_token: ""           # from @BotFather

sentinel:
  enabled: true
  idle_threshold_minutes: 15
  dream_interval_minutes: 30
  dream_budget_tokens: 2000
```

Full config reference: [config/prometheus.yaml](config/prometheus.yaml)

---

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Welcome message |
| `/status` | Model, uptime, tools, memory stats, SENTINEL state |
| `/help` | List commands and capabilities |
| `/reset` | Clear conversation context |
| `/model` | Current model name and provider |
| `/wiki` | Wiki stats — page count, recent updates |
| `/sentinel` | SENTINEL status — active/dreaming/idle, last dream results |

---

## Project Structure

```
prometheus/
├── src/prometheus/
│   ├── engine/          # Agent loop
│   ├── adapter/         # Model Adapter Layer (validator, formatter, enforcer, retry)
│   ├── providers/       # llama_cpp, ollama, anthropic, stub
│   ├── tools/builtin/   # 25 builtin tools
│   ├── hooks/           # PreToolUse / PostToolUse pipeline
│   ├── permissions/     # 4-level security gate
│   ├── memory/          # LCM, wiki compiler, extractor, store
│   ├── context/         # Token budget, compression, dynamic tools, prompt assembly
│   ├── skills/          # Skill loader + registry
│   ├── tasks/           # Background task manager
│   ├── gateway/         # Telegram, cron, heartbeat, archive
│   ├── sentinel/        # Observer, AutoDream, wiki lint, memory consolidation
│   ├── learning/        # Nudge, skill creator, skill refiner
│   ├── coordinator/     # Subagent spawning
│   ├── telemetry/       # Tool call tracking (SQLite)
│   └── config/          # Settings management
├── scripts/
│   ├── daemon.py        # Main entry point for always-on mode
│   └── health_check.sh  # Process health check
├── tests/               # 433+ tests
├── config/
│   └── prometheus.yaml  # Default configuration
├── benchmarks/          # Tier 1 + Tier 2 benchmark suite
├── skills/              # User skill files (.md)
└── PROMETHEUS.md        # Agent instructions (like CLAUDE.md)
```

---

## Provenance & Legal

Every file extracted from a donor project includes a header comment with source project, original path, license, and what was modified. All donors are MIT-licensed. The Model Adapter Layer, SENTINEL, and Wiki Knowledge System are novel code.

No Claude Code source is in this project. All Claude Code-derived architecture comes from clean-room reimplementations (OpenHarness, analysis-2-harness-engineering).

---

## Benchmarks

Sprint 8 includes a benchmark suite:
- **Tier 1**: 21 atomic tool call tests (single tool operations)
- **Tier 2**: 5 multi-step tests (plan → implement → verify)
- Scoring: SUCCESS, PARTIAL, RETRY_SUCCESS, FAIL, CRASH

```bash
python -m prometheus.benchmarks.runner --model gemma4-26b --tier 1
```

---

## Roadmap

- [x] Core agent loop with tool dispatch
- [x] Model Adapter Layer (validation, repair, GBNF, retry, telemetry)
- [x] 25 builtin tools
- [x] LCM context management (ported from TypeScript)
- [x] Security gate (4-level trust model)
- [x] Telegram gateway + always-on daemon
- [x] Wiki knowledge system (Karpathy-inspired)
- [x] SENTINEL proactive layer
- [x] Parallel tool execution
- [x] Setup wizard
- [x] Auto-detect loaded model from llama.cpp
- [ ] Slack gateway
- [ ] MCP integration (tool server discovery)
- [ ] Model router (task-based provider selection)
- [ ] Divergence detection + checkpoint/rollback
- [ ] DeepEval integration for automated testing
- [ ] Web UI for setup and monitoring
- [ ] Fine-tuning flywheel (LoRA on collected traces)

---

## Stats

- **~15,000 lines** of production Python
- **433+ tests** passing
- **25 builtin tools**
- **4 model providers**
- **8 LCM modules** (ported from 11.6K TypeScript)
- **Built in ~4 hours** of Claude Code sessions (estimated 20-28 hours)

---

## License

MIT

---

## Credits

Built by [Will Hieber](https://github.com/whieber1) / OAra AI Lab.

Architecture informed by: [OpenHarness](https://github.com/HKUDS/OpenHarness) (HKUDS), [Hermes Agent](https://github.com/NousResearch/hermes-agent) (NousResearch), [Lossless-Claw](https://github.com/Martian-Engineering/lossless-claw) (Martian Engineering), and [Andrej Karpathy's LLM Wiki concept](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

---

*"The model is the agent. The code is the harness. Build great harnesses."*
