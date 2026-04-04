# PROMETHEUS.md — Agent Instructions

This is Prometheus, a sovereign AI agent harness. It runs local LLMs (llama.cpp, Ollama)
via an abstract Model Adapter Layer — no Anthropic API dependency in the agent loop.

## Architecture

```
prometheus/
  engine/       AgentLoop — the main turn loop (Sprint 1)
  adapter/      Model Adapter Layer — provider abstraction (Sprint 3)
  tools/        Tool registry + builtin tools (Sprint 2)
  hooks/        PreToolUse / PostToolUse hooks (Sprint 2)
  permissions/  Security gate (Sprint 4)
  context/      Context management + compression (Sprint 4)
  providers/    Model provider routing (Sprint 3)
  gateway/      Telegram / messaging interface (Sprint 6)
  learning/     Learning loop + skill creation (Sprint 7)
  tasks/        Task persistence (Sprint 5)
  memory/       LCM + persistent memory (Sprint 5)
  skills/       Skill loading from .md files (Sprint 5)
  coordinator/  Multi-agent coordination (Sprint 8)
  telemetry/    Tool call tracking (Sprint 3)
  config/       Settings + path management
```

## Key Conventions

- All extracted donor code has a provenance header: Source, Original path, License, Modified
- Imports use `from prometheus.` not `from openharness.` or `from hermes.`
- Config is loaded from `config/prometheus.yaml` via `prometheus.config`
- Paths resolve through `prometheus.config.paths` (adapted from OpenHarness)
- Python 3.11+, managed with `uv`

## Sprint Status

- [x] Sprint 0: Skeleton
- [ ] Sprint 1: Agent loop (extract OpenHarness engine/)
- [ ] Sprint 2: Tools + hooks (extract OpenHarness tools/ + hooks/)
- [ ] Sprint 3: Model Adapter Layer (novel code)
- [ ] Sprint 4: Security + context management
- [ ] Sprint 5: Skills + memory
- [ ] Sprint 6: Gateway (Telegram)
- [ ] Sprint 7: Learning loop + LCM
- [ ] Sprint 8: Multi-agent + benchmarks

## Running

```bash
uv sync
uv run prometheus
# or
./scripts/start.sh
```
