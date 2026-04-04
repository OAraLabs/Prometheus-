# Prometheus

Sovereign AI agent harness for local LLMs. Runs Qwen, Gemma, and other models
via llama.cpp or Ollama — no cloud API dependency.

## Philosophy

Extract, don't reinvent. Prometheus is built by adapting proven patterns from
OpenHarness, Hermes, and OpenClaw, wired together with a novel Model Adapter
Layer that abstracts away the provider.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency management
- llama.cpp or Ollama running locally (for Sprint 1+)

## Quick Start

```bash
uv sync
uv run prometheus
```

## Configuration

Edit `config/prometheus.yaml`. Key settings:

- `model.base_url` — llama.cpp server URL (default: `http://localhost:8080`)
- `model.fallback_url` — Ollama URL (default: `http://localhost:11434`)
- `context.effective_limit` — token budget for the active context window
- `security.permission_mode` — `default`, `strict`, or `permissive`

## Project Structure

See `PROMETHEUS.md` for the full architecture and sprint status.

## License

MIT
