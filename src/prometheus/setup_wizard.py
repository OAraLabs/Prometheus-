"""First-run setup wizard for Prometheus.

Walks the user through provider selection, gateway configuration,
directory creation, config writing, and a smoke test.

Usage:
    python3 -m prometheus --setup
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import yaml

from prometheus.config.paths import get_config_dir

# Config file lives in the repo's config/ directory
_REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "prometheus.yaml"


def _input(prompt: str, default: str = "") -> str:
    """Read a line from stdin with an optional default."""
    suffix = f" [{default}]" if default else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nSetup cancelled.")
        sys.exit(1)
    return value or default


def _ask_choice(prompt: str, options: list[str], default: int = 1) -> int:
    """Present numbered options and return the 1-based choice."""
    print(f"\n{prompt}\n")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    raw = _input(f"\nChoice", str(default))
    try:
        choice = int(raw)
        if 1 <= choice <= len(options):
            return choice
    except ValueError:
        pass
    print(f"Invalid choice, using default ({default}).")
    return default


class SetupWizard:
    """Interactive first-run setup. Runs once, creates config, validates."""

    def __init__(self, gateway_only: bool = False) -> None:
        self._gateway_only = gateway_only
        self._provider: str = "llama_cpp"
        self._base_url: str = "http://localhost:8080"
        self._model_name: str = ""
        self._gateway: str = "cli"
        self._telegram_token: str = ""
        self._telegram_chat_ids: list[int] = []
        self._telegram_bot_name: str = ""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Run the full wizard. Returns True if setup succeeded."""
        existing = self._load_existing_config()

        if existing and not self._gateway_only:
            action = self._ask_rerun()
            if action == 4:
                print("Cancelled.")
                return False
            if action == 2:
                self._gateway_only = False
                self._prefill_from(existing)
                self._step_provider()
                self._merge_and_write(existing)
                return self._run_smoke_test()
            if action == 3:
                self._gateway_only = True
                self._prefill_from(existing)

        self._print_banner()

        if not self._gateway_only:
            if not self._check_prerequisites():
                return False
            self._step_provider()

        self._step_gateway()
        self._create_directories()

        if self._gateway_only and existing:
            self._merge_and_write(existing)
        else:
            self._write_config()

        passed = True
        if not self._gateway_only:
            passed = self._run_smoke_test()

        self._print_summary(passed)
        return passed

    # ------------------------------------------------------------------
    # Banner + prerequisites
    # ------------------------------------------------------------------

    def _print_banner(self) -> None:
        GREEN = "\033[32m"
        RESET = "\033[0m"
        print(f"""
{GREEN}    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║   🔥 P R O M E T H E U S                            ║
    ║                                                      ║
    ║      Sovereign AI Agent Harness                      ║
    ║      95 skills • 37 tools • Infinite potential       ║
    ║                                                      ║
    ║   "The model is the agent. The code is the harness." ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝{RESET}
""")

    def _check_prerequisites(self) -> bool:
        print("Checking prerequisites...")

        # Python version
        vi = sys.version_info
        if vi < (3, 11):
            print(f"  x Python {vi.major}.{vi.minor}.{vi.micro} — need 3.11+")
            print("\nPlease upgrade Python and try again.")
            return False
        print(f"  + Python {vi.major}.{vi.minor}.{vi.micro}")

        # Key dependencies
        missing: list[str] = []
        for mod in ("yaml", "httpx"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)

        if missing:
            print(f"  x Missing packages: {', '.join(missing)}")
            print('\n  Run: pip install -e ".[dev]"')
            return False
        print("  + All core dependencies installed")

        # Config dir
        config_dir = get_config_dir()
        print(f"  + Config directory: {config_dir}")
        print()
        return True

    # ------------------------------------------------------------------
    # Step 2: Model provider
    # ------------------------------------------------------------------

    def _step_provider(self) -> None:
        choice = _ask_choice(
            "Where is your LLM running?",
            [
                "llama.cpp (local or remote)",
                "Ollama (local or remote)",
                "I don't have one running yet",
            ],
            default=1,
        )

        if choice == 3:
            self._print_no_model_help()
            sys.exit(0)

        self._provider = "llama_cpp" if choice == 1 else "ollama"
        default_url = (
            "http://localhost:8080" if choice == 1 else "http://localhost:11434"
        )

        while True:
            url = _input(f"\nEnter the URL", default_url)
            url = url.rstrip("/")
            print(f"\n  Testing connection to {url}...")

            model = self._test_provider(url)
            if model is not None:
                print(f"  + Connected")
                if model:
                    print(f"  Detected model: {model}")
                    self._model_name = model
                self._base_url = url
                return

            print(f"  x Could not connect to {url}")
            print("    - Is the server running?")
            print("    - Is the URL correct?")
            print("    - If remote, use the full URL (e.g., http://192.168.1.100:8080)")

            retry = _input("\nTry a different URL? [y/N]", "N")
            if retry.lower() != "y":
                print("\nSaving config with this URL — you can fix it later.")
                self._base_url = url
                return

    def _test_provider(self, url: str) -> str | None:
        """Test connection to provider. Returns model name or None on failure."""
        # Try /v1/models (works for llama.cpp and Ollama OpenAI-compat)
        try:
            resp = httpx.get(f"{url}/v1/models", timeout=10.0)
            resp.raise_for_status()
            models = resp.json().get("data", [])
            if models:
                return models[0].get("id", "")
            return ""
        except Exception:
            pass

        # Try Ollama /api/tags
        if self._provider == "ollama":
            try:
                resp = httpx.get(f"{url}/api/tags", timeout=10.0)
                resp.raise_for_status()
                models = resp.json().get("models", [])
                if models:
                    return models[0].get("name", "")
                return ""
            except Exception:
                pass

        return None

    def _print_no_model_help(self) -> None:
        print(
            """
No problem. Here's how to get started:

  llama.cpp:
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp && make LLAMA_CUDA=1 -j$(nproc)
    ./llama-server -m models/your-model.gguf -c 32768 -ngl 99 --port 8080

  Ollama:
    curl -fsSL https://ollama.com/install.sh | sh
    ollama run qwen3.5:32b

Run this wizard again after your model is running:
  python3 -m prometheus --setup
"""
        )

    # ------------------------------------------------------------------
    # Step 3: Gateway
    # ------------------------------------------------------------------

    def _step_gateway(self) -> None:
        choice = _ask_choice(
            "How do you want to talk to Prometheus?",
            [
                "Telegram",
                "Slack (coming soon)",
                "CLI only (no messaging gateway)",
            ],
            default=3,
        )

        if choice == 2:
            print(
                "\nSlack support is coming soon. Choose Telegram or CLI for now."
            )
            return self._step_gateway()

        if choice == 3:
            self._gateway = "cli"
            print(
                "\n  Got it — CLI mode. Add Telegram later with:"
                "\n    python3 -m prometheus --setup --gateway-only"
            )
            return

        # Telegram flow
        self._gateway = "telegram"
        token = _input("\nEnter your Telegram bot token (from @BotFather)")
        if not token:
            print("  No token provided. Falling back to CLI mode.")
            self._gateway = "cli"
            return

        print("  Testing token...")
        bot_name = self._test_telegram_token(token)
        if bot_name:
            print(f"  + Bot connected: @{bot_name}")
            self._telegram_token = token
            self._telegram_bot_name = bot_name
        else:
            print("  x Invalid token or could not reach Telegram API.")
            keep = _input("Save this token anyway? [y/N]", "N")
            if keep.lower() == "y":
                self._telegram_token = token
            else:
                print("  Falling back to CLI mode.")
                self._gateway = "cli"
                return

        # Optional: restrict to chat ID
        print(
            "\n  Optional: Restrict to your chat ID only?"
            "\n  Send /start to your bot, then enter your chat ID here."
            "\n  Leave blank to allow all users."
        )
        chat_id_str = _input("Chat ID", "")
        if chat_id_str:
            try:
                self._telegram_chat_ids = [int(chat_id_str)]
            except ValueError:
                print("  Invalid chat ID, skipping restriction.")

    def _test_telegram_token(self, token: str) -> str | None:
        """Test a Telegram bot token via getMe. Returns bot username or None."""
        try:
            resp = httpx.get(
                f"https://api.telegram.org/bot{token}/getMe", timeout=10.0
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                return data["result"].get("username", "")
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Step 4: Create directories + write config
    # ------------------------------------------------------------------

    def _create_directories(self) -> None:
        print("\nCreating Prometheus data directories...")
        config_dir = get_config_dir()
        dirs = [
            config_dir / "workspace",
            config_dir / "wiki",
            config_dir / "sentinel",
            config_dir / "skills",
            config_dir / "data",
            config_dir / "data" / "sessions",
            config_dir / "logs",
            config_dir / "cache",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            print(f"  + {d}")

    def _write_config(self) -> None:
        """Write a fresh config, preserving defaults from the template."""
        print(f"\nWriting config to {_REPO_CONFIG}...")

        # Load template defaults
        base = self._load_existing_config() or {}

        # Update wizard-touched fields
        self._apply_wizard_fields(base)

        self._save_config(base)

    def _merge_and_write(self, existing: dict[str, Any]) -> None:
        """Merge wizard fields into existing config."""
        print(f"\nUpdating config at {_REPO_CONFIG}...")
        self._apply_wizard_fields(existing)
        self._save_config(existing)

    def _apply_wizard_fields(self, cfg: dict[str, Any]) -> None:
        """Apply wizard answers to the config dict."""
        if not self._gateway_only:
            model = cfg.setdefault("model", {})
            model["provider"] = self._provider
            if self._provider == "ollama":
                model["base_url"] = self._base_url
                model["fallback_url"] = self._base_url
            else:
                model["base_url"] = self._base_url
            if self._model_name:
                model["model"] = self._model_name

        gateway = cfg.setdefault("gateway", {})
        if self._gateway == "telegram":
            gateway["telegram_enabled"] = True
            gateway["telegram_token"] = self._telegram_token
            gateway["allowed_chat_ids"] = self._telegram_chat_ids
        elif self._gateway == "cli":
            gateway["telegram_enabled"] = False

    def _save_config(self, cfg: dict[str, Any]) -> None:
        """Write config dict to YAML file and update .gitignore."""
        _REPO_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        with _REPO_CONFIG.open("w", encoding="utf-8") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

        # Print summary of what was written
        model = cfg.get("model", {})
        gw = cfg.get("gateway", {})
        print(f"  + Model provider: {model.get('provider', '?')} @ {model.get('base_url', '?')}")
        if model.get("model"):
            print(f"  + Detected model: {model['model']}")
        if gw.get("telegram_enabled"):
            token = gw.get("telegram_token", "")
            masked = f"{token[:4]}...{token[-4:]}" if len(token) > 8 else "****"
            print(f"  + Gateway: telegram (token: {masked})")
        else:
            print("  + Gateway: CLI only")
        print("  + Config saved")

        # Ensure prometheus.yaml is in .gitignore (it may contain secrets)
        self._ensure_gitignore()

    def _ensure_gitignore(self) -> None:
        """Add config/prometheus.yaml to .gitignore if not already there."""
        gitignore = _REPO_CONFIG.parents[1] / ".gitignore"
        entry = "config/prometheus.yaml"
        if gitignore.exists():
            content = gitignore.read_text(encoding="utf-8")
            if entry in content:
                return
            if not content.endswith("\n"):
                content += "\n"
            content += f"\n# Prometheus config (contains bot tokens)\n{entry}\n"
            gitignore.write_text(content, encoding="utf-8")
        else:
            gitignore.write_text(f"# Prometheus config (contains bot tokens)\n{entry}\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Step 5: Smoke test
    # ------------------------------------------------------------------

    def _run_smoke_test(self) -> bool:
        """Send a simple prompt to the model and verify a response."""
        print("\nRunning smoke test...")
        print("  Testing LLM connection...")

        try:
            import time

            t0 = time.monotonic()
            url = f"{self._base_url.rstrip('/')}/v1/chat/completions"
            payload = {
                "model": self._model_name or "local",
                "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                "max_tokens": 32,
                "stream": False,
            }
            resp = httpx.post(url, json=payload, timeout=60.0)
            resp.raise_for_status()
            elapsed = time.monotonic() - t0

            data = resp.json()
            choices = data.get("choices", [])
            text = ""
            if choices:
                text = (choices[0].get("message", {}).get("content", "") or "").strip()

            print(f'  Sending test prompt: "What is 2+2?"')
            print(f"  Response: {text!r}")

            if "4" in text:
                print(f"  + Smoke test passed ({elapsed:.1f}s)")
                return True
            else:
                print(f"  ~ Smoke test got a response but '4' not found ({elapsed:.1f}s)")
                print("    The model is reachable — response quality may vary.")
                return True  # Connection works, that's what matters

        except Exception as exc:
            print(f"  x Smoke test failed: {exc}")
            print()
            print("  Config was saved. Fix the issue and test manually:")
            print('    python3 -m prometheus --once "What is 2+2?"')
            return False

    # ------------------------------------------------------------------
    # Step 6: Summary
    # ------------------------------------------------------------------

    def _print_summary(self, passed: bool) -> None:
        model_display = self._model_name or "(auto-detect)"
        provider_display = self._provider

        print()
        print("=" * 55)
        if passed:
            print("   + Prometheus is ready!")
        else:
            print("   ~ Prometheus config saved (smoke test failed)")
        print("=" * 55)
        print()
        print(f"  Model:    {model_display} via {provider_display}")
        if self._gateway == "telegram" and self._telegram_bot_name:
            print(f"  Gateway:  Telegram (@{self._telegram_bot_name})")
        elif self._gateway == "telegram":
            print("  Gateway:  Telegram")
        else:
            print("  Gateway:  CLI only")
        print(f"  Config:   {_REPO_CONFIG}")
        print(f"  Data:     {get_config_dir()}")
        print()
        print("Start Prometheus:")
        print()
        print("  Interactive:  python3 -m prometheus")
        print('  One-shot:     python3 -m prometheus --once "your question"')
        print(f"  Daemon:       python3 scripts/daemon.py --config {_REPO_CONFIG}")
        if self._gateway == "telegram" and self._telegram_bot_name:
            print()
            print(f"Send /start to @{self._telegram_bot_name} to begin.")
        print()
        print("=" * 55)

    # ------------------------------------------------------------------
    # Re-run handling
    # ------------------------------------------------------------------

    def _ask_rerun(self) -> int:
        """Ask what to do when config already exists. Returns 1-4."""
        return _ask_choice(
            f"Existing configuration found at {_REPO_CONFIG}",
            [
                "Start fresh (overwrite everything)",
                "Update model provider only",
                "Update gateway only",
                "Cancel",
            ],
            default=4,
        )

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _load_existing_config(self) -> dict[str, Any] | None:
        """Load existing config if it exists."""
        if _REPO_CONFIG.exists():
            try:
                with _REPO_CONFIG.open(encoding="utf-8") as fh:
                    return yaml.safe_load(fh) or {}
            except Exception:
                return None
        return None

    def _prefill_from(self, cfg: dict[str, Any]) -> None:
        """Pre-fill wizard state from existing config."""
        model = cfg.get("model", {})
        self._provider = model.get("provider", "llama_cpp")
        self._base_url = model.get("base_url", "http://localhost:8080")
        self._model_name = model.get("model", "")

        gw = cfg.get("gateway", {})
        if gw.get("telegram_enabled"):
            self._gateway = "telegram"
            self._telegram_token = gw.get("telegram_token", "")
            self._telegram_chat_ids = gw.get("allowed_chat_ids", [])
        else:
            self._gateway = "cli"
