"""Tests for Sprint 6: Telegram gateway, cron system, heartbeat, archive."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from prometheus.gateway.archive_writer import ArchiveWriter
from prometheus.gateway.config import Platform, PlatformConfig
from prometheus.gateway.cron_scheduler import (
    _jobs_due,
    append_history,
    execute_job,
    get_history_path,
    load_history,
    run_scheduler_loop,
)
from prometheus.gateway.cron_service import (
    delete_cron_job,
    get_cron_job,
    load_cron_jobs,
    mark_job_run,
    next_run_time,
    save_cron_jobs,
    set_job_enabled,
    upsert_cron_job,
    validate_cron_expression,
)
from prometheus.gateway.heartbeat import Heartbeat
from prometheus.gateway.platform_base import (
    MessageEvent,
    MessageType,
    SendResult,
)
from prometheus.gateway.telegram import (
    TelegramAdapter,
    chunk_message,
    escape_markdown_v2,
)
from prometheus.tools.base import ToolExecutionContext, ToolRegistry
from prometheus.tools.builtin.cron_create import CronCreateTool, CronCreateToolInput
from prometheus.tools.builtin.cron_delete import CronDeleteTool, CronDeleteToolInput
from prometheus.tools.builtin.cron_list import CronListTool, CronListToolInput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Override Prometheus data dir to a temp directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(
        "prometheus.gateway.cron_service.get_cron_registry_path",
        lambda: data_dir / "cron_jobs.json",
    )
    monkeypatch.setattr(
        "prometheus.gateway.cron_scheduler.get_data_dir",
        lambda: data_dir,
    )
    monkeypatch.setattr(
        "prometheus.gateway.cron_scheduler.get_logs_dir",
        lambda: tmp_path / "logs",
    )
    return data_dir


@pytest.fixture
def ctx(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(cwd=tmp_path)


# ---------------------------------------------------------------------------
# Platform config
# ---------------------------------------------------------------------------


class TestPlatformConfig:
    def test_default_config(self):
        config = PlatformConfig(platform=Platform.TELEGRAM, token="test-token")
        assert config.platform == Platform.TELEGRAM
        assert config.token == "test-token"
        assert config.max_message_length == 4096
        assert not config.is_restricted

    def test_chat_allowed_unrestricted(self):
        config = PlatformConfig(platform=Platform.TELEGRAM)
        assert config.chat_allowed(12345)

    def test_chat_allowed_restricted(self):
        config = PlatformConfig(
            platform=Platform.TELEGRAM, allowed_chat_ids=[100, 200]
        )
        assert config.is_restricted
        assert config.chat_allowed(100)
        assert not config.chat_allowed(999)


# ---------------------------------------------------------------------------
# Message event
# ---------------------------------------------------------------------------


class TestMessageEvent:
    def test_session_key(self):
        event = MessageEvent(
            chat_id=123,
            user_id=456,
            text="hello",
            message_id=1,
            platform=Platform.TELEGRAM,
        )
        assert event.session_key() == "telegram:123"

    def test_defaults(self):
        event = MessageEvent(
            chat_id=1,
            user_id=2,
            text="test",
            message_id=3,
            platform=Platform.CLI,
        )
        assert event.message_type == MessageType.TEXT
        assert event.username is None
        assert event.timestamp is not None


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------


class TestMarkdownV2:
    def test_escape_special_chars(self):
        result = escape_markdown_v2("Hello *world* [link](url)")
        assert "\\*" in result
        assert "\\[" in result
        assert "\\(" in result

    def test_no_escape_plain(self):
        result = escape_markdown_v2("Hello world")
        assert result == "Hello world"


class TestChunkMessage:
    def test_short_message(self):
        chunks = chunk_message("hello", max_length=100)
        assert chunks == ["hello"]

    def test_long_message_split_at_newline(self):
        text = "line1\nline2\nline3"
        chunks = chunk_message(text, max_length=10)
        assert len(chunks) > 1
        # All text should be preserved
        rejoined = "".join(chunks)
        assert "line1" in rejoined
        assert "line3" in rejoined

    def test_hard_truncate(self):
        text = "a" * 20
        chunks = chunk_message(text, max_length=8)
        assert all(len(c) <= 8 for c in chunks)
        assert "".join(chunks) == text


# ---------------------------------------------------------------------------
# Telegram adapter
# ---------------------------------------------------------------------------


class TestTelegramAdapter:
    def test_init_requires_token(self):
        config = PlatformConfig(platform=Platform.TELEGRAM, token="")
        agent_loop = MagicMock()
        registry = ToolRegistry()
        adapter = TelegramAdapter(
            config=config,
            agent_loop=agent_loop,
            tool_registry=registry,
        )
        assert not adapter.running

    @pytest.mark.asyncio
    async def test_start_raises_without_token(self):
        config = PlatformConfig(platform=Platform.TELEGRAM, token="")
        adapter = TelegramAdapter(
            config=config,
            agent_loop=MagicMock(),
            tool_registry=ToolRegistry(),
        )
        with pytest.raises(ValueError, match="token is required"):
            await adapter.start()

    @pytest.mark.asyncio
    async def test_send_without_app(self):
        config = PlatformConfig(platform=Platform.TELEGRAM, token="test")
        adapter = TelegramAdapter(
            config=config,
            agent_loop=MagicMock(),
            tool_registry=ToolRegistry(),
        )
        result = await adapter.send(123, "test")
        assert not result.success
        assert result.error == "Bot not initialized"

    @pytest.mark.asyncio
    async def test_on_message_unauthorized(self):
        config = PlatformConfig(
            platform=Platform.TELEGRAM,
            token="test",
            allowed_chat_ids=[100],
        )
        agent_loop = AsyncMock()
        adapter = TelegramAdapter(
            config=config,
            agent_loop=agent_loop,
            tool_registry=ToolRegistry(),
        )
        event = MessageEvent(
            chat_id=999,
            user_id=1,
            text="hello",
            message_id=1,
            platform=Platform.TELEGRAM,
        )
        await adapter.on_message(event)
        # Agent should NOT have been called
        agent_loop.run_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_to_agent(self):
        """Test that on_message dispatches to agent_loop.run_async."""
        config = PlatformConfig(platform=Platform.TELEGRAM, token="test")

        # Mock agent loop
        mock_result = MagicMock()
        mock_result.text = "Agent response"
        agent_loop = AsyncMock()
        agent_loop.run_async = AsyncMock(return_value=mock_result)

        # Mock send
        adapter = TelegramAdapter(
            config=config,
            agent_loop=agent_loop,
            tool_registry=ToolRegistry(),
        )
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id=1))

        event = MessageEvent(
            chat_id=123,
            user_id=456,
            text="test message",
            message_id=1,
            platform=Platform.TELEGRAM,
        )
        await adapter.on_message(event)

        agent_loop.run_async.assert_called_once()
        call_kwargs = agent_loop.run_async.call_args
        assert call_kwargs.kwargs.get("user_message") or call_kwargs[1].get("user_message") or "test message" in str(call_kwargs)

        adapter.send.assert_called_once_with(
            123, "Agent response", reply_to=1
        )


# ---------------------------------------------------------------------------
# Cron service
# ---------------------------------------------------------------------------


class TestCronService:
    def test_validate_cron_expression(self):
        assert validate_cron_expression("*/5 * * * *")
        assert validate_cron_expression("0 9 * * 1-5")
        assert not validate_cron_expression("invalid")
        assert not validate_cron_expression("")

    def test_next_run_time(self):
        base = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
        next_time = next_run_time("0 * * * *", base)
        assert next_time > base
        assert next_time.minute == 0

    def test_upsert_and_load(self, tmp_data_dir):
        upsert_cron_job(
            {
                "name": "test-job",
                "schedule": "*/5 * * * *",
                "command": "echo hello",
            }
        )
        jobs = load_cron_jobs()
        assert len(jobs) == 1
        assert jobs[0]["name"] == "test-job"
        assert jobs[0]["enabled"] is True
        assert "next_run" in jobs[0]

    def test_upsert_replaces(self, tmp_data_dir):
        upsert_cron_job({"name": "j1", "schedule": "0 * * * *", "command": "echo a"})
        upsert_cron_job({"name": "j1", "schedule": "0 * * * *", "command": "echo b"})
        jobs = load_cron_jobs()
        assert len(jobs) == 1
        assert jobs[0]["command"] == "echo b"

    def test_delete(self, tmp_data_dir):
        upsert_cron_job({"name": "j1", "schedule": "0 * * * *", "command": "echo a"})
        assert delete_cron_job("j1") is True
        assert load_cron_jobs() == []

    def test_delete_not_found(self, tmp_data_dir):
        assert delete_cron_job("nonexistent") is False

    def test_get_cron_job(self, tmp_data_dir):
        upsert_cron_job({"name": "j1", "schedule": "0 * * * *", "command": "echo a"})
        job = get_cron_job("j1")
        assert job is not None
        assert job["name"] == "j1"
        assert get_cron_job("nope") is None

    def test_set_job_enabled(self, tmp_data_dir):
        upsert_cron_job({"name": "j1", "schedule": "0 * * * *", "command": "echo a"})
        assert set_job_enabled("j1", False) is True
        job = get_cron_job("j1")
        assert job is not None
        assert job["enabled"] is False
        assert set_job_enabled("nope", True) is False

    def test_mark_job_run(self, tmp_data_dir):
        upsert_cron_job({"name": "j1", "schedule": "*/5 * * * *", "command": "echo a"})
        mark_job_run("j1", success=True)
        job = get_cron_job("j1")
        assert job is not None
        assert job["last_status"] == "success"
        assert "last_run" in job


# ---------------------------------------------------------------------------
# Cron scheduler
# ---------------------------------------------------------------------------


class TestCronScheduler:
    def test_jobs_due(self):
        now = datetime.now(timezone.utc)
        past = (now - timedelta(minutes=5)).isoformat()
        future = (now + timedelta(hours=1)).isoformat()

        jobs = [
            {"name": "due", "enabled": True, "schedule": "* * * * *", "next_run": past},
            {"name": "not-due", "enabled": True, "schedule": "0 * * * *", "next_run": future},
            {"name": "disabled", "enabled": False, "schedule": "* * * * *", "next_run": past},
        ]
        due = _jobs_due(jobs, now)
        assert len(due) == 1
        assert due[0]["name"] == "due"

    @pytest.mark.asyncio
    async def test_execute_job(self, tmp_data_dir):
        job = {
            "name": "test-echo",
            "command": "echo hello",
            "cwd": "/tmp",
            "schedule": "* * * * *",
        }
        entry = await execute_job(job)
        assert entry["status"] == "success"
        assert entry["returncode"] == 0
        assert "hello" in entry["stdout"]

    @pytest.mark.asyncio
    async def test_execute_job_failure(self, tmp_data_dir):
        job = {
            "name": "test-fail",
            "command": "exit 1",
            "cwd": "/tmp",
            "schedule": "* * * * *",
        }
        entry = await execute_job(job)
        assert entry["status"] == "failed"
        assert entry["returncode"] == 1

    @pytest.mark.asyncio
    async def test_scheduler_loop_once(self, tmp_data_dir):
        """Scheduler in once mode should complete without hanging."""
        upsert_cron_job({
            "name": "quick",
            "schedule": "* * * * *",
            "command": "echo done",
        })
        # Force the job to be due now
        jobs = load_cron_jobs()
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        for j in jobs:
            j["next_run"] = past
        save_cron_jobs(jobs)

        await run_scheduler_loop(once=True)
        # Job should have been marked as run
        job = get_cron_job("quick")
        assert job is not None
        assert "last_run" in job

    def test_history(self, tmp_data_dir):
        append_history({"name": "j1", "status": "success"})
        append_history({"name": "j2", "status": "failed"})
        history = load_history()
        assert len(history) == 2

        filtered = load_history(job_name="j1")
        assert len(filtered) == 1
        assert filtered[0]["name"] == "j1"


# ---------------------------------------------------------------------------
# Cron tools
# ---------------------------------------------------------------------------


class TestCronTools:
    @pytest.mark.asyncio
    async def test_cron_create_valid(self, tmp_data_dir, ctx):
        tool = CronCreateTool()
        args = CronCreateToolInput(
            name="my-job",
            schedule="*/5 * * * *",
            command="echo hi",
        )
        result = await tool.execute(args, ctx)
        assert not result.is_error
        assert "Created" in result.output
        assert len(load_cron_jobs()) == 1

    @pytest.mark.asyncio
    async def test_cron_create_invalid_schedule(self, tmp_data_dir, ctx):
        tool = CronCreateTool()
        args = CronCreateToolInput(
            name="bad", schedule="not-cron", command="echo"
        )
        result = await tool.execute(args, ctx)
        assert result.is_error
        assert "Invalid" in result.output

    @pytest.mark.asyncio
    async def test_cron_delete(self, tmp_data_dir, ctx):
        upsert_cron_job({"name": "j1", "schedule": "0 * * * *", "command": "echo"})
        tool = CronDeleteTool()
        result = await tool.execute(CronDeleteToolInput(name="j1"), ctx)
        assert not result.is_error
        assert "Deleted" in result.output

    @pytest.mark.asyncio
    async def test_cron_delete_not_found(self, tmp_data_dir, ctx):
        tool = CronDeleteTool()
        result = await tool.execute(CronDeleteToolInput(name="nope"), ctx)
        assert result.is_error

    @pytest.mark.asyncio
    async def test_cron_list_empty(self, tmp_data_dir, ctx):
        tool = CronListTool()
        result = await tool.execute(CronListToolInput(), ctx)
        assert not result.is_error
        assert "No cron jobs" in result.output

    @pytest.mark.asyncio
    async def test_cron_list_with_jobs(self, tmp_data_dir, ctx):
        upsert_cron_job({"name": "j1", "schedule": "0 * * * *", "command": "echo a"})
        upsert_cron_job({"name": "j2", "schedule": "*/5 * * * *", "command": "echo b"})
        tool = CronListTool()
        result = await tool.execute(CronListToolInput(), ctx)
        assert not result.is_error
        assert "j1" in result.output
        assert "j2" in result.output
        assert "Total jobs: 2" in result.output

    def test_cron_list_is_read_only(self):
        tool = CronListTool()
        assert tool.is_read_only(CronListToolInput()) is True


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------


class TestHeartbeat:
    @pytest.mark.asyncio
    async def test_check_no_subsystems(self, tmp_data_dir):
        hb = Heartbeat()
        status = await hb.check()
        assert "timestamp" in status
        assert status["cron_jobs_due"] == 0
        assert status["gateway_running"] is None
        assert status["tasks_running"] is None

    @pytest.mark.asyncio
    async def test_check_with_due_jobs(self, tmp_data_dir):
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        upsert_cron_job({"name": "j1", "schedule": "* * * * *", "command": "echo"})
        jobs = load_cron_jobs()
        for j in jobs:
            j["next_run"] = past
        save_cron_jobs(jobs)

        hb = Heartbeat()
        status = await hb.check()
        assert status["cron_jobs_due"] == 1

    @pytest.mark.asyncio
    async def test_check_with_gateway(self, tmp_data_dir):
        mock_gateway = MagicMock()
        mock_gateway.running = True
        mock_gateway.platform = Platform.TELEGRAM
        hb = Heartbeat(gateway=mock_gateway)
        status = await hb.check()
        assert status["gateway_running"] is True
        assert status["gateway_platform"] == "telegram"

    @pytest.mark.asyncio
    async def test_run_and_stop(self, tmp_data_dir):
        hb = Heartbeat(interval=1)
        task = asyncio.create_task(hb.run_forever())
        await asyncio.sleep(0.1)
        assert hb._running
        hb.stop()
        await asyncio.sleep(1.5)
        assert not hb._running
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Archive writer
# ---------------------------------------------------------------------------


class TestArchiveWriter:
    def test_write_and_read(self, tmp_path):
        writer = ArchiveWriter(path=tmp_path / "test_archive.jsonl")
        writer.archive_event("test_event", {"key": "value"})
        writer.archive_event("other_event", {"num": 42})

        events = writer.read_events()
        assert len(events) == 2
        assert events[0]["type"] == "test_event"
        assert events[0]["data"]["key"] == "value"

    def test_read_filtered(self, tmp_path):
        writer = ArchiveWriter(path=tmp_path / "test_archive.jsonl")
        writer.archive_event("a", {})
        writer.archive_event("b", {})
        writer.archive_event("a", {})

        events = writer.read_events(event_type="a")
        assert len(events) == 2

    def test_read_empty(self, tmp_path):
        writer = ArchiveWriter(path=tmp_path / "empty.jsonl")
        assert writer.read_events() == []

    def test_read_limited(self, tmp_path):
        writer = ArchiveWriter(path=tmp_path / "test_archive.jsonl")
        for i in range(10):
            writer.archive_event("evt", {"i": i})
        events = writer.read_events(limit=3)
        assert len(events) == 3
        assert events[0]["data"]["i"] == 7  # last 3 of 10
