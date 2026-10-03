"""Tests for trigger → integration-script execution (no LLM wake).

A trigger naming ``run_integration`` runs that integration when it
fires instead of minting a conversation. Success is silent; a failed
run — or an output carrying the reserved ``escalate`` key — wakes the
agent with the trigger's ``instructions`` plus the script output.

Also covers the security boundary that makes unattended execution
safe: action contexts created by the integration runner are marked
``origin="integration"`` and are structurally denied the whole
``tasks.*`` surface.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from boxbot.core.events import TriggerFired
from boxbot.core.scheduler import create_trigger, get_trigger
from boxbot.integrations.manifest import IntegrationMeta
from boxbot.tools._sandbox_actions import (
    ActionContext,
    _handle_tasks_action,
)


@pytest.fixture(autouse=True)
def patch_scheduler_db(tmp_path):
    """Point the scheduler module's DB_PATH to a temp directory."""
    test_db = tmp_path / "scheduler" / "scheduler.db"
    with patch("boxbot.core.scheduler.DB_PATH", test_db):
        yield test_db


def _meta(name: str = "porch_light", **inputs) -> IntegrationMeta:
    """A minimal on-disk-shaped integration manifest."""
    return IntegrationMeta(
        name=name,
        description="Test integration.",
        inputs=inputs,
        outputs={},
        secrets=(),
        timeout=10,
        root_path=Path("/nonexistent") / name,
        manifest_path=Path("/nonexistent") / name / "manifest.yaml",
        script_path=Path("/nonexistent") / name / "script.py",
    )


def _patch_lookup(meta: IntegrationMeta | None):
    """Patch the loader lookup used by create_trigger's validation."""
    return patch(
        "boxbot.integrations.loader.get_integration",
        return_value=meta,
    )


# ---------------------------------------------------------------------------
# Creation + create-time validation
# ---------------------------------------------------------------------------


class TestCreateRunIntegrationTrigger:
    @pytest.mark.asyncio
    async def test_stored_and_round_trips(self):
        with _patch_lookup(_meta(after_hour={"type": "int"})):
            tid = await create_trigger(
                description="Porch light on late-night detection",
                instructions="Porch-light scene failed — say what broke.",
                cron="*/2 * * * *",
                run_integration="porch_light",
                run_inputs={"after_hour": 21},
            )
        trigger = await get_trigger(tid)
        assert trigger["run_integration"] == "porch_light"
        assert trigger["run_inputs"] == '{"after_hour": 21}'

    @pytest.mark.asyncio
    async def test_unknown_integration_rejected(self):
        with _patch_lookup(None):
            with pytest.raises(ValueError, match="unknown integration"):
                await create_trigger(
                    description="x", instructions="y",
                    cron="*/5 * * * *",
                    run_integration="nope",
                )

    @pytest.mark.asyncio
    async def test_bad_inputs_rejected_at_creation(self):
        """Bad wiring must fail on create, not at 3 a.m. when it fires."""
        with _patch_lookup(_meta(after_hour={"type": "int"})):
            with pytest.raises(ValueError, match="unexpected inputs"):
                await create_trigger(
                    description="x", instructions="y",
                    cron="*/5 * * * *",
                    run_integration="porch_light",
                    run_inputs={"typoed_key": 1},
                )

    @pytest.mark.asyncio
    async def test_missing_required_input_rejected_at_creation(self):
        with _patch_lookup(_meta(after_hour={"type": "int", "required": True})):
            with pytest.raises(ValueError, match="requires input"):
                await create_trigger(
                    description="x", instructions="y",
                    cron="*/5 * * * *",
                    run_integration="porch_light",
                )

    @pytest.mark.asyncio
    async def test_run_inputs_without_integration_rejected(self):
        with pytest.raises(ValueError, match="requires run_integration"):
            await create_trigger(
                description="x", instructions="y",
                cron="*/5 * * * *",
                run_inputs={"a": 1},
            )

    @pytest.mark.asyncio
    async def test_plain_trigger_unaffected(self):
        tid = await create_trigger(
            description="Dentist", instructions="Remind Jacob",
            fire_after="30m",
        )
        trigger = await get_trigger(tid)
        assert trigger["run_integration"] is None
        assert trigger["run_inputs"] is None


# ---------------------------------------------------------------------------
# Firing: run the script, don't wake the model
# ---------------------------------------------------------------------------


def _agent():
    """A BoxBotAgent with the conversation seam stubbed out."""
    from boxbot.core.agent import BoxBotAgent

    mem = MagicMock()
    mem.read_system_memory = MagicMock(return_value="")
    agent = BoxBotAgent(memory_store=mem)
    agent._client = MagicMock()
    agent._start_trigger_conversation = AsyncMock()
    return agent


def _event(**overrides) -> TriggerFired:
    fields = {
        "trigger_id": "t_abc123",
        "description": "Porch light scene",
        "instructions": "Porch-light scene failed — say what broke.",
        "run_integration": "porch_light",
        "run_inputs": {"after_hour": 21},
    }
    fields.update(overrides)
    return TriggerFired(**fields)


def _patch_run(result=None, side_effect=None):
    return patch(
        "boxbot.integrations.runner.run",
        new=AsyncMock(return_value=result, side_effect=side_effect),
    )


class TestFireRunsIntegration:
    @pytest.mark.asyncio
    async def test_runs_integration_without_conversation(self):
        agent = _agent()
        with _patch_run({"status": "ok", "output": {"lit": True}}) as run:
            await agent._on_trigger_fired(_event())
        run.assert_awaited_once_with("porch_light", {"after_hour": 21})
        agent._start_trigger_conversation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_silent_on_success_without_output(self):
        agent = _agent()
        with _patch_run({"status": "ok", "output": None}):
            await agent._on_trigger_fired(_event())
        agent._start_trigger_conversation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_escalate_key_wakes_agent(self):
        agent = _agent()
        with _patch_run(
            {"status": "ok", "output": {"escalate": "Back door open 14 min"}}
        ):
            await agent._on_trigger_fired(_event())
        agent._start_trigger_conversation.assert_awaited_once()
        text = agent._start_trigger_conversation.await_args.args[1]
        assert "Back door open 14 min" in text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("escalate", [True, 123])
    async def test_non_string_truthy_escalate_wakes_agent(self, escalate):
        """A non-string escalate key still escalates, coerced to str."""
        agent = _agent()
        with _patch_run({"status": "ok", "output": {"escalate": escalate}}):
            await agent._on_trigger_fired(_event())
        agent._start_trigger_conversation.assert_awaited_once()
        text = agent._start_trigger_conversation.await_args.args[1]
        assert str(escalate) in text

    @pytest.mark.asyncio
    @pytest.mark.parametrize("escalate", ["", "   ", False, 0, None])
    async def test_falsey_escalate_stays_silent(self, escalate):
        agent = _agent()
        with _patch_run({"status": "ok", "output": {"escalate": escalate}}):
            await agent._on_trigger_fired(_event())
        agent._start_trigger_conversation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_run_wakes_agent(self):
        agent = _agent()
        with _patch_run(
            {"status": "error", "error": "device bridge unreachable"}
        ):
            await agent._on_trigger_fired(_event())
        agent._start_trigger_conversation.assert_awaited_once()
        text = agent._start_trigger_conversation.await_args.args[1]
        assert "device bridge unreachable" in text
        assert "status: error" in text

    @pytest.mark.asyncio
    async def test_timeout_wakes_agent(self):
        agent = _agent()
        with _patch_run({"status": "timeout", "error": "exceeded 10s"}):
            await agent._on_trigger_fired(_event())
        agent._start_trigger_conversation.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_runner_exception_wakes_agent(self):
        agent = _agent()
        with _patch_run(side_effect=RuntimeError("unknown integration")):
            await agent._on_trigger_fired(_event())
        agent._start_trigger_conversation.assert_awaited_once()
        text = agent._start_trigger_conversation.await_args.args[1]
        assert "unknown integration" in text

    @pytest.mark.asyncio
    async def test_plain_trigger_still_opens_conversation(self):
        agent = _agent()
        with _patch_run({"status": "ok"}) as run:
            await agent._on_trigger_fired(
                _event(run_integration=None, run_inputs=None)
            )
        run.assert_not_awaited()
        agent._start_trigger_conversation.assert_awaited_once()


# ---------------------------------------------------------------------------
# Security: origin="integration" denials
# ---------------------------------------------------------------------------


class TestIntegrationOriginDenied:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("sub", ["create_trigger", "list_triggers", "cancel"])
    async def test_tasks_actions_denied(self, sub):
        result = await _handle_tasks_action(
            f"tasks.{sub}", {}, ActionContext(origin="integration"),
        )
        assert result["status"] == "error"
        assert "not available to unattended scripts" in result["message"]

    @pytest.mark.asyncio
    async def test_agent_origin_reaches_the_scheduler(self):
        """Default origin is unaffected — execute_script keeps working."""
        result = await _handle_tasks_action(
            "tasks.create_trigger",
            {
                "description": "Dentist",
                "instructions": "Remind Jacob",
                "fire_after": "30m",
            },
            ActionContext(),
        )
        assert result["status"] == "ok"
        assert result["id"].startswith("t_")

    def test_default_origin_is_agent(self):
        assert ActionContext().origin == "agent"

    @pytest.mark.asyncio
    async def test_absent_context_is_not_an_integration(self):
        """In-process callers pass ctx=None and stay unguarded."""
        result = await _handle_tasks_action("tasks.list_triggers", {}, None)
        assert result["status"] == "ok"

    @pytest.mark.asyncio
    async def test_runner_marks_its_context_as_integration(self):
        """The runner's action pump must set the origin marker."""
        import asyncio

        from boxbot.integrations.runner import SDK_ACTION_MARKER, _pump_actions

        seen: list[str] = []

        async def fake_process_action(action, ctx):
            seen.append(ctx.origin)
            return {"status": "ok"}

        stdout = asyncio.StreamReader()
        stdout.feed_data(f'{SDK_ACTION_MARKER}{{"type": "tasks.get"}}\n'.encode())
        stdout.feed_eof()
        proc = MagicMock()
        proc.stdout = stdout
        proc.stdin = MagicMock()

        with patch(
            "boxbot.tools._sandbox_actions.process_action",
            new=fake_process_action,
        ):
            await _pump_actions(proc)

        assert seen == ["integration"]


# ---------------------------------------------------------------------------
# run_script — trigger-run workspace scripts (no manifest, no registry)
# ---------------------------------------------------------------------------


def _patch_workspace(tmp_path: Path, *files: str):
    """Point the workspace root at tmp_path, pre-creating ``files``."""
    for rel in files:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("print('chore')\n", encoding="utf-8")
    return patch("boxbot.workspace.store.DEFAULT_ROOT", tmp_path)


class TestCreateRunScriptTrigger:
    @pytest.mark.asyncio
    async def test_stored_and_round_trips(self, tmp_path):
        with _patch_workspace(tmp_path, "scripts/lockup.py"):
            tid = await create_trigger(
                description="Nightly lock-up",
                instructions="Lock-up check hit a problem.",
                cron="0 22 * * *",
                run_script="scripts/lockup.py",
                run_inputs={"strict": True},
            )
        trigger = await get_trigger(tid)
        assert trigger["run_script"] == "scripts/lockup.py"
        assert trigger["run_inputs"] == '{"strict": true}'

    @pytest.mark.asyncio
    async def test_missing_file_rejected_at_creation(self, tmp_path):
        with _patch_workspace(tmp_path):
            with pytest.raises(ValueError, match="run_script"):
                await create_trigger(
                    description="x", instructions="y",
                    cron="0 22 * * *",
                    run_script="scripts/typo.py",
                )

    @pytest.mark.asyncio
    async def test_escaping_path_rejected(self, tmp_path):
        with _patch_workspace(tmp_path):
            with pytest.raises(ValueError, match="run_script"):
                await create_trigger(
                    description="x", instructions="y",
                    cron="0 22 * * *",
                    run_script="../outside.py",
                )

    @pytest.mark.asyncio
    async def test_non_py_file_rejected(self, tmp_path):
        with _patch_workspace(tmp_path, "notes/todo.md"):
            with pytest.raises(ValueError, match=r"\.py"):
                await create_trigger(
                    description="x", instructions="y",
                    cron="0 22 * * *",
                    run_script="notes/todo.md",
                )

    @pytest.mark.asyncio
    async def test_mutually_exclusive_with_run_integration(self, tmp_path):
        with _patch_workspace(tmp_path, "scripts/lockup.py"):
            with pytest.raises(ValueError, match="mutually exclusive"):
                await create_trigger(
                    description="x", instructions="y",
                    cron="0 22 * * *",
                    run_integration="porch_light",
                    run_script="scripts/lockup.py",
                )

    @pytest.mark.asyncio
    async def test_run_inputs_allowed_with_run_script(self, tmp_path):
        with _patch_workspace(tmp_path, "scripts/lockup.py"):
            tid = await create_trigger(
                description="x", instructions="y",
                cron="0 22 * * *",
                run_script="scripts/lockup.py",
                run_inputs={"a": 1},
            )
        assert (await get_trigger(tid))["run_script"] == "scripts/lockup.py"


def _patch_run_script(result=None, side_effect=None):
    return patch(
        "boxbot.integrations.runner.run_workspace_script",
        new=AsyncMock(return_value=result, side_effect=side_effect),
    )


class TestFireRunsScript:
    @pytest.mark.asyncio
    async def test_runs_script_without_conversation(self):
        agent = _agent()
        event = _event(run_integration=None,
                       run_script="scripts/lockup.py",
                       run_inputs={"strict": True})
        with _patch_run_script({"status": "ok", "output": None}) as run:
            await agent._on_trigger_fired(event)
        run.assert_awaited_once_with("scripts/lockup.py", {"strict": True})
        agent._start_trigger_conversation.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_script_escalate_wakes_agent(self):
        agent = _agent()
        event = _event(run_integration=None, run_script="scripts/lockup.py")
        with _patch_run_script(
            {"status": "ok", "output": {"escalate": "garage stuck open"}}
        ):
            await agent._on_trigger_fired(event)
        agent._start_trigger_conversation.assert_awaited_once()
        text = agent._start_trigger_conversation.await_args.args[1]
        assert "garage stuck open" in text

    @pytest.mark.asyncio
    async def test_script_failure_wakes_agent_with_name(self):
        agent = _agent()
        event = _event(run_integration=None, run_script="scripts/lockup.py")
        with _patch_run_script({"status": "error", "error": "boom"}):
            await agent._on_trigger_fired(event)
        text = agent._start_trigger_conversation.await_args.args[1]
        assert "script:scripts/lockup.py" in text
        assert "boom" in text

    @pytest.mark.asyncio
    async def test_runner_exception_wakes_agent(self):
        agent = _agent()
        event = _event(run_integration=None, run_script="scripts/lockup.py")
        with _patch_run_script(side_effect=RuntimeError("sandbox down")):
            await agent._on_trigger_fired(event)
        text = agent._start_trigger_conversation.await_args.args[1]
        assert "sandbox down" in text
