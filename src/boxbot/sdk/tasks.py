"""Trigger and to-do management for agent scripts.

Shares the same backend as the manage_tasks core tool. This SDK module
is for complex multi-step task management within scripts — batch
operations, conditional logic, or combining task management with other
SDK calls.

Usage:
    from boxbot_sdk import tasks

    tasks.create_trigger(
        description="Dentist reminder for Jacob",
        instructions="Remind Jacob about his dentist appointment",
        fire_at="2026-02-21T15:30:00",
        for_person="Jacob"
    )

    tasks.create_todo(
        description="Return library books",
        notes="Due Saturday.",
        for_person="Jacob",
        due_date="2026-02-22"
    )

    for todo in tasks.list_todos(status="pending"):
        print(f"To-do: {todo.description}")
"""

from __future__ import annotations

from typing import Any

from . import _transport, _validators as v


class TriggerRecord:
    """A trigger record."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def description(self) -> str:
        return self._data.get("description", "")

    @property
    def instructions(self) -> str:
        return self._data.get("instructions", "")

    @property
    def status(self) -> str:
        return self._data.get("status", "")

    @property
    def fire_at(self) -> str | None:
        return self._data.get("fire_at")

    @property
    def fire_after(self) -> str | None:
        return self._data.get("fire_after")

    @property
    def cron(self) -> str | None:
        return self._data.get("cron")

    @property
    def person(self) -> str | None:
        return self._data.get("person")

    @property
    def for_person(self) -> str | None:
        return self._data.get("for_person")

    @property
    def created_at(self) -> str:
        return self._data.get("created_at", "")

    def __repr__(self) -> str:
        return (f"TriggerRecord(id={self.id!r}, "
                f"description={self.description!r}, "
                f"status={self.status!r})")


class TodoRecord:
    """A to-do item record."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @property
    def id(self) -> str:
        return self._data.get("id", "")

    @property
    def description(self) -> str:
        return self._data.get("description", "")

    @property
    def notes(self) -> str | None:
        return self._data.get("notes")

    @property
    def status(self) -> str:
        return self._data.get("status", "")

    @property
    def for_person(self) -> str | None:
        return self._data.get("for_person")

    @property
    def due_date(self) -> str | None:
        return self._data.get("due_date")

    @property
    def created_at(self) -> str:
        return self._data.get("created_at", "")

    def __repr__(self) -> str:
        return (f"TodoRecord(id={self.id!r}, "
                f"description={self.description!r}, "
                f"status={self.status!r})")


# --- Triggers ---

def create_trigger(description: str, instructions: str, *,
                   fire_at: str | None = None,
                   fire_after: str | None = None,
                   cron: str | None = None,
                   person: str | None = None,
                   for_person: str | None = None,
                   todo_id: str | None = None) -> None:
    """Create a trigger (wake condition).

    Triggers use AND logic — all specified conditions must be met.
    At least one condition (fire_at, fire_after, cron, or person) is required.

    Args:
        description: Human-readable trigger description.
        instructions: What the agent should do when triggered.
        fire_at: ISO datetime for point-in-time trigger.
        fire_after: Duration string for timer trigger (max 24h), e.g. "2h", "30m".
        cron: Cron expression for recurring trigger.
        person: Person name for presence trigger.
        for_person: Person this trigger is about (for context).
        todo_id: Link to a to-do item.
    """
    v.require_str(description, "description")
    v.require_str(instructions, "instructions")

    has_condition = any([fire_at, fire_after, cron, person])
    if not has_condition:
        raise ValueError(
            "At least one trigger condition is required: "
            "fire_at, fire_after, cron, or person"
        )

    payload: dict[str, Any] = {
        "description": description,
        "instructions": instructions,
    }
    if fire_at is not None:
        payload["fire_at"] = v.require_str(fire_at, "fire_at")
    if fire_after is not None:
        payload["fire_after"] = v.require_str(fire_after, "fire_after")
    if cron is not None:
        payload["cron"] = v.require_str(cron, "cron")
    if person is not None:
        payload["person"] = v.require_str(person, "person")
    if for_person is not None:
        payload["for_person"] = v.require_str(for_person, "for_person")
    if todo_id is not None:
        payload["todo_id"] = v.require_str(todo_id, "todo_id")

    _transport.emit_action("tasks.create_trigger", payload)


def list_triggers(*, status: str | None = None) -> list[TriggerRecord]:
    """List triggers.

    Args:
        status: Filter by status — active, expired, cancelled.

    Returns:
        List of TriggerRecord objects.
    """
    payload: dict[str, Any] = {}
    if status is not None:
        v.validate_one_of(status, "status", v.VALID_TRIGGER_STATUSES)
        payload["status"] = status

    response = _transport.request("tasks.list_triggers", payload, timeout=30)
    results = response.get("results", [])
    return [TriggerRecord(r) for r in results]


# --- To-do items ---

def create_todo(description: str, *,
                notes: str | None = None,
                for_person: str | None = None,
                due_date: str | None = None) -> None:
    """Create a to-do item.

    Args:
        description: Short description of the to-do.
        notes: Detailed notes (loaded on demand via get()).
        for_person: Person this to-do is for.
        due_date: Due date string (YYYY-MM-DD).
    """
    v.require_str(description, "description")

    payload: dict[str, Any] = {"description": description}
    if notes is not None:
        payload["notes"] = v.require_str(notes, "notes")
    if for_person is not None:
        payload["for_person"] = v.require_str(for_person, "for_person")
    if due_date is not None:
        payload["due_date"] = v.require_str(due_date, "due_date")

    _transport.emit_action("tasks.create_todo", payload)


def list_todos(*, status: str | None = None) -> list[TodoRecord]:
    """List to-do items.

    Args:
        status: Filter by status — pending, completed, cancelled.

    Returns:
        List of TodoRecord objects.
    """
    payload: dict[str, Any] = {}
    if status is not None:
        v.validate_one_of(status, "status", v.VALID_TODO_STATUSES)
        payload["status"] = status

    response = _transport.request("tasks.list_todos", payload, timeout=30)
    results = response.get("results", [])
    return [TodoRecord(r) for r in results]


# --- Shared operations ---

def get(item_id: str) -> TriggerRecord | TodoRecord:
    """Get full details for a trigger or to-do item.

    Args:
        item_id: Item ID (trigger or to-do).

    Returns:
        TriggerRecord or TodoRecord with full details.
    """
    v.require_str(item_id, "item_id")
    response = _transport.request("tasks.get", {"id": item_id}, timeout=30)
    item_type = response.get("item_type", "")
    if item_type == "trigger":
        return TriggerRecord(response)
    return TodoRecord(response)


def complete(item_id: str) -> None:
    """Complete a to-do item.

    Args:
        item_id: To-do item ID.
    """
    v.require_str(item_id, "item_id")
    _transport.emit_action("tasks.complete", {"id": item_id})


def cancel(item_id: str) -> None:
    """Cancel a trigger or to-do item.

    Args:
        item_id: Item ID (trigger or to-do).
    """
    v.require_str(item_id, "item_id")
    _transport.emit_action("tasks.cancel", {"id": item_id})
