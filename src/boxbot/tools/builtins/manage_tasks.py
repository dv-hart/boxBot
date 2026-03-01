"""manage_tasks tool — manage triggers (wake conditions) and to-do list.

The agent's internal planning system. Routes CRUD actions to the scheduler
module (boxbot.core.scheduler). Distinct from the family calendar, which
is managed by calendar skills.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


class ManageTasksTool(Tool):
    """Manage triggers (wake conditions) and the persistent to-do list."""

    name = "manage_tasks"
    description = (
        "Manage your triggers (wake conditions) and to-do list. Actions: "
        "create_trigger (set wake conditions with AND logic — fire_at, "
        "fire_after, cron, person), create_todo (add persistent action items), "
        "list (view triggers/todos), get (full details), update (modify fields), "
        "complete (mark todo done), cancel (cancel trigger or todo). "
        "This is your internal planning system, not the family calendar."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create_trigger",
                    "create_todo",
                    "list",
                    "get",
                    "update",
                    "complete",
                    "cancel",
                ],
                "description": "The action to perform.",
            },
            # create_trigger fields
            "description": {
                "type": "string",
                "description": "Human-readable summary (create_trigger, create_todo).",
            },
            "instructions": {
                "type": "string",
                "description": "What to do when the trigger fires (create_trigger).",
            },
            "fire_at": {
                "type": "string",
                "description": "ISO datetime for point-in-time trigger.",
            },
            "fire_after": {
                "type": "string",
                "description": (
                    "Relative duration (e.g., '30m', '2h'). Max 24h. "
                    "Converted to fire_at on creation."
                ),
            },
            "cron": {
                "type": "string",
                "description": (
                    "Cron expression for recurring triggers. "
                    "Mutually exclusive with fire_at/fire_after."
                ),
            },
            "person": {
                "type": "string",
                "description": (
                    "Person name — fires when detected. "
                    "Combine with time fields for compound triggers (AND)."
                ),
            },
            "for_person": {
                "type": "string",
                "description": "Who this relates to (context).",
            },
            "expires": {
                "type": "string",
                "description": "ISO datetime expiry override.",
            },
            "todo_id": {
                "type": "string",
                "description": "Link trigger to an existing to-do item.",
            },
            # create_todo fields
            "notes": {
                "type": "string",
                "description": (
                    "Detailed context and instructions for a to-do item. "
                    "Loaded on demand via 'get', not shown in lists."
                ),
            },
            "due_date": {
                "type": "string",
                "description": "ISO date soft deadline for a to-do item.",
            },
            # list fields
            "type": {
                "type": "string",
                "enum": ["triggers", "todos", "all"],
                "description": "What to list (default: 'all').",
            },
            "status": {
                "type": "string",
                "enum": ["active", "completed", "expired", "all"],
                "description": "Filter by status (default: 'active').",
            },
            # get/update/complete/cancel fields
            "id": {
                "type": "string",
                "description": "Trigger or to-do ID.",
            },
        },
        "required": ["action"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        action: str = kwargs["action"]

        try:
            if action == "create_trigger":
                return await self._create_trigger(kwargs)
            elif action == "create_todo":
                return await self._create_todo(kwargs)
            elif action == "list":
                return await self._list(kwargs)
            elif action == "get":
                return await self._get(kwargs)
            elif action == "update":
                return await self._update(kwargs)
            elif action == "complete":
                return await self._complete(kwargs)
            elif action == "cancel":
                return await self._cancel(kwargs)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            logger.exception("manage_tasks error: action=%s", action)
            return json.dumps({"error": str(e)})

    async def _create_trigger(self, kwargs: dict[str, Any]) -> str:
        from boxbot.core.scheduler import create_trigger

        description = kwargs.get("description")
        instructions = kwargs.get("instructions")
        if not description or not instructions:
            return json.dumps({
                "error": "create_trigger requires 'description' and 'instructions'."
            })

        trigger_id = await create_trigger(
            description=description,
            instructions=instructions,
            fire_at=kwargs.get("fire_at"),
            fire_after=kwargs.get("fire_after"),
            cron=kwargs.get("cron"),
            person=kwargs.get("person"),
            for_person=kwargs.get("for_person"),
            expires=kwargs.get("expires"),
            todo_id=kwargs.get("todo_id"),
        )

        logger.info("Created trigger: %s (%s)", trigger_id, description)
        return json.dumps({
            "status": "created",
            "id": trigger_id,
            "description": description,
        })

    async def _create_todo(self, kwargs: dict[str, Any]) -> str:
        from boxbot.core.scheduler import create_todo

        description = kwargs.get("description")
        if not description:
            return json.dumps({
                "error": "create_todo requires 'description'."
            })

        todo_id = await create_todo(
            description=description,
            notes=kwargs.get("notes"),
            for_person=kwargs.get("for_person"),
            due_date=kwargs.get("due_date"),
        )

        logger.info("Created todo: %s (%s)", todo_id, description)
        return json.dumps({
            "status": "created",
            "id": todo_id,
            "description": description,
        })

    async def _list(self, kwargs: dict[str, Any]) -> str:
        from boxbot.core.scheduler import list_todos, list_triggers

        list_type = kwargs.get("type", "all")
        status_filter = kwargs.get("status", "active")
        for_person = kwargs.get("for_person")

        result: dict[str, Any] = {}

        # Map status filter for triggers
        trigger_status = None if status_filter == "all" else status_filter
        todo_status = None if status_filter == "all" else (
            "pending" if status_filter == "active" else status_filter
        )

        if list_type in ("triggers", "all"):
            triggers = await list_triggers(
                status=trigger_status, for_person=for_person
            )
            result["triggers"] = [
                {
                    "id": t["id"],
                    "description": t["description"],
                    "fire_at": t.get("fire_at"),
                    "cron": t.get("cron"),
                    "person": t.get("person"),
                    "for_person": t.get("for_person"),
                    "status": t["status"],
                }
                for t in triggers
            ]

        if list_type in ("todos", "all"):
            todos = await list_todos(
                status=todo_status, for_person=for_person
            )
            result["todos"] = [
                {
                    "id": t["id"],
                    "description": t["description"],
                    "for_person": t.get("for_person"),
                    "due_date": t.get("due_date"),
                    "status": t["status"],
                }
                for t in todos
            ]

        return json.dumps(result)

    async def _get(self, kwargs: dict[str, Any]) -> str:
        from boxbot.core.scheduler import get_todo, get_trigger

        item_id: str | None = kwargs.get("id")
        if not item_id:
            return json.dumps({"error": "get requires 'id'."})

        # Try trigger first, then todo
        if item_id.startswith("t_"):
            trigger = await get_trigger(item_id)
            if trigger is None:
                return json.dumps({"error": f"Trigger {item_id} not found."})
            return json.dumps({
                "id": trigger["id"],
                "type": "trigger",
                "description": trigger["description"],
                "instructions": trigger["instructions"],
                "fire_at": trigger.get("fire_at"),
                "cron": trigger.get("cron"),
                "person": trigger.get("person"),
                "for_person": trigger.get("for_person"),
                "todo_id": trigger.get("todo_id"),
                "status": trigger["status"],
                "created_at": trigger["created_at"],
                "expires": trigger.get("expires"),
                "last_fired": trigger.get("last_fired"),
                "fire_count": trigger.get("fire_count", 0),
            })
        elif item_id.startswith("d_"):
            todo = await get_todo(item_id)
            if todo is None:
                return json.dumps({"error": f"To-do {item_id} not found."})
            return json.dumps({
                "id": todo["id"],
                "type": "todo",
                "description": todo["description"],
                "notes": todo.get("notes"),
                "for_person": todo.get("for_person"),
                "due_date": todo.get("due_date"),
                "status": todo["status"],
                "created_at": todo["created_at"],
                "completed_at": todo.get("completed_at"),
            })
        else:
            # Try both
            trigger = await get_trigger(item_id)
            if trigger is not None:
                return json.dumps({
                    "id": trigger["id"],
                    "type": "trigger",
                    "description": trigger["description"],
                    "instructions": trigger["instructions"],
                    "status": trigger["status"],
                    "created_at": trigger["created_at"],
                })
            todo = await get_todo(item_id)
            if todo is not None:
                return json.dumps({
                    "id": todo["id"],
                    "type": "todo",
                    "description": todo["description"],
                    "notes": todo.get("notes"),
                    "status": todo["status"],
                    "created_at": todo["created_at"],
                })
            return json.dumps({"error": f"Item {item_id} not found."})

    async def _update(self, kwargs: dict[str, Any]) -> str:
        from boxbot.core.scheduler import update_todo, update_trigger

        item_id: str | None = kwargs.get("id")
        if not item_id:
            return json.dumps({"error": "update requires 'id'."})

        # Collect updatable fields
        if item_id.startswith("t_"):
            fields: dict[str, Any] = {}
            for key in (
                "description", "instructions", "for_person", "expires",
            ):
                if key in kwargs and kwargs[key] is not None:
                    fields[key] = kwargs[key]
            if not fields:
                return json.dumps({"error": "No fields to update."})
            updated = await update_trigger(item_id, **fields)
            if not updated:
                return json.dumps({"error": f"Trigger {item_id} not found."})
            return json.dumps({
                "status": "updated",
                "id": item_id,
                "updated_fields": list(fields.keys()),
            })
        elif item_id.startswith("d_"):
            fields = {}
            for key in (
                "description", "notes", "for_person", "due_date",
            ):
                if key in kwargs and kwargs[key] is not None:
                    fields[key] = kwargs[key]
            if not fields:
                return json.dumps({"error": "No fields to update."})
            updated = await update_todo(item_id, **fields)
            if not updated:
                return json.dumps({"error": f"To-do {item_id} not found."})
            return json.dumps({
                "status": "updated",
                "id": item_id,
                "updated_fields": list(fields.keys()),
            })
        else:
            return json.dumps({
                "error": "Cannot determine item type. Use 't_' prefix for "
                "triggers or 'd_' prefix for to-dos."
            })

    async def _complete(self, kwargs: dict[str, Any]) -> str:
        from boxbot.core.scheduler import complete_todo

        item_id: str | None = kwargs.get("id")
        if not item_id:
            return json.dumps({"error": "complete requires 'id'."})

        completed = await complete_todo(item_id)
        if not completed:
            return json.dumps({"error": f"To-do {item_id} not found."})

        return json.dumps({
            "status": "completed",
            "id": item_id,
            "message": f"To-do {item_id} marked as completed.",
        })

    async def _cancel(self, kwargs: dict[str, Any]) -> str:
        from boxbot.core.scheduler import cancel_todo, cancel_trigger

        item_id: str | None = kwargs.get("id")
        if not item_id:
            return json.dumps({"error": "cancel requires 'id'."})

        if item_id.startswith("t_"):
            cancelled = await cancel_trigger(item_id)
            item_type = "Trigger"
        elif item_id.startswith("d_"):
            cancelled = await cancel_todo(item_id)
            item_type = "To-do"
        else:
            # Try trigger first, then todo
            cancelled = await cancel_trigger(item_id)
            item_type = "Trigger"
            if not cancelled:
                cancelled = await cancel_todo(item_id)
                item_type = "To-do"

        if not cancelled:
            return json.dumps({"error": f"Item {item_id} not found."})

        return json.dumps({
            "status": "cancelled",
            "id": item_id,
            "message": f"{item_type} {item_id} cancelled.",
        })
