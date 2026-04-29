"""Per-tool-call context shared between the agent and tool implementations.

Tools are singletons registered globally; they have no constructor-time
hook for "the conversation that triggered me." For the (small, but real)
set of tools that need conversation-scoped state — currently
``execute_script`` reaching the conversation's long-lived sandbox
runner — the agent sets a ContextVar around each ``tool.execute()``
invocation, and the tool reads it with the helper here.

A ContextVar (rather than a thread-local) is the right shape because
tool dispatch is asyncio-driven: each conversation's generation runs in
its own task, with its own copy of the ContextVar set on entry. Two
conversations dispatching tools in parallel see their own contexts
independently.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from boxbot.core.conversation import Conversation


current_conversation: ContextVar["Conversation | None"] = ContextVar(
    "current_conversation", default=None
)


def get_current_conversation() -> "Conversation | None":
    """Return the conversation whose generation is currently running this
    tool call, or None if the tool was invoked outside a conversation
    (tests, ad-hoc calls)."""
    return current_conversation.get()
