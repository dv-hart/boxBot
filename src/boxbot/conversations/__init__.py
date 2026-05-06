"""Persistent conversation storage.

Today only the WhatsApp channel uses this — voice and trigger keep
their transient in-memory threads. The store gives WhatsApp:

  * Threads that survive process restart.
  * A long, time-based active window (default 4 hours) instead of a
    voice-style silence timer.
  * A sweep-based extraction trigger that fires once per closed
    conversation, not every 3 minutes.

See ``store.ConversationStore`` for the persistence API and
``boxbot.core.conversation.Conversation``'s ``lifecycle_mode`` for
how the in-memory object integrates.
"""

from boxbot.conversations.store import (
    ConversationRecord,
    ConversationStore,
    TurnRecord,
)

__all__ = [
    "ConversationRecord",
    "ConversationStore",
    "TurnRecord",
]
