"""Unified cost tracking for external API calls.

One cost log, one writer, one place that turns a provider response
into a row. Pricing lives in ``config/pricing.yaml`` — never in source.
Where the SDK returns a dollar amount directly (Agent SDK
``ResultMessage.total_cost_usd``), that value is used verbatim;
everywhere else we compute from API-returned units (tokens, billed
characters, measured audio seconds) using the YAML prices.

Public surface:

    CostEvent                       # dataclass holding one row's worth
    record(store, event)            # append to cost_log
    from_anthropic_usage(...)       # raw messages.create response
    from_agent_sdk_result(...)      # claude_agent_sdk ResultMessage
    from_elevenlabs_tts(...)        # billed chars from x-character-count
    from_elevenlabs_stt(...)        # measured input audio seconds

Parts 2-5 of the cost-tracking refactor wire these into the main
agent loop, web search sub-agent, ElevenLabs adapters, and dream
poller respectively.
"""

from boxbot.cost.compute import (
    from_agent_sdk_result,
    from_anthropic_usage,
    from_elevenlabs_stt,
    from_elevenlabs_tts,
)
from boxbot.cost.event import CostEvent
from boxbot.cost.pricing import Pricing, get_pricing, reload_pricing
from boxbot.cost.record import record

__all__ = [
    "CostEvent",
    "Pricing",
    "from_agent_sdk_result",
    "from_anthropic_usage",
    "from_elevenlabs_stt",
    "from_elevenlabs_tts",
    "get_pricing",
    "record",
    "reload_pricing",
]
