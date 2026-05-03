"""CostEvent — one row of cost_log in dataclass form.

Provider-neutral: Anthropic-shaped fields (tokens, cache splits,
is_batch) and ElevenLabs-shaped fields (characters, audio seconds)
both live here. Unused fields default to 0/None so a TTS event and a
Claude turn can share the same writer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CostEvent:
    purpose: str
    provider: str
    model: str
    cost_usd: float

    # Anthropic-shaped
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_5m_tokens: int = 0
    cache_write_1h_tokens: int = 0
    is_batch: bool = False

    # ElevenLabs-shaped
    character_count: int = 0
    audio_seconds: float = 0.0

    # Loop bookkeeping (web_search sub-agent, etc.)
    iterations: int = 0

    # Cross-event correlation (e.g. all rows for a single user turn)
    correlation_id: str | None = None

    # Anything else worth keeping (conversation_id, batch_id, request_id…)
    metadata: dict | None = field(default=None)
