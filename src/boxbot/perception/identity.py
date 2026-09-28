"""Camera-free identity core: person store + enrollment + session commit.

Runs on every device class. On the Pi the visual ``PerceptionPipeline``
composes this service; on camera-less hardware it runs
alone, so voice identity — cloud matching, ``identify_person``
enrollment, and the ``VoiceSessionEnded`` commit — works without a
camera or Hailo NPU.

Imports here must stay free of cv2/hailo: this module is what voice
identity reaches for on devices where the visual pipeline cannot even
be imported.

Usage:
    from boxbot.perception.identity import IdentityService, get_identity

    identity = IdentityService()
    await identity.start()
    # ...
    await identity.stop()

    # For tool/adapter access:
    get_identity().enrollment
"""

from __future__ import annotations

import logging

from boxbot.core.events import VoiceSessionEnded, get_event_bus
from boxbot.perception.clouds import CloudStore
from boxbot.perception.enrollment import EnrollmentManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton for tool/adapter access
# ---------------------------------------------------------------------------

_identity_instance: IdentityService | None = None


def get_identity() -> IdentityService:
    """Return the running identity service instance.

    Raises RuntimeError if the service has not been started.
    """
    if _identity_instance is None:
        raise RuntimeError(
            "Identity service not started. "
            "Call IdentityService.start() during system startup."
        )
    return _identity_instance


class IdentityService:
    """Owns the person store and enrollment session state.

    Subscribes to ``VoiceSessionEnded`` and flushes enrollment buffers to
    the store — the only path by which buffered embeddings are persisted.

    Args:
        cloud_store: Existing store to wrap. Created (and owned, i.e.
            closed on :meth:`stop`) when omitted.
    """

    def __init__(self, cloud_store: CloudStore | None = None) -> None:
        self._cloud_store = cloud_store
        self._owns_cloud_store = cloud_store is None
        self._enrollment: EnrollmentManager | None = None
        self._started = False

    # ── Lifecycle ──────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize the store and subscribe to session-end commits.

        Idempotent — the pipeline calls this on a service main.py may
        have already started.
        """
        global _identity_instance

        if self._started:
            return

        if self._cloud_store is None:
            self._cloud_store = CloudStore()
            await self._cloud_store.initialize()

        self._enrollment = EnrollmentManager(self._cloud_store)

        get_event_bus().subscribe(
            VoiceSessionEnded, self._on_voice_session_ended
        )

        self._started = True
        _identity_instance = self
        logger.info("Identity service started")

    async def stop(self) -> None:
        """Unsubscribe and close the store if this service created it."""
        global _identity_instance

        if not self._started:
            return

        get_event_bus().unsubscribe(
            VoiceSessionEnded, self._on_voice_session_ended
        )

        if self._owns_cloud_store and self._cloud_store is not None:
            await self._cloud_store.close()

        self._started = False
        _identity_instance = None
        logger.info("Identity service stopped")

    # ── Public API ─────────────────────────────────────────────────

    @property
    def cloud_store(self) -> CloudStore | None:
        """Cloud store for embedding and person-record persistence."""
        return self._cloud_store

    @property
    def enrollment(self) -> EnrollmentManager | None:
        """Enrollment manager for the identify_person tool."""
        return self._enrollment

    # ── Event handlers ─────────────────────────────────────────────

    async def _on_voice_session_ended(self, event: VoiceSessionEnded) -> None:
        """Voice session fully ended — commit enrollment buffers.

        A single voice session can span many conversations (each wake
        word → transcript → agent turn is its own ConversationStarted/
        Ended), but the enrollment buffers accumulate across all of
        them. We flush once the voice session itself ends, routing each
        ref's buffered voice + visual embeddings to the person their
        session claim points at, or dropping them if no identity was
        ever resolved.
        """
        if self._enrollment is None:
            return
        try:
            summary = await self._enrollment.commit_session()
            logger.info(
                "Enrollment committed on voice session end (%s): %s",
                event.conversation_id, summary,
            )
        except Exception:
            logger.exception(
                "commit_session failed for voice session %s",
                event.conversation_id,
            )
