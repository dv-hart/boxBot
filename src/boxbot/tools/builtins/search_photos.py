"""search_photos tool — search and retrieve photos from the photo library.

Routes to the shared photo search backend (boxbot.photos.search). Two modes:
search (ranked results with metadata) and get (full details for a single
photo). Uses lazy imports since the photo system may not be built yet.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from boxbot.tools.base import Tool

logger = logging.getLogger(__name__)


class SearchPhotosTool(Tool):
    """Search and retrieve photos from the photo library."""

    name = "search_photos"
    description = (
        "Photo library. Modes: 'search' (ranked text match, filterable by "
        "tags, people, date range, source), 'get' (full detail by "
        "photo_id). To put results on screen: "
        "switch_display('picture', args={'image_ids': [...]})."
    )
    parameters = {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["search", "get"],
                "description": "'search' = ranked results. 'get' = one photo.",
            },
            "query": {
                "type": "string",
                "description": (
                    "Text match against photo descriptions. Required for "
                    "'search'."
                ),
            },
            "photo_id": {
                "type": "string",
                "description": "Required for 'get'.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by tags. AND.",
            },
            "people": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Filter by person names. AND.",
            },
            "date_from": {
                "type": "string",
                "description": "ISO date. From here onward.",
            },
            "date_to": {
                "type": "string",
                "description": "ISO date. Up to here.",
            },
            "source": {
                "type": "string",
                "description": "'whatsapp', 'camera', etc.",
            },
            "in_slideshow": {
                "type": "boolean",
                "description": "Filter by slideshow membership.",
            },
            "include_deleted": {
                "type": "boolean",
                "description": "Include soft-deleted. Default false.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results for 'search'. Default 20.",
            },
        },
        "required": ["mode"],
        "additionalProperties": False,
    }

    async def execute(self, **kwargs: Any) -> str:
        mode: str = kwargs["mode"]
        query: str | None = kwargs.get("query")
        photo_id: str | None = kwargs.get("photo_id")

        logger.info(
            "search_photos: mode=%s, query=%s, photo_id=%s",
            mode,
            query[:50] if query else None,
            photo_id,
        )

        try:
            return await self._search_via_backend(mode, kwargs)
        except ImportError:
            logger.warning("Photo search backend not available yet")
            return json.dumps({
                "error": (
                    "Photo search backend not available. "
                    "The photo system may not be set up yet."
                ),
            })
        except ValueError as e:
            return json.dumps({"error": str(e)})
        except Exception as e:
            logger.exception("search_photos error")
            return json.dumps({"error": f"Photo search failed: {e}"})

    async def _search_via_backend(
        self, mode: str, kwargs: dict[str, Any]
    ) -> str:
        """Route to the photo search backend (lazy import)."""
        # Lazy import — photo system may not be built yet
        from boxbot.photos.search import search_photos

        result = await search_photos(
            mode=mode,
            query=kwargs.get("query"),
            photo_id=kwargs.get("photo_id"),
            tags=kwargs.get("tags"),
            people=kwargs.get("people"),
            date_from=kwargs.get("date_from"),
            date_to=kwargs.get("date_to"),
            source=kwargs.get("source"),
            in_slideshow=kwargs.get("in_slideshow"),
            include_deleted=kwargs.get("include_deleted", False),
            limit=kwargs.get("limit", 20),
        )

        return json.dumps(result)
