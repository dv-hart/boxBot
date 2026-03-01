"""Photo management subsystem.

Ingesting, tagging, searching, and displaying the household photo library.
Photos arrive via WhatsApp (or camera capture), pass through an async intake
pipeline, and are stored locally with rich metadata for search and slideshow.

Public API:
    PhotoStore — persistence, CRUD, tags, slideshow, storage quota
    IntakePipeline — async intake with resize, detection, tagging
    hybrid_search — shared search backend (vector + BM25)
    get_photo_detail — full photo record by ID
"""

from boxbot.photos.intake import IntakePipeline
from boxbot.photos.search import SearchResult, get_photo_detail, hybrid_search
from boxbot.photos.store import (
    PhotoRecord,
    PhotoStore,
    StorageInfo,
    TagRecord,
)

__all__ = [
    "IntakePipeline",
    "PhotoRecord",
    "PhotoStore",
    "SearchResult",
    "StorageInfo",
    "TagRecord",
    "get_photo_detail",
    "hybrid_search",
]
