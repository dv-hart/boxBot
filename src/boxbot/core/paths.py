"""Project root + canonical data directory locations.

All persistent runtime data is anchored to the project root, derived
from this file's location, so the same paths resolve regardless of
the process's current working directory.

Override the data root with ``BOXBOT_DATA_DIR`` (e.g. for tests or
alternate deployments).
"""

from __future__ import annotations

import os
from pathlib import Path

# This file lives at <repo>/src/boxbot/core/paths.py — four parents up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]

_data_override = os.environ.get("BOXBOT_DATA_DIR")
DATA_DIR = (
    Path(_data_override).expanduser().resolve()
    if _data_override
    else PROJECT_ROOT / "data"
)

PERCEPTION_DIR = DATA_DIR / "perception"
PERCEPTION_CROPS_DIR = PERCEPTION_DIR / "crops"
PERCEPTION_MODELS_DIR = PERCEPTION_DIR / "models"

MEMORY_DIR = DATA_DIR / "memory"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
SCHEDULER_DIR = DATA_DIR / "scheduler"
WORKSPACE_DIR = DATA_DIR / "workspace"
PHOTOS_DIR = DATA_DIR / "photos"
DISPLAYS_DIR = DATA_DIR / "displays"
CREDENTIALS_DIR = DATA_DIR / "credentials"
AUTH_DIR = DATA_DIR / "auth"
SANDBOX_DIR = DATA_DIR / "sandbox"
PREVIEWS_DIR = DATA_DIR / "previews"
CALIBRATION_DIR = DATA_DIR / "calibration"
INTEGRATIONS_DIR = DATA_DIR / "integrations"
