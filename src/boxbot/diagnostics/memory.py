"""Memory diagnostics — RSS sampling, live-object accounting, tracemalloc.

The system monitor calls :func:`snapshot` on every tick to log a one-line
summary. When the leak under investigation has been pinned down, the
feature can be turned off via ``config.diagnostics.memory.enabled``.

No new runtime deps: RSS/VSZ come from ``/proc/self/status`` and live
object counts come from already-instantiated singletons (agent's
``_conversations`` dict, the photo intake pipeline). The agent is
registered at boot via :func:`set_agent_ref`.
"""

from __future__ import annotations

import ctypes
import json
import logging
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from boxbot.core.agent import BoxBotAgent

logger = logging.getLogger(__name__)


# ── glibc malloc introspection (ctypes) ───────────────────────────────
# These talk to the C runtime directly. The leak we're chasing is native
# (every Python-level subsystem tested clean in isolation, yet the live
# process retains ~700 MB after a conversation), which points at glibc
# heap fragmentation: freed blocks that malloc holds in per-arena free
# lists instead of returning to the OS. mallinfo2 exposes exactly that.

try:
    _libc: ctypes.CDLL | None = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    _libc = None


class _MallInfo2(ctypes.Structure):
    """glibc struct mallinfo2 (glibc >= 2.33; size_t fields, no overflow)."""

    _fields_ = [
        ("arena", ctypes.c_size_t),     # non-mmapped space allocated (heap)
        ("ordblks", ctypes.c_size_t),   # number of free chunks
        ("smblks", ctypes.c_size_t),    # number of fastbin blocks
        ("hblks", ctypes.c_size_t),     # number of mmapped regions
        ("hblkhd", ctypes.c_size_t),    # space in mmapped regions
        ("usmblks", ctypes.c_size_t),   # always 0 (legacy)
        ("fsmblks", ctypes.c_size_t),   # space in fastbin free blocks
        ("uordblks", ctypes.c_size_t),  # total allocated space (in use)
        ("fordblks", ctypes.c_size_t),  # total free space (held, not returned)
        ("keepcost", ctypes.c_size_t),  # releasable (top-most) free space
    ]


@dataclass(frozen=True)
class MallocStats:
    arena_mb: float       # heap (non-mmapped) reserved by malloc
    in_use_mb: float      # actually allocated and live (uordblks)
    free_held_mb: float   # freed but NOT returned to OS (fordblks) — fragmentation
    mmap_mb: float        # large allocations via mmap (hblkhd)


def read_malloc_stats() -> MallocStats | None:
    """Read glibc mallinfo2. None if unavailable (non-glibc / old glibc)."""
    if _libc is None or not hasattr(_libc, "mallinfo2"):
        return None
    try:
        _libc.mallinfo2.restype = _MallInfo2
        mi = _libc.mallinfo2()
    except Exception:
        return None
    return MallocStats(
        arena_mb=mi.arena / 1024.0 / 1024.0,
        in_use_mb=mi.uordblks / 1024.0 / 1024.0,
        free_held_mb=mi.fordblks / 1024.0 / 1024.0,
        mmap_mb=mi.hblkhd / 1024.0 / 1024.0,
    )


def malloc_trim() -> bool:
    """Ask glibc to return free heap pages at the top of each arena to the OS.

    Returns True if memory was actually released. Cheap to call; the
    counterpart to MALLOC_ARENA_MAX — the arena cap limits how badly the
    heap fragments, malloc_trim hands back what's already free.
    """
    if _libc is None or not hasattr(_libc, "malloc_trim"):
        return False
    try:
        return bool(_libc.malloc_trim(0))
    except Exception:
        return False


# ── Agent reference ────────────────────────────────────────────────────
# Set once at boot by ``boxbot.core.main`` after the agent is constructed.
# Diagnostics reads through this so the system module never has to import
# the agent (avoids a circular dependency).

_agent_ref: "BoxBotAgent | None" = None


def set_agent_ref(agent: "BoxBotAgent") -> None:
    """Register the live agent so diagnostics can introspect conversations."""
    global _agent_ref
    _agent_ref = agent


def get_agent_ref() -> "BoxBotAgent | None":
    return _agent_ref


# ── Process-level RSS / VSZ ───────────────────────────────────────────


@dataclass(frozen=True)
class ProcessMemory:
    rss_mb: float
    vsz_mb: float
    swap_mb: float


def read_process_memory() -> ProcessMemory | None:
    """Read RSS/VSZ/swap from /proc/self/status. Linux-only."""
    try:
        with open("/proc/self/status") as f:
            info: dict[str, int] = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].endswith(":"):
                    key = parts[0][:-1]
                    try:
                        info[key] = int(parts[1])
                    except ValueError:
                        continue
    except (FileNotFoundError, OSError):
        return None

    # values are kB
    return ProcessMemory(
        rss_mb=info.get("VmRSS", 0) / 1024.0,
        vsz_mb=info.get("VmSize", 0) / 1024.0,
        swap_mb=info.get("VmSwap", 0) / 1024.0,
    )


# ── Full snapshot ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class MemorySnapshot:
    rss_mb: float
    vsz_mb: float
    swap_mb: float
    conv_count: int
    conv_thread_bytes_total: int
    conv_thread_msgs_total: int
    largest_conv: tuple[str, int, int] | None  # (conv_id, msgs, bytes)
    sandbox_runners_live: int
    photo_intake_depth: int
    malloc: MallocStats | None = None


def snapshot(include_malloc: bool = True) -> MemorySnapshot | None:
    """Take one diagnostics snapshot. Returns None if /proc is unavailable."""
    proc = read_process_memory()
    if proc is None:
        return None

    conv_count = 0
    thread_bytes_total = 0
    thread_msgs_total = 0
    largest: tuple[str, int, int] | None = None
    sandbox_live = 0

    agent = get_agent_ref()
    if agent is not None:
        convs = getattr(agent, "_conversations", {}) or {}
        conv_count = len(convs)
        for conv_id, conv in convs.items():
            thread = getattr(conv, "_thread", None)
            if thread is None:
                continue
            try:
                serialized = json.dumps(thread, default=str)
                size = len(serialized)
            except (TypeError, ValueError):
                # Fallback if thread carries non-JSONable bytes
                size = sum(len(repr(m)) for m in thread)
            n_msgs = len(thread)
            thread_bytes_total += size
            thread_msgs_total += n_msgs
            if largest is None or size > largest[2]:
                largest = (conv_id, n_msgs, size)
            if getattr(conv, "sandbox_runner", None) is not None:
                sandbox_live += 1

    photo_depth = 0
    try:
        from boxbot.photos.intake import get_intake_pipeline

        pipeline = get_intake_pipeline()
        if pipeline is not None:
            queue = getattr(pipeline, "_queue", None)
            if queue is not None:
                photo_depth = queue.qsize()
    except Exception:
        # Diagnostics must never raise — best-effort only.
        pass

    return MemorySnapshot(
        rss_mb=proc.rss_mb,
        vsz_mb=proc.vsz_mb,
        swap_mb=proc.swap_mb,
        conv_count=conv_count,
        conv_thread_bytes_total=thread_bytes_total,
        conv_thread_msgs_total=thread_msgs_total,
        largest_conv=largest,
        sandbox_runners_live=sandbox_live,
        photo_intake_depth=photo_depth,
        malloc=read_malloc_stats() if include_malloc else None,
    )


def format_snapshot(snap: MemorySnapshot) -> str:
    """Render a snapshot as a single log line (compact key=value form)."""
    largest = snap.largest_conv
    largest_part = (
        f" largest_conv={largest[0]}:{largest[1]}msg/{largest[2]/1024:.0f}KB"
        if largest is not None
        else ""
    )
    m = snap.malloc
    malloc_part = (
        f" malloc_arena={m.arena_mb:.0f}MB "
        f"malloc_inuse={m.in_use_mb:.0f}MB "
        f"malloc_free_held={m.free_held_mb:.0f}MB "
        f"malloc_mmap={m.mmap_mb:.0f}MB"
        if m is not None
        else ""
    )
    return (
        f"rss={snap.rss_mb:.0f}MB "
        f"vsz={snap.vsz_mb:.0f}MB "
        f"swap={snap.swap_mb:.0f}MB "
        f"convs={snap.conv_count} "
        f"thread_msgs={snap.conv_thread_msgs_total} "
        f"thread_bytes={snap.conv_thread_bytes_total/1024:.0f}KB "
        f"sandbox_runners={snap.sandbox_runners_live} "
        f"photo_intake_q={snap.photo_intake_depth}"
        f"{malloc_part}"
        f"{largest_part}"
    )


# ── tracemalloc ────────────────────────────────────────────────────────


_tracemalloc_started = False
_last_tracemalloc: tracemalloc.Snapshot | None = None


def enable_tracemalloc(nframe: int = 10) -> None:
    """Start tracing allocations. Safe to call multiple times."""
    global _tracemalloc_started
    if _tracemalloc_started:
        return
    tracemalloc.start(nframe)
    _tracemalloc_started = True
    logger.info("tracemalloc enabled (nframe=%d)", nframe)


def log_tracemalloc_top(top_n: int = 20) -> None:
    """Log top-N allocators by current size, plus diff vs previous call."""
    global _last_tracemalloc
    if not _tracemalloc_started:
        return

    current = tracemalloc.take_snapshot()
    # Strip frames inside tracemalloc / linecache to keep the report tidy.
    current = current.filter_traces((
        tracemalloc.Filter(False, tracemalloc.__file__),
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<frozen importlib._bootstrap_external>"),
    ))

    top_stats = current.statistics("lineno")[:top_n]
    logger.info("tracemalloc top %d by current size:", top_n)
    for i, stat in enumerate(top_stats, 1):
        logger.info(
            "  #%2d  %.1f KB  count=%d  %s",
            i,
            stat.size / 1024.0,
            stat.count,
            stat.traceback.format()[0] if stat.traceback else "<no traceback>",
        )

    if _last_tracemalloc is not None:
        diff = current.compare_to(_last_tracemalloc, "lineno")[:top_n]
        logger.info("tracemalloc top %d by growth since last snapshot:", top_n)
        for i, stat in enumerate(diff, 1):
            if stat.size_diff == 0 and stat.count_diff == 0:
                continue
            logger.info(
                "  #%2d  %+.1f KB (count %+d, now %.1f KB)  %s",
                i,
                stat.size_diff / 1024.0,
                stat.count_diff,
                stat.size / 1024.0,
                stat.traceback.format()[0] if stat.traceback else "<no traceback>",
            )

    _last_tracemalloc = current


# ── Snapshot + log helper ──────────────────────────────────────────────


def log_snapshot() -> MemorySnapshot | None:
    """Take a snapshot and log it. Returns the snapshot for caller checks."""
    snap = snapshot()
    if snap is None:
        return None
    logger.info("memory: %s", format_snapshot(snap))
    return snap
