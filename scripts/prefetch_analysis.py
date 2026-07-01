#!/usr/bin/env python3
"""Offline analysis harness for the prefetch layer (read-only).

Joins ``prefetch_events`` to ``tool_invocations`` (bridging scheduled
triggers via ``prefetch_cache.conversation_id``) and reports the signals
that tune the prefetcher and gate shadow -> active:

  * turns-to-first-message      (how fast the agent got to a reply)
  * tool-calls per conversation (bloat proxy)
  * repeated-search rate        (wasted re-fetches)
  * skill precision / recall    (EXACT — load_skill carries the name)
  * memory / workspace re-search signal
  * bundle token cost

Run on a dev box against a COPY of the Pi's memory.db:

    scp pi:software/boxBot/data/memory/memory.db /tmp/bb.db
    python3 scripts/prefetch_analysis.py --db /tmp/bb.db

NOTE on the honest limit: ``tool_invocations`` logs tool NAMES + INPUTS,
not RESULTS. So memory-id-level hit-rate ("did the agent's live search
surface exactly the memory we prefetched?") is NOT computable from the
current schema — only skill-level precision (load_skill names) is exact.
Memory/workspace are reported as re-search *rates* (did the agent still
search after we prefetched that category?). To get id-level memory
hit-rate later, log a compact result summary alongside each invocation.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

_SEARCH_TOOLS = {
    "search_memory", "load_skill", "workspace_search",
    "workspace_read", "search_photos", "search_history",
}


def _base(name: str) -> str:
    """Strip the SDK's ``mcp__<server>__`` prefix; bare names pass through."""
    if name.startswith("mcp__") and "__" in name[5:]:
        return name.split("__")[-1]
    return name


def _json_list(raw: str | None) -> list[Any]:
    if not raw:
        return []
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else []
    except Exception:
        return []


def _skill_name(raw_input: str | None) -> str | None:
    if not raw_input:
        return None
    try:
        return json.loads(raw_input).get("name")
    except Exception:
        return None


def _load_events(cur: sqlite3.Cursor) -> list[dict[str, Any]]:
    cur.execute(
        """SELECT key, key_kind, channel, mode, predicted_memory_ids,
                  predicted_skills, predicted_workspace_paths,
                  bundle_token_estimate
           FROM prefetch_events"""
    )
    return [
        {
            "key": r[0], "key_kind": r[1], "channel": r[2], "mode": r[3],
            "memory_ids": _json_list(r[4]),
            "skills": _json_list(r[5]),
            "workspace_paths": _json_list(r[6]),
            "token_estimate": r[7] or 0,
        }
        for r in cur.fetchall()
    ]


def _resolve_conversation_id(cur: sqlite3.Cursor, ev: dict[str, Any]) -> str | None:
    if ev["key_kind"] == "conversation":
        return ev["key"]
    # trigger: bridge via prefetch_cache
    cur.execute(
        "SELECT conversation_id FROM prefetch_cache WHERE trigger_id = ?",
        (ev["key"],),
    )
    row = cur.fetchone()
    return row[0] if row and row[0] else None


def _conversation_calls(cur: sqlite3.Cursor, conv_id: str) -> list[dict[str, Any]]:
    cur.execute(
        """SELECT tool_name, tool_input_redacted, turn_number
           FROM tool_invocations WHERE conversation_id = ?
           ORDER BY turn_number, id""",
        (conv_id,),
    )
    return [
        {"name": _base(r[0]), "input": r[1], "turn": r[2]}
        for r in cur.fetchall()
    ]


def _analyze_conversation(calls: list[dict[str, Any]]) -> dict[str, Any]:
    seen: set[tuple[str, str]] = set()
    repeated = total_search = 0
    loaded_skills: set[str] = set()
    searched_memory = 0
    first_message_turn: int | None = None
    for c in calls:
        name = c["name"]
        if name == "message" and first_message_turn is None:
            first_message_turn = c["turn"]
        if name in _SEARCH_TOOLS:
            total_search += 1
            sig = (name, c["input"] or "")
            if sig in seen:
                repeated += 1
            seen.add(sig)
        if name == "load_skill":
            sk = _skill_name(c["input"])
            if sk:
                loaded_skills.add(sk)
        if name == "search_memory":
            searched_memory += 1
    return {
        "total_calls": len(calls),
        "total_search": total_search,
        "repeated": repeated,
        "loaded_skills": loaded_skills,
        "searched_memory": searched_memory,
        "first_message_turn": first_message_turn,
    }


def _report(bucket: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        print(f"\n[{bucket}] no conversations with matched telemetry")
        return
    n = len(rows)

    def avg(key: str) -> float:
        vals = [r[key] for r in rows if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    total_search = sum(r["total_search"] for r in rows)
    repeated = sum(r["repeated"] for r in rows)

    # Skill precision/recall (exact): predicted vs actually load_skill'd.
    pred_skill_total = inter_total = loaded_total = 0
    for r in rows:
        pred = set(r["predicted_skills"])
        loaded = r["loaded_skills"]
        inter = pred & loaded
        pred_skill_total += len(pred)
        loaded_total += len(loaded)
        inter_total += len(inter)

    print(f"\n[{bucket}] conversations analyzed: {n}")
    print(f"  tool calls / conv (avg):        {avg('total_calls'):.2f}")
    print(f"  turns-to-first-message (avg):   {avg('first_message_turn'):.2f}")
    print(f"  repeated-search rate:           "
          f"{(repeated / total_search * 100) if total_search else 0:.1f}% "
          f"({repeated}/{total_search})")
    print(f"  memory searches / conv (avg):   {avg('searched_memory'):.2f}")
    print(f"  bundle tokens / conv (avg):     {avg('token_estimate'):.1f}")
    if pred_skill_total:
        print(f"  skill precision (pred used):    "
              f"{inter_total / pred_skill_total * 100:.1f}% "
              f"({inter_total}/{pred_skill_total})")
    if loaded_total:
        print(f"  skill recall (foresaw loads):   "
              f"{inter_total / loaded_total * 100:.1f}% "
              f"({inter_total}/{loaded_total})")
    print("  interpretation:")
    print("    shadow → precision/recall vs what the agent actually did.")
    print("    active → LOW subsequent memory-search / skill-load of predicted")
    print("            items means the bundle SATISFIED the need (a win).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--db", default="data/memory/memory.db",
        help="Path to memory.db (use a copy of the Pi's).",
    )
    args = ap.parse_args()
    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"db not found: {db_path}")

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    events = _load_events(cur)
    if not events:
        print("No prefetch_events rows yet — run in shadow mode first.")
        return

    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unmatched = 0
    for ev in events:
        conv_id = _resolve_conversation_id(cur, ev)
        if not conv_id:
            unmatched += 1
            continue
        calls = _conversation_calls(cur, conv_id)
        if not calls:
            unmatched += 1
            continue
        row = _analyze_conversation(calls)
        row["predicted_skills"] = ev["skills"]
        row["predicted_memory_ids"] = ev["memory_ids"]
        row["token_estimate"] = ev["token_estimate"]
        by_mode[ev["mode"]].append(row)

    print("=" * 64)
    print("Prefetch analysis")
    print(f"  prefetch_events: {len(events)}  unmatched (no telemetry): {unmatched}")
    for mode in ("shadow", "active"):
        _report(mode, by_mode.get(mode, []))
    conn.close()


if __name__ == "__main__":
    main()
