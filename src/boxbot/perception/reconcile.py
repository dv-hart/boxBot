"""Nightly identity-cloud reconciliation (dream-cycle hygiene).

The NOW phase (see docs/plans/person-id-overhaul.md) admits visual embeddings
liberally — voice teaches vision, provenance-tagged. Liberal admission is only
safe with active maintenance, which is this module's job. It runs in the nightly
dream window (folded into the ``[dream-cycle]`` trigger) and, like the memory
dream, **defaults to audit-only**: it produces a report of what it *would* do
without mutating anything until we trust it.

Three deterministic jobs (no model calls — the LLM judge layer is NEXT-C):

1. **Outlier detection** — per person, score each non-anchor embedding by
   isolation (1 − mean top-k cosine to its neighbours). Contaminated /
   mis-attributed faces are lonely → high isolation. Reported (evicted only
   when ``audit_only=False``).
2. **Duplicate-person candidates** — pairs of person records that are likely
   the same human: small name edit-distance (e.g. "Eric"/"Erik") and/or high
   cloud-centroid similarity. Reported for merge (always flag-only here;
   merging is destructive — decision 2026-06-07 keeps it audit-only).
3. **Mislabel candidates** — non-anchor embeddings whose nearest
   ``agent_identify`` anchor across *all* people belongs to a different person
   (with margin). The retroactive "this point looks more like X" signal.
   Reported for relabel (applied only when ``audit_only=False``).

Anchors (``agent_identify``) are ground truth: never evicted, never relabelled,
and the only points mislabel-scoring compares against.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from boxbot.perception.clouds import ANCHOR_PROVENANCES, CloudStore

logger = logging.getLogger(__name__)

# Defaults — middle-road, calibrate from audit reports over time.
DEFAULT_ISOLATION_THRESHOLD = 0.55   # flag if mean top-k neighbour cosine < 0.45
DEFAULT_DUP_NAME_MAX_DISTANCE = 1    # "Eric" vs "Erik"
DEFAULT_DUP_CENTROID_MIN_SIM = 0.80  # same face under two names
DEFAULT_MISLABEL_MIN_SIM = 0.75      # looks strongly like another's anchor
DEFAULT_MISLABEL_MARGIN = 0.10       # ... and clearly more than its own
_TOPK = 3


def _levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings (case-insensitive, small inputs)."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def _centroid(embs: np.ndarray) -> np.ndarray:
    """L2-normalized mean of a (N, D) embedding stack."""
    mean = embs.mean(axis=0).astype(np.float32)
    n = np.linalg.norm(mean)
    return mean / n if n > 0 else mean


def _isolation_scores(embs: np.ndarray, topk: int = _TOPK) -> np.ndarray:
    """Per-row isolation = 1 − mean(top-k cosine to other rows). Higher = lonelier."""
    n = embs.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)
    sims = embs @ embs.T
    np.fill_diagonal(sims, -1.0)
    k = min(topk, n - 1)
    neighbour = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    return (1.0 - neighbour).astype(np.float32)


async def run_id_reconcile(
    *,
    cloud_store: CloudStore | None = None,
    audit_only: bool = True,
    isolation_threshold: float = DEFAULT_ISOLATION_THRESHOLD,
    dup_name_max_distance: int = DEFAULT_DUP_NAME_MAX_DISTANCE,
    dup_centroid_min_sim: float = DEFAULT_DUP_CENTROID_MIN_SIM,
    mislabel_min_sim: float = DEFAULT_MISLABEL_MIN_SIM,
    mislabel_margin: float = DEFAULT_MISLABEL_MARGIN,
) -> dict[str, Any]:
    """Run the deterministic identity-cloud reconciliation pass.

    Opens its own CloudStore if one isn't injected (file-backed, WAL — safe to
    read alongside the live perception process). Returns an audit report;
    mutates nothing unless ``audit_only=False`` (eviction + relabel; person
    merge is never auto-applied here).
    """
    own_store = cloud_store is None
    if own_store:
        cloud_store = CloudStore()
        await cloud_store.initialize()

    try:
        persons = await cloud_store.list_persons()
        # Per-person records + centroid of anchor points (ground truth).
        records: dict[str, list[dict]] = {}
        anchor_centroid: dict[str, np.ndarray] = {}
        cloud_centroid: dict[str, np.ndarray] = {}
        name_by_id = {p["id"]: p["name"] for p in persons}

        total_visual = 0
        for p in persons:
            pid = p["id"]
            recs = await cloud_store.get_visual_records(pid)
            records[pid] = recs
            total_visual += len(recs)
            if recs:
                cloud_centroid[pid] = _centroid(
                    np.stack([r["embedding"] for r in recs])
                )
            anchors = [
                r["embedding"] for r in recs
                if r["provenance"] in ANCHOR_PROVENANCES
            ]
            if anchors:
                anchor_centroid[pid] = _centroid(np.stack(anchors))

        outliers = _find_outliers(records, name_by_id, isolation_threshold)
        duplicate_persons = _find_duplicate_persons(
            persons, cloud_centroid, dup_name_max_distance, dup_centroid_min_sim,
        )
        mislabels = _find_mislabels(
            records, name_by_id, anchor_centroid,
            mislabel_min_sim, mislabel_margin,
        )

        report: dict[str, Any] = {
            "audit_only": audit_only,
            "persons": len(persons),
            "visual_embeddings": total_visual,
            "anchored_persons": len(anchor_centroid),
            "outliers": outliers,
            "duplicate_persons": duplicate_persons,
            "mislabels": mislabels,
            "applied": {"evicted": 0, "relabelled": 0},
        }

        if not audit_only:
            report["applied"] = await _apply(cloud_store, outliers, mislabels)

        logger.info(
            "id-reconcile (%s): %d persons, %d visual embeddings | "
            "%d outliers, %d duplicate-person candidate(s), %d mislabel(s)%s",
            "audit" if audit_only else "apply",
            report["persons"], report["visual_embeddings"],
            len(outliers), len(duplicate_persons), len(mislabels),
            "" if audit_only else
            f" | evicted={report['applied']['evicted']} "
            f"relabelled={report['applied']['relabelled']}",
        )
        for dp in duplicate_persons:
            logger.info(
                "id-reconcile: possible duplicate persons %r ~ %r "
                "(name_distance=%d, centroid_sim=%.3f) — %s",
                dp["a"], dp["b"], dp["name_distance"],
                dp["centroid_sim"], dp["reason"],
            )
        return report
    finally:
        if own_store:
            await cloud_store.close()


def _find_outliers(
    records: dict[str, list[dict]],
    name_by_id: dict[str, str],
    isolation_threshold: float,
) -> list[dict]:
    """Flag isolated non-anchor embeddings (likely contamination)."""
    out: list[dict] = []
    for pid, recs in records.items():
        if len(recs) < 3:
            continue  # too few to judge isolation meaningfully
        embs = np.stack([r["embedding"] for r in recs])
        iso = _isolation_scores(embs)
        for r, score in zip(recs, iso):
            if r["provenance"] in ANCHOR_PROVENANCES:
                continue
            if score >= isolation_threshold:
                out.append({
                    "person": name_by_id.get(pid),
                    "person_id": pid,
                    "embedding_id": r["id"],
                    "provenance": r["provenance"],
                    "isolation": round(float(score), 3),
                })
    return out


def _find_duplicate_persons(
    persons: list[dict],
    cloud_centroid: dict[str, np.ndarray],
    name_max_distance: int,
    centroid_min_sim: float,
) -> list[dict]:
    """Flag person-record pairs that are probably the same human."""
    out: list[dict] = []
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            a, b = persons[i], persons[j]
            name_dist = _levenshtein(a["name"], b["name"])
            ca = cloud_centroid.get(a["id"])
            cb = cloud_centroid.get(b["id"])
            sim = float(ca @ cb) if ca is not None and cb is not None else 0.0
            name_hit = name_dist <= name_max_distance
            face_hit = sim >= centroid_min_sim
            if not (name_hit or face_hit):
                continue
            reason = ", ".join(filter(None, [
                "similar names" if name_hit else "",
                "similar faces" if face_hit else "",
            ]))
            out.append({
                "a": a["name"], "a_id": a["id"],
                "b": b["name"], "b_id": b["id"],
                "name_distance": name_dist,
                "centroid_sim": round(sim, 3),
                "reason": reason,
            })
    return out


def _find_mislabels(
    records: dict[str, list[dict]],
    name_by_id: dict[str, str],
    anchor_centroid: dict[str, np.ndarray],
    min_sim: float,
    margin: float,
) -> list[dict]:
    """Flag non-anchor points closer to another person's anchor than their own."""
    out: list[dict] = []
    if len(anchor_centroid) < 1:
        return out
    anchor_ids = list(anchor_centroid.keys())
    anchor_mat = np.stack([anchor_centroid[pid] for pid in anchor_ids])
    for pid, recs in records.items():
        for r in recs:
            if r["provenance"] in ANCHOR_PROVENANCES:
                continue
            sims = anchor_mat @ r["embedding"]
            order = np.argsort(sims)[::-1]
            best_pid = anchor_ids[order[0]]
            best_sim = float(sims[order[0]])
            own_sim = float(sims[anchor_ids.index(pid)]) if pid in anchor_ids else -1.0
            if (
                best_pid != pid
                and best_sim >= min_sim
                and (best_sim - own_sim) >= margin
            ):
                out.append({
                    "embedding_id": r["id"],
                    "current_person": name_by_id.get(pid),
                    "current_person_id": pid,
                    "suggested_person": name_by_id.get(best_pid),
                    "suggested_person_id": best_pid,
                    "anchor_sim": round(best_sim, 3),
                    "margin": round(best_sim - own_sim, 3),
                })
    return out


async def _apply(
    cloud_store: CloudStore,
    outliers: list[dict],
    mislabels: list[dict],
) -> dict[str, int]:
    """Apply non-destructive corrections (eviction + relabel). NOT person merge.

    Only reached when ``audit_only=False``. Person-record merges are never
    auto-applied (decision 2026-06-07) — they stay flag-only in the report.
    """
    db = cloud_store._ensure_db()
    evicted = 0
    for o in outliers:
        await db.execute(
            "DELETE FROM visual_embeddings WHERE id = ?", (o["embedding_id"],)
        )
        evicted += 1
    relabelled = 0
    for m in mislabels:
        await db.execute(
            "UPDATE visual_embeddings SET person_id = ?, provenance = ? "
            "WHERE id = ?",
            (m["suggested_person_id"], "context_inferred", m["embedding_id"]),
        )
        relabelled += 1
    await db.commit()
    return {"evicted": evicted, "relabelled": relabelled}
