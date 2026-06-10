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

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from boxbot.core.paths import PERCEPTION_DIR
from boxbot.perception.clouds import ANCHOR_PROVENANCES, CloudStore

logger = logging.getLogger(__name__)

# Latest audit report — persisted after every run so the agent can read
# the findings later (identify_person action="list_flags").
REPORT_PATH = PERCEPTION_DIR / "id-reconcile-latest.json"

# Defaults — middle-road, calibrate from audit reports over time.
DEFAULT_ISOLATION_THRESHOLD = 0.55   # flag if mean top-k neighbour cosine < 0.45
DEFAULT_DUP_NAME_MAX_DISTANCE = 1    # "Eric" vs "Erik"
DEFAULT_DUP_CENTROID_MIN_SIM = 0.80  # same face under two names
DEFAULT_MISLABEL_MIN_SIM = 0.75      # looks strongly like another's anchor
DEFAULT_MISLABEL_MARGIN = 0.10       # ... and clearly more than its own
_TOPK = 3

# --- Multimodal judge (NEXT-C) ---------------------------------------------
# Provenances eligible for the cluster-verify judge (weak, unconfirmed admits).
# Anchors + voice_visual_agree are trusted and skipped.
JUDGE_WEAK_PROVENANCES = frozenset(
    {"voice_doa", "visual_reid", "context_inferred", "legacy", "seed"}
)
DEFAULT_JUDGE_CLUSTER_SIM = 0.80     # greedy cluster threshold for daily admits
DEFAULT_JUDGE_MIN_CONFIDENCE = 0.85  # auto-apply gate
DEFAULT_JUDGE_MAX_CALLS = 20         # per-night budget cap (dup pairs + clusters)
_MAX_REFERENCE_CROPS = 3

_DUP_PERSON_TOOL = {
    "name": "duplicate_person_verdict",
    "description": "Decide whether two face crops are the same person.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["same_person", "different", "unsure"],
            },
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "confidence", "reasoning"],
    },
}

_CLUSTER_VERIFY_TOOL = {
    "name": "cluster_identity_verdict",
    "description": (
        "Decide whether a face crop belongs to the person it was filed under. "
        "If it is clearly a different person ON THE PROVIDED LIST, name them in "
        "suggested_person; otherwise leave suggested_person empty. Never invent "
        "a name that is not in the provided list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["belongs", "wrong_person", "unsure"],
            },
            "suggested_person": {
                "type": "string",
                "description": "An existing name from the provided list, or empty.",
            },
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["verdict", "confidence", "reasoning"],
    },
}


def _image_block(crop_path: str | None) -> dict | None:
    """Build an Anthropic base64 image content block from a crop path, or None."""
    if not crop_path:
        return None
    try:
        data = Path(crop_path).read_bytes()
    except OSError:
        return None
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(data).decode("ascii"),
        },
    }


async def _judge_call(
    client: Any, model: str, tool: dict, content: list[dict]
) -> dict | None:
    """Run one structured-output multimodal judge call; return tool input or None."""
    try:
        resp = await client.messages.create(
            model=model,
            max_tokens=600,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": content}],
        )
    except Exception:
        logger.exception("id-reconcile judge call failed")
        return None
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return None


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
    client: Any = None,
    model: str = "",
    auto_apply: bool = False,
    judge_min_confidence: float = DEFAULT_JUDGE_MIN_CONFIDENCE,
    judge_max_calls: int = DEFAULT_JUDGE_MAX_CALLS,
    judge_cluster_sim: float = DEFAULT_JUDGE_CLUSTER_SIM,
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Run the deterministic identity-cloud reconciliation pass.

    Opens its own CloudStore if one isn't injected (file-backed, WAL — safe to
    read alongside the live perception process). Returns an audit report;
    mutates nothing unless ``audit_only=False`` (eviction + relabel; person
    merge is never auto-applied here). The report is also persisted to
    ``report_path`` (default ``data/perception/id-reconcile-latest.json``)
    so the agent can read the latest findings via
    ``identify_person(action="list_flags")``.
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
            "applied": {"evicted": 0, "relabelled": 0, "confirmed": 0},
            "judge": None,
        }

        # Deterministic eviction of pure outliers (geometry is reliable here).
        # Relabels are NOT auto-applied deterministically — moving a face
        # between people needs eyes, so that happens only via the judge.
        if not audit_only:
            report["applied"]["evicted"] = await _evict_outliers(
                cloud_store, outliers
            )

        # Multimodal LLM judge (NEXT-C): adjudicates what geometry can't.
        if client is not None and model:
            report["judge"] = await _run_judge(
                cloud_store=cloud_store,
                client=client,
                model=model,
                records=records,
                name_by_id=name_by_id,
                duplicate_persons=duplicate_persons,
                audit_only=audit_only,
                auto_apply=auto_apply,
                min_confidence=judge_min_confidence,
                max_calls=judge_max_calls,
                cluster_sim=judge_cluster_sim,
                report=report,
            )

        logger.info(
            "id-reconcile (%s): %d persons, %d visual embeddings | "
            "%d outliers, %d duplicate-person candidate(s), %d mislabel(s) | "
            "applied evicted=%d relabelled=%d confirmed=%d | judge=%s",
            "audit" if audit_only else "apply",
            report["persons"], report["visual_embeddings"],
            len(outliers), len(duplicate_persons), len(mislabels),
            report["applied"]["evicted"], report["applied"]["relabelled"],
            report["applied"]["confirmed"],
            "off" if report["judge"] is None
            else f"{report['judge']['calls']} call(s)",
        )
        for dp in duplicate_persons:
            logger.info(
                "id-reconcile: possible duplicate persons %r ~ %r "
                "(name_distance=%d, centroid_sim=%.3f) — %s",
                dp["a"], dp["b"], dp["name_distance"],
                dp["centroid_sim"], dp["reason"],
            )
        _persist_report(report, report_path)
        return report
    finally:
        if own_store:
            await cloud_store.close()


def _persist_report(report: dict[str, Any], report_path: Path | None) -> None:
    """Write the audit report to disk (best-effort; never raises)."""
    path = report_path or REPORT_PATH
    try:
        report["ran_at"] = datetime.now(timezone.utc).isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))
    except Exception:
        logger.exception("Failed to persist id-reconcile report to %s", path)


def load_latest_report(report_path: Path | None = None) -> dict[str, Any] | None:
    """Read the most recent persisted reconcile report, or None.

    This is the read side of the flag surface: the agent consumes it via
    ``identify_person(action="list_flags")``.
    """
    path = report_path or REPORT_PATH
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("Failed to read id-reconcile report at %s", path)
        return None


def duplicate_todo_description(pair: dict[str, Any]) -> str:
    """One-line to-do description for a flagged duplicate-person pair.

    Stable for a given (a, b) pair so repeat nightly flags can be
    deduplicated against existing to-dos by exact description match.
    """
    a, b = sorted([str(pair.get("a")), str(pair.get("b"))])
    return (
        f"[id-reconcile] Possible duplicate people: '{a}' and '{b}' — "
        f"verify with the household, then identify_person(action=\"merge\") "
        f"if they are the same person"
    )


async def nudge_duplicate_todos(report: dict[str, Any]) -> int:
    """Create a to-do for each NEWLY flagged duplicate-person pair.

    "New" means no existing to-do (any status) already carries the
    pair's stable description — so a pair flagged every night until it's
    resolved produces exactly one nudge, and completing/cancelling the
    to-do suppresses re-nudging. The to-do surfaces through the
    ``[To-do: N items]`` status line, prompting the agent to investigate
    via ``identify_person(action="list_flags")``.

    Returns the number of to-dos created. Best-effort: scheduler
    failures are logged, not raised (this runs inside the dream cycle).
    """
    pairs = report.get("duplicate_persons") or []
    if not pairs:
        return 0
    try:
        from boxbot.core import scheduler

        existing = {
            t.get("description") for t in await scheduler.list_todos()
        }
        created = 0
        for pair in pairs:
            desc = duplicate_todo_description(pair)
            if desc in existing:
                continue
            await scheduler.create_todo(
                desc,
                notes=(
                    f"Flagged by the nightly identity reconcile "
                    f"(reason: {pair.get('reason')}, "
                    f"name_distance={pair.get('name_distance')}, "
                    f"centroid_sim={pair.get('centroid_sim')}). "
                    f"Read the full findings with "
                    f"identify_person(action=\"list_flags\"). Merging is "
                    f"destructive — confirm with the people involved "
                    f"before calling identify_person(action=\"merge\")."
                ),
                source="agent",
            )
            existing.add(desc)
            created += 1
        if created:
            logger.info(
                "id-reconcile: created %d duplicate-person to-do nudge(s)",
                created,
            )
        return created
    except Exception:
        logger.exception("id-reconcile: duplicate to-do nudge failed")
        return 0


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


async def _evict_outliers(
    cloud_store: CloudStore, outliers: list[dict]
) -> int:
    """Evict geometrically-isolated outliers (deterministic, non-destructive).

    Only reached when ``audit_only=False``. Relabels and person-merges are NOT
    done here — relabels require the multimodal judge (eyes), and person merges
    are flag-only (decision 2026-06-07).
    """
    evicted = 0
    for o in outliers:
        await cloud_store.delete_visual_embedding(o["embedding_id"])
        await cloud_store.log_correction(
            source="reconcile_outlier",
            source_ref=o["embedding_id"],
            from_person_id=o["person_id"],
            detail=f"isolation={o['isolation']} provenance={o['provenance']}",
        )
        evicted += 1
    return evicted


# ---------------------------------------------------------------------------
# Multimodal LLM judge (NEXT-C)
# ---------------------------------------------------------------------------


def _greedy_clusters(
    recs: list[dict], threshold: float
) -> list[list[dict]]:
    """Greedy single-pass clustering of records by cosine to a cluster seed."""
    clusters: list[list[dict]] = []
    seeds: list[np.ndarray] = []
    for r in recs:
        placed = False
        for i, seed in enumerate(seeds):
            if float(seed @ r["embedding"]) >= threshold:
                clusters[i].append(r)
                placed = True
                break
        if not placed:
            clusters.append([r])
            seeds.append(r["embedding"])
    return clusters


def _medoid(recs: list[dict]) -> dict:
    """Return the record most central to its cluster (max mean cosine)."""
    if len(recs) == 1:
        return recs[0]
    embs = np.stack([r["embedding"] for r in recs])
    sims = embs @ embs.T
    return recs[int(sims.mean(axis=1).argmax())]


async def _run_judge(
    *,
    cloud_store: CloudStore,
    client: Any,
    model: str,
    records: dict[str, list[dict]],
    name_by_id: dict[str, str],
    duplicate_persons: list[dict],
    audit_only: bool,
    auto_apply: bool,
    min_confidence: float,
    max_calls: int,
    cluster_sim: float,
    report: dict,
) -> dict[str, Any]:
    """Adjudicate ambiguous candidates with the multimodal model.

    Two streams:
    1. duplicate-person pairs → compare the two persons' representative crops
       (flag-only; person merge is never auto-applied).
    2. clustered unreconciled weak admits → verify each cluster's representative
       belongs to the person it was filed under, against that person's anchor
       crops. Auto-applies (confirm / relabel-to-existing / evict) when
       ``auto_apply`` and confidence ≥ gate and not ``audit_only``.
    """
    calls = 0
    dup_results: list[dict] = []
    cluster_results: list[dict] = []
    dropped = 0

    # Representative crop per person (medoid of their cropped points).
    person_rep_crop: dict[str, str] = {}
    anchor_crops: dict[str, list[str]] = {}
    for pid, recs in records.items():
        cropped = [r for r in recs if r["crop_path"]]
        if cropped:
            person_rep_crop[pid] = _medoid(cropped)["crop_path"]
        anchors = [
            r["crop_path"] for r in recs
            if r["provenance"] in ANCHOR_PROVENANCES and r["crop_path"]
        ]
        if anchors:
            anchor_crops[pid] = anchors[:_MAX_REFERENCE_CROPS]

    # --- Stream 1: duplicate-person confirmation (flag-only) ---------------
    for dp in duplicate_persons:
        if calls >= max_calls:
            dropped += 1
            continue
        ca = _image_block(person_rep_crop.get(dp["a_id"]))
        cb = _image_block(person_rep_crop.get(dp["b_id"]))
        if ca is None or cb is None:
            dup_results.append({**dp, "judge": "no_crops"})
            continue
        content = [
            {"type": "text", "text": (
                f"Two stored people may be the same person. Image 1 is filed "
                f"as {dp['a']!r}; image 2 as {dp['b']!r}. Are these the same "
                f"person? Return duplicate_person_verdict."
            )},
            ca, cb,
        ]
        calls += 1
        v = await _judge_call(client, model, _DUP_PERSON_TOOL, content)
        dup_results.append({**dp, "verdict": v})
        # Person merges are flag-only — record, never auto-merge.

    # --- Stream 2: cluster-verify of unreconciled weak admits -------------
    for pid, recs in records.items():
        refs = [_image_block(p) for p in anchor_crops.get(pid, [])]
        refs = [r for r in refs if r is not None]
        if not refs:
            continue  # no reference faces for this person yet → can't verify
        weak = [
            r for r in recs
            if not r["reconciled"]
            and r["provenance"] in JUDGE_WEAK_PROVENANCES
            and r["crop_path"]
        ]
        if not weak:
            continue
        for cluster in _greedy_clusters(weak, cluster_sim):
            if calls >= max_calls:
                dropped += 1
                continue
            rep = _medoid(cluster)
            rep_img = _image_block(rep["crop_path"])
            if rep_img is None:
                continue
            others = sorted({
                n for i, n in name_by_id.items() if i != pid
            })
            content = [
                {"type": "text", "text": (
                    f"The first {len(refs)} image(s) are confirmed photos of "
                    f"{name_by_id.get(pid)!r}. The LAST image was auto-filed "
                    f"under {name_by_id.get(pid)!r} but is unconfirmed. Does the "
                    f"last image show {name_by_id.get(pid)!r}? If it is clearly "
                    f"someone else on this list, name them: {others}. "
                    f"Return cluster_identity_verdict."
                )},
                *refs, rep_img,
            ]
            calls += 1
            v = await _judge_call(client, model, _CLUSTER_VERIFY_TOOL, content)
            member_ids = [c["id"] for c in cluster]
            applied = await _apply_cluster_verdict(
                cloud_store, pid, member_ids, v, name_by_id,
                audit_only=audit_only, auto_apply=auto_apply,
                min_confidence=min_confidence, report=report,
            )
            cluster_results.append({
                "person": name_by_id.get(pid),
                "person_id": pid,
                "cluster_size": len(cluster),
                "verdict": v,
                "applied": applied,
            })

    if dropped:
        logger.info(
            "id-reconcile judge: budget cap hit, %d candidate group(s) dropped",
            dropped,
        )
    return {
        "calls": calls,
        "dropped": dropped,
        "duplicate_persons": dup_results,
        "clusters": cluster_results,
    }


async def _apply_cluster_verdict(
    cloud_store: CloudStore,
    person_id: str,
    member_ids: list[str],
    verdict: dict | None,
    name_by_id: dict[str, str],
    *,
    audit_only: bool,
    auto_apply: bool,
    min_confidence: float,
    report: dict,
) -> str:
    """Apply a cluster-verify verdict (confirm / relabel / evict). Returns action.

    No-op (returns 'audit') when audit_only, auto_apply off, verdict missing,
    or confidence below the gate.
    """
    if verdict is None:
        return "no_verdict"
    decision = verdict.get("verdict")
    confidence = float(verdict.get("confidence") or 0.0)
    if audit_only or not auto_apply or confidence < min_confidence:
        return "audit"

    name_to_id = {n: i for i, n in name_by_id.items()}

    if decision == "belongs":
        await cloud_store.mark_visual_reconciled(member_ids)
        report["applied"]["confirmed"] += len(member_ids)
        return "confirmed"

    if decision == "wrong_person":
        suggested = (verdict.get("suggested_person") or "").strip()
        target_id = name_to_id.get(suggested)
        if target_id and target_id != person_id:
            for mid in member_ids:
                await cloud_store.reassign_visual_embedding(
                    mid, target_id, "context_inferred"
                )
            await cloud_store.log_correction(
                source="reconcile_judge_relabel",
                from_person_id=person_id, to_person_id=target_id,
                detail=f"{len(member_ids)} embedding(s) relabelled",
            )
            report["applied"]["relabelled"] += len(member_ids)
            return "relabelled"
        # Wrong person but not anyone on file → evict (never create a person).
        for mid in member_ids:
            await cloud_store.delete_visual_embedding(mid)
        await cloud_store.log_correction(
            source="reconcile_judge_evict",
            from_person_id=person_id,
            detail=f"{len(member_ids)} embedding(s) evicted (wrong, unknown)",
        )
        report["applied"]["evicted"] += len(member_ids)
        return "evicted"

    return "unsure"
