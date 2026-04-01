from __future__ import annotations

import json
import os
import platform
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def _utc_timestamp_compact() -> str:
    # e.g. 20260127T045512Z
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _truncate(value: Any, limit: int = 200000) -> str:
    s = "" if value is None else str(value)
    if len(s) <= limit:
        return s
    return s[:limit] + f"...(truncated, {len(s)} chars total)"


def _is_jsonable(obj: Any) -> bool:
    try:
        json.dumps(obj)
        return True
    except Exception:
        return False


def _safe(obj: Any, limit: int = 200000) -> Any:
    """Best-effort: return JSON-serializable structure or a truncated repr."""
    if _is_jsonable(obj):
        return obj
    try:
        return _truncate(repr(obj), limit=limit)
    except Exception:
        return f"<unreprable {type(obj).__name__}>"


def _guess_working_dir(state: Any) -> Optional[str]:
    if not isinstance(state, dict):
        return None
    session = state.get("session")
    wd = None
    if isinstance(session, dict):
        wd = session.get("working_dir") or session.get("workdir") or session.get("cwd")
    else:
        wd = (
            getattr(session, "working_dir", None)
            or getattr(session, "workdir", None)
            or getattr(session, "cwd", None)
        )
    if wd:
        try:
            return str(wd)
        except Exception:
            return None
    return None


def _state_snapshot(state: Any) -> Dict[str, Any]:
    if not isinstance(state, dict):
        return {"state_type": type(state).__name__, "state_repr": _safe(state, limit=8000)}

    snap: Dict[str, Any] = {
        "keys": sorted(list(state.keys())),
        "last_status": state.get("last_status"),
        "run_id": state.get("run_id"),
        "run_started_utc": state.get("run_started_utc"),
        "run_finished_utc": state.get("run_finished_utc"),
    }

    run_log = state.get("run_log")
    if isinstance(run_log, list):
        snap["run_log_tail"] = run_log[-20:]

    artifacts = state.get("artifacts")
    if isinstance(artifacts, dict):
        snap["artifact_keys"] = sorted(list(artifacts.keys()))

    node_results = state.get("node_results")
    if isinstance(node_results, dict):
        snap["node_results_keys"] = sorted(list(node_results.keys()))

    session = state.get("session")
    snap["session_type"] = type(session).__name__ if session is not None else None

    wd = _guess_working_dir(state)
    if wd:
        snap["working_dir"] = wd

    return snap


# ---------------------------
# Bug report
# ---------------------------


def build_bug_report(
    *,
    node_id: str,
    node_spec: Dict[str, Any],
    state: Any,
    error: BaseException,
    stage: str = "node_execution",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compact, JSON-friendly runtime bug report for unexpected exceptions."""

    tb = traceback.format_exc()
    report: Dict[str, Any] = {
        "type": "orchestrator_runtime_bug",
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "node_id": node_id,
        "node_spec": _safe(node_spec),
        "exception": {"class": type(error).__name__, "message": str(error)},
        "traceback": _truncate(tb, limit=200000),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
        },
        "state_snapshot": _state_snapshot(state),
    }
    if extra:
        report["extra"] = _safe(extra)
    return report


def _bug_report_to_markdown(report: Dict[str, Any]) -> str:
    exc = (report.get("exception") or {})
    node_id = report.get("node_id")
    ts = report.get("timestamp_utc")
    stage = report.get("stage")
    tb = report.get("traceback") or ""

    lines = []
    lines.append("# Orchestrator runtime bug report")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {ts}")
    lines.append(f"- Stage: {stage}")
    lines.append(f"- Node: {node_id}")
    lines.append(f"- Exception: {exc.get('class')}: {exc.get('message')}")
    lines.append("")

    lines.append("## State snapshot")
    snap = report.get("state_snapshot") or {}
    for k in ("run_id", "run_started_utc", "run_finished_utc", "last_status", "working_dir", "session_type"):
        if k in snap:
            lines.append(f"- {k}: {snap.get(k)}")
    if snap.get("artifact_keys") is not None:
        lines.append(f"- artifact_keys: {snap.get('artifact_keys')}")
    if snap.get("node_results_keys") is not None:
        lines.append(f"- node_results_keys: {snap.get('node_results_keys')}")
    if snap.get("run_log_tail") is not None:
        lines.append("")
        lines.append("### run_log_tail")
        lines.append("```json")
        lines.append(json.dumps(snap.get("run_log_tail"), indent=2, ensure_ascii=False))
        lines.append("```")

    lines.append("")
    lines.append("## Traceback")
    lines.append("```")
    lines.append(tb.rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Node spec (JSON)")
    lines.append("```json")
    lines.append(json.dumps(report.get("node_spec"), indent=2, ensure_ascii=False))
    lines.append("```")
    return "\n".join(lines)


def write_bug_report(
    report: Dict[str, Any],
    *,
    out_dir: str,
    prefix: str = "orchestrator",
) -> Tuple[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    node_id = report.get("node_id") or "unknown_node"
    ts = _utc_timestamp_compact()
    stem = f"{prefix}_{ts}_{node_id}"

    json_path = str(Path(out_dir) / f"{stem}.json")
    md_path = str(Path(out_dir) / f"{stem}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_bug_report_to_markdown(report))

    return json_path, md_path


def report_bug_and_persist(
    *,
    node_id: str,
    node_spec: Dict[str, Any],
    state: Any,
    error: BaseException,
    out_dir: Optional[str] = None,
    prefix: str = "orchestrator",
    stage: str = "node_execution",
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str, str]:
    if out_dir is None:
        wd = _guess_working_dir(state)
        out_dir = str(Path(wd) / "bug_reports") if wd else "bug_reports"

    report = build_bug_report(
        node_id=node_id,
        node_spec=node_spec,
        state=state,
        error=error,
        stage=stage,
        extra=extra,
    )
    json_path, md_path = write_bug_report(report, out_dir=out_dir, prefix=prefix)
    return report, json_path, md_path


# ---------------------------
# Runtime report
# ---------------------------


def build_runtime_report(
    *,
    plan: Dict[str, Any],
    state: Any,
    final_report: Optional[Dict[str, Any]] = None,
    stage: str = "graph_complete",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """JSON-friendly, end-of-run report for the whole graph execution."""

    if not isinstance(state, dict):
        state_obj: Dict[str, Any] = {"state_type": type(state).__name__, "state_repr": _safe(state)}
    else:
        state_obj = dict(state)

    run_log = state_obj.get("run_log") if isinstance(state_obj.get("run_log"), list) else []
    artifacts = state_obj.get("artifacts") if isinstance(state_obj.get("artifacts"), dict) else {}
    node_results = state_obj.get("node_results") if isinstance(state_obj.get("node_results"), dict) else {}

    node_results_tail = {}
    if isinstance(node_results, dict):
        # small tail snapshot, stable ordering
        keys = sorted(node_results.keys())
        tail_keys = keys[-10:]
        for k in tail_keys:
            node_results_tail[k] = _safe(node_results.get(k), limit=8000)

    report: Dict[str, Any] = {
        "type": "orchestrator_runtime_report",
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": state_obj.get("run_id"),
        "run_started_utc": state_obj.get("run_started_utc"),
        "run_finished_utc": state_obj.get("run_finished_utc"),
        "final_status": state_obj.get("last_status"),
        "plan": _safe(plan),
        "summary": {
            "nodes_total": len((plan.get("nodes") or [])) if isinstance(plan, dict) else None,
            "run_log_len": len(run_log),
            "artifact_keys": sorted(list(artifacts.keys())),
            "node_results_keys": sorted(list(node_results.keys())),
        },
        "run_log": _safe(run_log),
        "final_report": _safe(final_report) if final_report is not None else _safe(state_obj.get("final_report")),
        "node_results_tail": node_results_tail,
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cwd": os.getcwd(),
            "working_dir": _guess_working_dir(state_obj),
        },
        "state_snapshot": _state_snapshot(state_obj),
    }
    if extra:
        report["extra"] = _safe(extra)
    return report


def _runtime_report_to_markdown(report: Dict[str, Any]) -> str:
    lines = []
    lines.append("# Orchestrator runtime report")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {report.get('timestamp_utc')}")
    lines.append(f"- Run ID: {report.get('run_id')}")
    lines.append(f"- Started (UTC): {report.get('run_started_utc')}")
    lines.append(f"- Finished (UTC): {report.get('run_finished_utc')}")
    lines.append(f"- Final status: {report.get('final_status')}")
    lines.append("")

    summary = report.get("summary") or {}
    lines.append("## Summary")
    for k in ("nodes_total", "run_log_len"):
        if k in summary:
            lines.append(f"- {k}: {summary.get(k)}")
    lines.append(f"- artifact_keys: {summary.get('artifact_keys')}")
    lines.append(f"- node_results_keys: {summary.get('node_results_keys')}")

    lines.append("")
    lines.append("## Run log")
    lines.append("```json")
    lines.append(json.dumps(report.get("run_log") or [], indent=2, ensure_ascii=False))
    lines.append("```")

    lines.append("")
    lines.append("## Final report")
    lines.append("```json")
    lines.append(json.dumps(report.get("final_report") or {}, indent=2, ensure_ascii=False))
    lines.append("```")

    lines.append("")
    lines.append("## Node results tail")
    lines.append("```json")
    lines.append(json.dumps(report.get("node_results_tail") or {}, indent=2, ensure_ascii=False))
    lines.append("```")

    lines.append("")
    lines.append("## Plan")
    lines.append("```json")
    lines.append(json.dumps(report.get("plan") or {}, indent=2, ensure_ascii=False))
    lines.append("```")

    return "\n".join(lines)


def write_runtime_report(
    report: Dict[str, Any],
    *,
    out_dir: str,
    prefix: str = "orchestrator",
) -> Tuple[str, str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ts = _utc_timestamp_compact()
    run_id = report.get("run_id") or "run"
    stem = f"{prefix}_{ts}_{run_id}"

    json_path = str(Path(out_dir) / f"{stem}.json")
    md_path = str(Path(out_dir) / f"{stem}.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_runtime_report_to_markdown(report))

    return json_path, md_path


def report_runtime_and_persist(
    *,
    plan: Dict[str, Any],
    state: Any,
    final_report: Optional[Dict[str, Any]] = None,
    out_dir: Optional[str] = None,
    prefix: str = "orchestrator",
    stage: str = "graph_complete",
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], str, str]:
    if out_dir is None:
        wd = _guess_working_dir(state)
        out_dir = str(Path(wd) / "runtime_reports") if wd else "runtime_reports"

    report = build_runtime_report(
        plan=plan,
        state=state,
        final_report=final_report,
        stage=stage,
        extra=extra,
    )
    json_path, md_path = write_runtime_report(report, out_dir=out_dir, prefix=prefix)
    return report, json_path, md_path
