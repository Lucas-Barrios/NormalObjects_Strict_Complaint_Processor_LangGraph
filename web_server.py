"""
NormalObjects Complaint Processor — Web Server
FastAPI + SSE backend for the complaint-tracking UI
"""

import sys
import json
import uuid
import asyncio
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

# ── Thread-local stdout multiplexer ───────────────────────────────────────────
# Installed BEFORE importing the workflow module so every print() inside nodes
# is automatically routed to the active complaint's log buffer.

class _ThreadLocalWriter:
    """Multiplexes print() to both the real stdout and a per-thread log buffer."""

    def __init__(self):
        self._local = threading.local()
        self._orig = sys.__stdout__

    def set_callback(self, fn):
        self._local.callback = fn

    def clear_callback(self):
        self._local.callback = None

    def write(self, text: str):
        self._orig.write(text)
        cb = getattr(self._local, "callback", None)
        if cb and text.strip():
            cb({"type": "log", "text": text.rstrip()})

    def flush(self):
        self._orig.flush()

    def isatty(self) -> bool:
        return False


_writer = _ThreadLocalWriter()
sys.stdout = _writer  # must happen before workflow import

# ── Workflow imports ───────────────────────────────────────────────────────────

from normalobjects_langgraph import (  # noqa: E402
    app as _graph,
    save_complaint_summary,
    list_saved_complaints,
    load_complaint_state,
    ComplaintState,
    DB_PATH,
)
from langgraph.types import Command  # noqa: E402

# ── FastAPI app ────────────────────────────────────────────────────────────────

api = FastAPI(title="NormalObjects Complaint Processor")

# ── In-memory run registry ─────────────────────────────────────────────────────
# { cid: { status, final, logs: list[dict], hitl: dict|None } }
_runs: dict[str, dict] = {}


# ── Pydantic models ────────────────────────────────────────────────────────────

class ComplaintRequest(BaseModel):
    complainant_name: str
    raw_complaint: str
    complaint_id: Optional[str] = None


class DecisionRequest(BaseModel):
    decision: str  # "approve" | "reject" | override text


# ── Helpers ────────────────────────────────────────────────────────────────────

def _initial_state(cid: str, complainant: str, complaint: str) -> ComplaintState:
    return {
        "complaint_id": cid, "raw_complaint": complaint,
        "complainant_name": complainant,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "category": None, "parsed_details": None,
        "missing_fields": None, "duplicate_of": None,
        "is_valid": None, "validation_notes": None,
        "investigation_angle": None, "partial_findings": [],
        "investigation_findings": None, "investigation_complete": None,
        "resolution": None, "resolution_protocol": None,
        "effectiveness_rating": None, "requires_escalation": None,
        "human_approval_required": None, "human_decision": None,
        "resolution_applied": None, "customer_satisfaction": None,
        "closed_at": None, "follow_up_required": None,
        "current_step": "intake", "workflow_path": [],
        "error_message": None, "retry_counts": {}, "messages": [],
    }


# ── Background thread workers ─────────────────────────────────────────────────

def _run_thread(cid: str, complainant: str, complaint: str) -> None:
    run = _runs[cid]
    _writer.set_callback(lambda msg: run["logs"].append(msg))
    try:
        config = {"configurable": {"thread_id": cid}}
        final = None
        for chunk in _graph.stream(_initial_state(cid, complainant, complaint), config, stream_mode="values"):
            final = chunk
            path = chunk.get("workflow_path", [])
            if path:
                run["logs"].append({"type": "step", "step": path[-1]})

        snapshot = _graph.get_state(config)
        if snapshot.next:
            run["status"] = "awaiting_approval"
            run["hitl"] = {
                "complaint_id": cid,
                "resolution":   (final or {}).get("resolution", ""),
                "protocol":     (final or {}).get("resolution_protocol", ""),
                "rating":       (final or {}).get("effectiveness_rating", ""),
            }
            run["logs"].append({"type": "hitl",   "data": run["hitl"]})
            run["logs"].append({"type": "status",  "status": "awaiting_approval"})
        else:
            run["final"]  = final
            run["status"] = (final or {}).get("current_step", "failed")
            if final:
                save_complaint_summary(final)
            run["logs"].append({"type": "status", "status": run["status"]})

    except Exception as exc:
        run["logs"].append({"type": "error", "text": str(exc)})
        run["status"] = "error"
    finally:
        _writer.clear_callback()
        run["logs"].append({"type": "done"})


def _resume_thread(cid: str, decision: str) -> None:
    run = _runs[cid]
    _writer.set_callback(lambda msg: run["logs"].append(msg))
    try:
        config = {"configurable": {"thread_id": cid}}
        final = None
        for chunk in _graph.stream(Command(resume=decision), config, stream_mode="values"):
            final = chunk
            path = chunk.get("workflow_path", [])
            if path:
                run["logs"].append({"type": "step", "step": path[-1]})

        run["final"]  = final
        run["status"] = (final or {}).get("current_step", "failed")
        if final:
            save_complaint_summary(final)
        run["logs"].append({"type": "status", "status": run["status"]})

    except Exception as exc:
        run["logs"].append({"type": "error", "text": str(exc)})
        run["status"] = "error"
    finally:
        _writer.clear_callback()
        run["logs"].append({"type": "done"})


# ── API Routes ─────────────────────────────────────────────────────────────────

@api.post("/api/complaints", status_code=202)
def submit_complaint(body: ComplaintRequest):
    cid = body.complaint_id or f"C-{uuid.uuid4().hex[:6].upper()}"
    if _runs.get(cid, {}).get("status") == "processing":
        raise HTTPException(400, f"{cid} is already processing.")
    _runs[cid] = {"status": "processing", "final": None, "logs": [], "hitl": None}
    threading.Thread(
        target=_run_thread,
        args=(cid, body.complainant_name, body.raw_complaint),
        daemon=True,
    ).start()
    return {"complaint_id": cid, "status": "processing"}


@api.post("/api/complaints/{cid}/decide")
def decide_complaint(cid: str, body: DecisionRequest):
    run = _runs.get(cid)
    if not run:
        raise HTTPException(404, "Complaint not found in current session.")
    if run["status"] != "awaiting_approval":
        raise HTTPException(400, f"Status is '{run['status']}', not 'awaiting_approval'.")
    run["status"] = "processing"
    run["logs"].append({"type": "log", "text": f"[APPROVAL] Resuming with decision: {body.decision}"})
    threading.Thread(target=_resume_thread, args=(cid, body.decision), daemon=True).start()
    return {"complaint_id": cid, "decision": body.decision}


@api.get("/api/complaints")
def list_all_complaints():
    rows = list_saved_complaints()
    persisted = {r["complaint_id"] for r in rows}
    for cid, run in _runs.items():
        if cid not in persisted:
            rows.insert(0, {
                "complaint_id":    cid,
                "complainant_name": "—",
                "category":        None,
                "final_step":      run["status"],
                "effectiveness":   None,
                "protocol":        None,
                "closed_at":       None,
                "saved_at":        None,
            })
    return rows


@api.get("/api/complaints/{cid}")
def get_complaint(cid: str):
    state = load_complaint_state(cid)
    run   = _runs.get(cid, {})
    if not state and not run:
        raise HTTPException(404, "Complaint not found.")
    return {
        "state":         state,
        "memory_status": run.get("status"),
        "hitl_data":     run.get("hitl"),
    }


@api.get("/api/complaints/{cid}/logs")
async def stream_logs(cid: str):
    run = _runs.get(cid)
    if not run:
        raise HTTPException(404, "No active run in this session.")

    async def generate():
        idx = 0
        while True:
            logs = run["logs"]
            while idx < len(logs):
                yield f"data: {json.dumps(logs[idx])}\n\n"
                idx += 1
            if run["status"] not in ("processing", "awaiting_approval") and idx >= len(logs):
                yield 'data: {"type":"done"}\n\n'
                break
            await asyncio.sleep(0.2)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Frontend ───────────────────────────────────────────────────────────────────

@api.get("/", response_class=HTMLResponse)
def index():
    try:
        with open("static/index.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return HTMLResponse("<h1>static/index.html not found</h1>", status_code=500)
