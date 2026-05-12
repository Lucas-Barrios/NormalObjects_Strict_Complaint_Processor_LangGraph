"""
NormalObjects Strict Complaint Processor
Bloyce's Protocol — LangGraph Implementation
"""

import os
import json
import re
import time
import sqlite3
import operator
from typing import TypedDict, Optional, Literal, Annotated
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send, interrupt, Command
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()


# ─── LLM Setup ────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ─── Category Types ───────────────────────────────────────────────────────────

ComplaintCategory = Literal["portal", "monster", "psychic", "environmental", "other"]

EffectivenessRating = Literal["high", "medium", "low"]

WorkflowStatus = Literal[
    "intake",
    "validate",
    "investigate",
    "resolve",
    "close",
    "rejected",
    "needs_clarification",
    "escalated",
    "failed",            # LLM retries exhausted
    "awaiting_approval", # paused at HITL checkpoint
]


# ─── State Definition ─────────────────────────────────────────────────────────

class ComplaintState(TypedDict):
    # ── Raw input ──────────────────────────────────────────────────────────────
    complaint_id: str
    raw_complaint: str
    complainant_name: str
    submitted_at: str                        # ISO-8601 timestamp

    # ── Intake ─────────────────────────────────────────────────────────────────
    category: Optional[ComplaintCategory]    # assigned during intake
    parsed_details: Optional[dict]           # who/what/when/where extracted
    missing_fields: Optional[list[str]]      # fields absent from raw complaint
    duplicate_of: Optional[str]              # complaint_id of original if duplicate

    # ── Validation ─────────────────────────────────────────────────────────────
    is_valid: Optional[bool]
    validation_notes: Optional[str]          # reason for rejection or concerns

    # ── Investigation ──────────────────────────────────────────────────────────
    investigation_angle: Optional[str]                       # set per parallel sub-task
    partial_findings: Annotated[list[str], operator.add]     # reducer merges parallel results
    investigation_findings: Optional[str]                    # synthesized after merge
    investigation_complete: Optional[bool]

    # ── Resolution ─────────────────────────────────────────────────────────────
    resolution: Optional[str]               # specific resolution text
    resolution_protocol: Optional[str]      # Downside Up procedure referenced
    effectiveness_rating: Optional[EffectivenessRating]
    requires_escalation: Optional[bool]

    # ── Human-in-the-loop ──────────────────────────────────────────────────────
    human_approval_required: Optional[bool]  # True when rating is not "high"
    human_decision: Optional[str]            # "approve", "reject", or override text

    # ── Closure ────────────────────────────────────────────────────────────────
    resolution_applied: Optional[bool]
    customer_satisfaction: Optional[str]    # verified response from complainant
    closed_at: Optional[str]               # ISO-8601 timestamp
    follow_up_required: Optional[bool]     # true when effectiveness_rating == "low"

    # ── Workflow control ───────────────────────────────────────────────────────
    current_step: WorkflowStatus
    workflow_path: list[str]               # ordered list of completed steps
    error_message: Optional[str]           # populated on rejection or error
    retry_counts: Optional[dict]           # maps step name → retries used (0 = first-attempt success)
    messages: list[dict]                   # full LLM message history


# ─── Helper ───────────────────────────────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON from an LLM response."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    return json.loads(text)


# Fields managed by Annotated reducers — must NOT be re-emitted by regular
# (non-parallel) nodes, otherwise the reducer doubles their value on each step.
_REDUCER_FIELDS = frozenset({"partial_findings"})


def _base(state: ComplaintState) -> dict:
    """Return state dict without reducer-managed fields, safe to spread in returns."""
    return {k: v for k, v in state.items() if k not in _REDUCER_FIELDS}


# ─── Retry Helpers ────────────────────────────────────────────────────────────

MAX_RETRIES = 3      # maximum LLM call attempts per step
RETRY_DELAY = 1.0    # base back-off in seconds (doubles each retry)


def _invoke_json(messages: list, step: str, max_retries: int = MAX_RETRIES) -> tuple[dict, int]:
    """
    Call the LLM and parse the JSON response with exponential back-off.
    Returns (parsed_dict, retries_used).  Raises RuntimeError after max_retries.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(messages)
            return _parse_json(response.content), attempt - 1
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            print(f"[RETRY:{step}] Attempt {attempt}/{max_retries} — JSON parse error: {exc}")
        except Exception as exc:
            last_error = exc
            print(f"[RETRY:{step}] Attempt {attempt}/{max_retries} — LLM error: {exc}")
        if attempt < max_retries:
            delay = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"[RETRY:{step}] Waiting {delay:.1f}s before retry {attempt + 1}...")
            time.sleep(delay)
    raise RuntimeError(
        f"Step '{step}' failed after {max_retries} attempt(s): {last_error}"
    ) from last_error


def _invoke_text(messages: list, step: str, max_retries: int = MAX_RETRIES) -> tuple[str, int]:
    """
    Call the LLM and return plain text with exponential back-off.
    Returns (text, retries_used).  Raises RuntimeError after max_retries.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(messages)
            return response.content.strip(), attempt - 1
        except Exception as exc:
            last_error = exc
            print(f"[RETRY:{step}] Attempt {attempt}/{max_retries} — LLM error: {exc}")
        if attempt < max_retries:
            delay = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"[RETRY:{step}] Waiting {delay:.1f}s before retry {attempt + 1}...")
            time.sleep(delay)
    raise RuntimeError(
        f"Step '{step}' failed after {max_retries} attempt(s): {last_error}"
    ) from last_error


def _failed_state(state: ComplaintState, step: str, error: Exception) -> dict:
    """Build a terminal 'failed' state dict when a node exhausts all its retries."""
    msg = str(error)
    print(f"[{step.upper()}] Exhausted retries — marking complaint as failed.")
    return {
        **_base(state),
        "current_step": "failed",
        "error_message": f"[{step}] {msg}",
        "workflow_path": state.get("workflow_path", []) + [step],
        "messages": state.get("messages", []) + [{
            "role": "system", "step": step,
            "content": f"FAILED after retries: {msg}",
        }],
    }


# ─── Node 1: Intake ───────────────────────────────────────────────────────────

def intake_node(state: ComplaintState) -> ComplaintState:
    """Step 1: Intake — Parse and categorize the complaint."""
    print("\n[INTAKE] Processing complaint...")

    complaint = state["raw_complaint"]
    complainant = state["complainant_name"]

    prompt = f"""You are processing a complaint for NormalObjects (Downside Up division).

Analyze this complaint and respond with a JSON object containing:
- "category": one of "portal", "monster", "psychic", "environmental", "other"
- "parsed_details": object with "who", "what", "when", "where" (null if missing)
- "missing_fields": list of which of [who, what, when, where] are absent or unclear

Category definitions:
- portal: Issues with portal timing, location, or behavior
- monster: Issues with creature behavior (demogorgons, etc.)
- psychic: Issues with psychic abilities or limitations
- environmental: Issues with electricity, weather, or physical environment
- other: Anything else

Complainant: {complainant}
Complaint: {complaint}

Respond ONLY with valid JSON."""

    try:
        result, retries = _invoke_json([HumanMessage(content=prompt)], "intake")
    except RuntimeError as exc:
        return _failed_state(state, "intake", exc)

    category = result.get("category", "other")
    parsed_details = result.get("parsed_details", {})
    missing_fields = result.get("missing_fields", [])

    print(f"[INTAKE] Category : {category}")
    print(f"[INTAKE] Missing  : {missing_fields or 'none'}")

    next_step: WorkflowStatus = "needs_clarification" if missing_fields else "validate"

    return {
        **_base(state),
        "category": category,
        "parsed_details": parsed_details,
        "missing_fields": missing_fields,
        "current_step": next_step,
        "workflow_path": state.get("workflow_path", []) + ["intake"],
        "retry_counts": {**(state.get("retry_counts") or {}), "intake": retries},
        "messages": state["messages"] + [{
            "role": "assistant", "step": "intake",
            "content": f"Categorized as '{category}'. Missing fields: {missing_fields}",
        }],
    }


# ─── Node 2: Validation ───────────────────────────────────────────────────────

_VALIDATION_RULES = {
    "portal":        "The complaint must reference specific location or timing anomalies.",
    "monster":       "The complaint must describe creature behavior or interactions.",
    "psychic":       "The complaint must reference specific ability limitations or malfunctions.",
    "environmental": "The complaint must connect to electricity, weather, or observable physical phenomena.",
}

def validate_node(state: ComplaintState) -> ComplaintState:
    """Step 2: Validation — Check complaint against Bloyce's category rules."""
    print("\n[VALIDATE] Checking complaint validity...")

    category = state["category"]
    complaint = state["raw_complaint"]
    parsed = state.get("parsed_details", {})

    # 'other' is always escalated per Bloyce's Protocol
    if category == "other":
        print("[VALIDATE] Category 'other' → auto-escalated for manual review.")
        return {
            **_base(state),
            "is_valid": False,
            "validation_notes": "Category 'other' requires manual review per Bloyce's Protocol.",
            "requires_escalation": True,
            "current_step": "escalated",
            "workflow_path": state["workflow_path"] + ["validate"],
            "messages": state["messages"] + [{
                "role": "assistant", "step": "validate",
                "content": "Escalated: category 'other' requires manual review.",
            }],
        }

    rule = _VALIDATION_RULES[category]

    prompt = f"""You are a NormalObjects complaint validator applying Bloyce's Protocol.

Validation rule for '{category}' complaints:
{rule}

Complaint: {complaint}
Parsed details: {json.dumps(parsed)}

Does this complaint satisfy the rule above?
Respond ONLY with a JSON object:
- "is_valid": true or false
- "reason": one-sentence explanation"""

    try:
        result, retries = _invoke_json([HumanMessage(content=prompt)], "validate")
    except RuntimeError as exc:
        return _failed_state(state, "validate", exc)

    is_valid: bool = bool(result.get("is_valid", False))
    reason: str = result.get("reason", "No reason provided.")

    print(f"[VALIDATE] Valid  : {is_valid}")
    print(f"[VALIDATE] Reason : {reason}")

    next_step: WorkflowStatus = "investigate" if is_valid else "rejected"

    return {
        **_base(state),
        "is_valid": is_valid,
        "validation_notes": reason,
        "error_message": None if is_valid else f"Rejected during validation: {reason}",
        "current_step": next_step,
        "workflow_path": state["workflow_path"] + ["validate"],
        "retry_counts": {**(state.get("retry_counts") or {}), "validate": retries},
        "messages": state["messages"] + [{
            "role": "assistant", "step": "validate",
            "content": f"Valid: {is_valid}. {reason}",
        }],
    }


# ─── Node 3: Parallel Investigation ──────────────────────────────────────────
#
# Each category is investigated from multiple angles simultaneously via
# LangGraph's Send API (fan-out). Results accumulate in partial_findings via
# the Annotated reducer, then merge_findings_node synthesises them into a
# single report before the workflow continues to resolution.

_INVESTIGATION_ANGLES: dict[str, list[str]] = {
    "portal":        ["temporal_patterns", "location_consistency", "environmental_factors"],
    "monster":       ["behavioral_data",   "interaction_patterns", "environmental_triggers"],
    "psychic":       ["ability_specifications", "tested_limitations", "contextual_factors"],
    "environmental": ["power_line_activity", "atmospheric_conditions", "anomaly_correlation"],
    "other":         ["general_analysis"],
}


def investigate_angle_node(state: ComplaintState) -> dict:
    """
    Single-angle sub-investigator — runs in parallel with sibling instances.
    Returns ONLY the delta (partial_findings) so that the reducer can merge
    results from all concurrent tasks without overwriting each other.
    """
    angle: str = state.get("investigation_angle") or "general_analysis"
    category: str = state["category"]
    complaint: str = state["raw_complaint"]
    parsed: dict = state.get("parsed_details") or {}

    prompt = f"""You are a NormalObjects field investigator (Bloyce's Protocol).
You are examining ONE specific angle of a '{category}' complaint.

Angle: {angle.replace('_', ' ')}
Complaint: {complaint}
Parsed details: {json.dumps(parsed)}

Write 1–2 sentences of concrete, angle-specific findings. Plain text only."""

    try:
        text, retries = _invoke_text([HumanMessage(content=prompt)], f"investigate:{angle}")
        if retries:
            print(f"[INVESTIGATE:{angle}] Succeeded after {retries} retry/retries.")
    except RuntimeError as exc:
        text = f"ERROR — retries exhausted: {exc}"
        print(f"[INVESTIGATE:{angle}] Retries exhausted — recording error as finding.")

    finding = f"[{angle}] {text}"
    print(f"[INVESTIGATE:{angle}] {finding[:72]}...")
    return {"partial_findings": [finding]}   # reducer concatenates these


def merge_findings_node(state: ComplaintState) -> dict:
    """
    Fan-in node — synthesises all parallel partial_findings into one report,
    then advances current_step so the router can proceed to resolve.
    """
    partial: list[str] = state.get("partial_findings") or []

    print(f"\n[MERGE] Consolidating {len(partial)} parallel finding(s)...")
    for p in partial:
        print(f"  · {p[:78]}")

    combined = "\n".join(partial)

    prompt = f"""You are synthesising investigation results for a NormalObjects complaint.

Findings from all angles:
{combined}

Write a single cohesive investigation report (3–4 sentences) integrating all findings."""

    try:
        merged, retries = _invoke_text([HumanMessage(content=prompt)], "merge")
    except RuntimeError as exc:
        return _failed_state(state, "merge_findings", exc)

    print(f"[MERGE] Synthesis complete.")

    return {
        "investigation_findings": merged,
        "investigation_complete": True,
        "current_step": "resolve",
        "workflow_path": state["workflow_path"] + ["investigate"],
        "retry_counts": {**(state.get("retry_counts") or {}), "merge": retries},
        "messages": state["messages"] + [{
            "role": "assistant", "step": "investigate",
            "content": merged,
        }],
    }


# ─── Node 4: Resolution ───────────────────────────────────────────────────────

_RESOLUTION_PROTOCOLS = {
    "portal":        "Downside Up Portal Stabilization Protocol (DSP-7): recalibrate temporal anchors and location markers.",
    "monster":       "Downside Up Creature Containment Protocol (DCC-3): coordinate with Field Response Team for containment.",
    "psychic":       "Downside Up Psychic Ability Restoration Protocol (DPA-5): document baselines and execute recovery steps.",
    "environmental": "Downside Up Environmental Anomaly Protocol (DEA-2): coordinate with Power Grid and Atmospheric teams.",
}

# Ratings below "high" require a human officer to sign off before closure.
_APPROVAL_REQUIRED_RATINGS: frozenset[str] = frozenset({"low", "medium"})

def resolve_node(state: ComplaintState) -> ComplaintState:
    """Step 4: Resolution — Apply a specific, protocol-backed fix with effectiveness rating."""
    print("\n[RESOLVE] Generating resolution...")

    category = state["category"]
    complaint = state["raw_complaint"]
    findings = state.get("investigation_findings", "")
    protocol_context = _RESOLUTION_PROTOCOLS.get(category, "General Incident Protocol (GIP-1).")

    prompt = f"""You are a NormalObjects resolution specialist applying Bloyce's Protocol.

Protocol guidance for '{category}' complaints:
{protocol_context}

Original complaint: {complaint}
Investigation findings: {findings}

Generate a resolution. Respond ONLY with a JSON object:
- "resolution": specific resolution steps referencing the protocol (2–3 sentences)
- "resolution_protocol": exact name/code of the Downside Up protocol applied
- "effectiveness_rating": "high", "medium", or "low"
- "requires_escalation": true only if the situation is severe and needs a specialized team"""

    try:
        result, retries = _invoke_json([HumanMessage(content=prompt)], "resolve")
    except RuntimeError as exc:
        return _failed_state(state, "resolve", exc)

    resolution: str = result.get("resolution", "No resolution generated.")
    protocol: str = result.get("resolution_protocol", "GIP-1")
    rating: str = result.get("effectiveness_rating", "medium")
    escalation: bool = bool(result.get("requires_escalation", False))

    print(f"[RESOLVE] Protocol  : {protocol}")
    print(f"[RESOLVE] Rating    : {rating}")
    print(f"[RESOLVE] Escalate  : {escalation}")

    needs_approval = (rating in _APPROVAL_REQUIRED_RATINGS) and not escalation
    next_step: WorkflowStatus = (
        "escalated"          if escalation else
        "awaiting_approval"  if needs_approval else
        "close"
    )

    if needs_approval:
        print(f"[RESOLVE] Approval  : required (rating is '{rating}', not 'high')")
    else:
        print(f"[RESOLVE] Approval  : not required")

    return {
        **_base(state),
        "resolution": resolution,
        "resolution_protocol": protocol,
        "effectiveness_rating": rating,
        "requires_escalation": escalation,
        "human_approval_required": needs_approval,
        "current_step": next_step,
        "workflow_path": state["workflow_path"] + ["resolve"],
        "retry_counts": {**(state.get("retry_counts") or {}), "resolve": retries},
        "messages": state["messages"] + [{
            "role": "assistant", "step": "resolve",
            "content": f"Protocol: {protocol}. Rating: {rating}. {resolution}",
        }],
    }


# ─── Node 5: Closure ──────────────────────────────────────────────────────────

def close_node(state: ComplaintState) -> ComplaintState:
    """Step 5: Closure — Confirm resolution applied, verify satisfaction, log outcome."""
    print("\n[CLOSE] Closing complaint...")

    category = state["category"]
    resolution = state.get("resolution", "")
    rating = state.get("effectiveness_rating", "medium")
    protocol = state.get("resolution_protocol", "")

    prompt = f"""You are closing a NormalObjects complaint per Bloyce's Protocol.

Category         : {category}
Protocol applied : {protocol}
Resolution       : {resolution}
Effectiveness    : {rating}

Generate a closure record. Respond ONLY with a JSON object:
- "resolution_applied": true
- "customer_satisfaction": one-sentence simulated satisfaction note from the complainant
- "outcome_summary": one-sentence outcome suitable for the complaint log"""

    try:
        result, retries = _invoke_json([HumanMessage(content=prompt)], "close")
    except RuntimeError as exc:
        return _failed_state(state, "close", exc)

    satisfaction: str = result.get("customer_satisfaction", "Satisfaction not recorded.")
    outcome: str = result.get("outcome_summary", "Complaint closed.")
    follow_up: bool = (rating == "low")
    closed_at: str = datetime.utcnow().isoformat() + "Z"

    print(f"[CLOSE] Satisfaction  : {satisfaction}")
    print(f"[CLOSE] Follow-up     : {follow_up}")
    print(f"[CLOSE] Closed at     : {closed_at}")

    return {
        **_base(state),
        "resolution_applied": True,
        "customer_satisfaction": satisfaction,
        "closed_at": closed_at,
        "follow_up_required": follow_up,
        "current_step": "close",
        "workflow_path": state["workflow_path"] + ["close"],
        "retry_counts": {**(state.get("retry_counts") or {}), "close": retries},
        "messages": state["messages"] + [{
            "role": "assistant", "step": "close",
            "content": outcome,
        }],
    }


# ─── Node 6: Human Approval ──────────────────────────────────────────────────
#
# Triggered when resolve_node sets effectiveness_rating to "medium" or "low".
# Uses LangGraph's interrupt() to pause the graph, checkpoint state, and wait
# for the caller to provide a Command(resume=decision).
#
# decision values:
#   "approve"  → accept the LLM resolution and proceed to close
#   "reject"   → stop here; complaint is rejected by the reviewer
#   <any text> → override: replace the resolution with the reviewer's text,
#                then proceed to close

def human_approval_node(state: ComplaintState) -> dict:
    """Step 6: Human Approval — pause and wait for a senior officer to review."""
    complaint_id = state["complaint_id"]
    rating       = state.get("effectiveness_rating", "medium")
    resolution   = state.get("resolution", "")
    protocol     = state.get("resolution_protocol", "")

    print("\n" + "─" * 60)
    print(f"[APPROVAL] Human sign-off required for [{complaint_id}]")
    print(f"[APPROVAL] Rating    : {rating.upper()}  (below auto-approval threshold)")
    print(f"[APPROVAL] Protocol  : {protocol}")
    print(f"[APPROVAL] Resolution: {resolution[:120]}")
    print("─" * 60)
    print("[APPROVAL] Respond: 'approve' | 'reject' | override text")

    # ── Pause graph here; resume value becomes decision ────────────────────────
    decision: str = interrupt({
        "complaint_id": complaint_id,
        "prompt":       "Approve, reject, or provide an override resolution.",
        "resolution":   resolution,
        "protocol":     protocol,
        "rating":       rating,
    })

    decision_str = str(decision).strip()
    approved  = decision_str.lower() == "approve"
    rejected  = decision_str.lower() in ("reject", "no")
    overridden = not approved and not rejected

    if approved:
        print(f"[APPROVAL] APPROVED — proceeding to closure.")
        next_step: WorkflowStatus = "close"
        final_resolution = resolution
    elif rejected:
        print(f"[APPROVAL] REJECTED — complaint closed without resolution.")
        next_step = "rejected"
        final_resolution = resolution
    else:
        print(f"[APPROVAL] OVERRIDE — reviewer substituted a new resolution.")
        next_step = "close"
        final_resolution = decision_str

    return {
        **_base(state),
        "human_decision":  decision_str,
        "resolution":      final_resolution,
        "error_message":   "Rejected by human reviewer." if rejected else None,
        "current_step":    next_step,
        "workflow_path":   state["workflow_path"] + ["human_approval"],
        "messages":        state["messages"] + [{
            "role": "human", "step": "human_approval",
            "content": f"Decision: {decision_str}"
                       + (" (override)" if overridden else ""),
        }],
    }


# ─── Conditional Routers ──────────────────────────────────────────────────────
# Each router reads current_step (set by the node that just ran) and returns
# the name of the next node — or END for terminal states.

def _route_after_intake(state: ComplaintState) -> str:
    """
    Intake → validate            (happy path)
           → END (clarification) (missing fields flagged)
           → END (failed)        (retries exhausted)
    """
    step = state["current_step"]
    if step == "failed":
        print("[ROUTER] Intake → failed (END)")
        return END
    if step == "needs_clarification":
        print("[ROUTER] Intake → needs clarification (END)")
        return END
    print("[ROUTER] Intake → validate")
    return "validate"


def _route_after_validate(state: ComplaintState):
    """
    Validate → fan-out to N parallel investigate_angle tasks via Send
             → END (rejected, escalated, or failed)
    """
    step = state["current_step"]
    if step == "failed":
        print("[ROUTER] Validate → failed (END)")
        return END
    if step == "investigate":
        angles = _INVESTIGATION_ANGLES.get(state["category"], ["general_analysis"])
        print(f"\n[DISPATCH] Launching {len(angles)} parallel sub-investigation(s): {angles}")
        return [Send("investigate_angle", {**_base(state), "investigation_angle": a}) for a in angles]
    if step == "escalated":
        print("[ROUTER] Validate → escalated (END)")
        return END
    print("[ROUTER] Validate → rejected (END)")
    return END


def _route_after_merge(state: ComplaintState) -> str:
    """
    merge_findings → resolve (synthesis complete)
                  → END      (failed or data fundamentally insufficient)
    """
    if state["current_step"] == "failed":
        print("[ROUTER] Merge → failed (END)")
        return END
    if state.get("investigation_complete"):
        print("[ROUTER] Merge → resolve")
        return "resolve"
    print("[ROUTER] Merge → rejected (END)")
    return END


def _route_after_resolve(state: ComplaintState) -> str:
    """
    Resolve → human_approval (medium/low rating — needs sign-off)
            → close          (high rating — auto-approved)
            → END            (escalated or failed)
    """
    step = state["current_step"]
    if step == "failed":
        print("[ROUTER] Resolve → failed (END)")
        return END
    if step == "awaiting_approval":
        print("[ROUTER] Resolve → human_approval (sign-off required)")
        return "human_approval"
    if step == "close":
        print("[ROUTER] Resolve → close (high effectiveness — auto-approved)")
        return "close"
    print("[ROUTER] Resolve → escalated (END)")
    return END


def _route_after_approval(state: ComplaintState) -> str:
    """
    human_approval → close  (approved or overridden by reviewer)
                  → END     (rejected by reviewer)
    """
    step = state["current_step"]
    if step == "close":
        print("[ROUTER] Approval → close")
        return "close"
    print("[ROUTER] Approval → rejected by human (END)")
    return END


# ─── Graph Construction ───────────────────────────────────────────────────────

workflow = StateGraph(ComplaintState)

# ── Add nodes ─────────────────────────────────────────────────────────────────
workflow.add_node("intake",            intake_node)
workflow.add_node("validate",          validate_node)
workflow.add_node("investigate_angle", investigate_angle_node)   # parallel sub-task
workflow.add_node("merge_findings",    merge_findings_node)      # fan-in
workflow.add_node("resolve",           resolve_node)
workflow.add_node("human_approval",    human_approval_node)      # HITL checkpoint
workflow.add_node("close",             close_node)

# ── Entry point ───────────────────────────────────────────────────────────────
workflow.set_entry_point("intake")

# ── Edges ─────────────────────────────────────────────────────────────────────
#
#   intake ──► validate ──► Send(×N) ──► investigate_angle ──► merge_findings
#       │           │                                                  │
#       ▼           ▼                                             resolve ──► human_approval ──► close ──► END
#      END         END                                                │               │
#  (clarif.)  (reject/esc.)                                     (esc/fail)    (reject → END)
#                                          (high rating skips human_approval ──────────────►)
#
workflow.add_conditional_edges(
    "intake",
    _route_after_intake,
    {"validate": "validate", END: END},
)

# Returns list[Send] for parallel dispatch — no path_map needed
workflow.add_conditional_edges("validate", _route_after_validate)

# All parallel sub-tasks funnel into the single merge node
workflow.add_edge("investigate_angle", "merge_findings")

workflow.add_conditional_edges(
    "merge_findings",
    _route_after_merge,
    {"resolve": "resolve", END: END},
)

workflow.add_conditional_edges(
    "resolve",
    _route_after_resolve,
    {"close": "close", "human_approval": "human_approval", END: END},
)

workflow.add_conditional_edges(
    "human_approval",
    _route_after_approval,
    {"close": "close", END: END},
)

# close is always terminal
workflow.add_edge("close", END)

# ── Step 9: SQLite Persistence ────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "complaints.db")

_db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# Human-readable audit table — separate from LangGraph's checkpoint tables
_db_conn.execute("""
    CREATE TABLE IF NOT EXISTS complaint_summaries (
        complaint_id      TEXT PRIMARY KEY,
        complainant_name  TEXT,
        submitted_at      TEXT,
        category          TEXT,
        final_step        TEXT,
        effectiveness     TEXT,
        protocol          TEXT,
        human_decision    TEXT,
        closed_at         TEXT,
        follow_up         INTEGER,
        error_message     TEXT,
        saved_at          TEXT
    )""")
_db_conn.commit()

_checkpointer = SqliteSaver(_db_conn)

# ── Compile ───────────────────────────────────────────────────────────────────
app = workflow.compile(checkpointer=_checkpointer)

print("NormalObjects complaint graph compiled successfully.")
print(f"Nodes : {list(workflow.nodes.keys())}")


# ─── Step 5: Visualization ────────────────────────────────────────────────────

# Node display order used when rendering the execution trace
_NODE_ORDER = [
    "intake", "validate", "investigate", "merge_findings",
    "resolve", "human_approval", "close",
]

_STEP_LABELS = {
    "intake":          "INTAKE          — Parse & categorize",
    "validate":        "VALIDATE        — Check against rules",
    "investigate":     "INVESTIGATE (×N)— Parallel sub-investigations",
    "merge_findings":  "MERGE           — Synthesise findings",
    "resolve":         "RESOLVE         — Apply fix",
    "human_approval":  "HUMAN APPROVAL  — Senior officer sign-off",
    "close":           "CLOSE           — Confirm & log",
}

_OUTCOME_LABELS = {
    "close":               "CLOSED",
    "escalated":           "ESCALATED        — forwarded to specialist team",
    "rejected":            "REJECTED         — insufficient detail or human veto",
    "needs_clarification": "NEEDS CLARIFICATION — awaiting more info",
    "failed":              "FAILED           — LLM retries exhausted",
    "awaiting_approval":   "AWAITING APPROVAL — paused for human sign-off",
}


def print_graph_structure() -> None:
    """
    Print the static workflow graph in two formats:
      1. LangGraph's built-in ASCII diagram (requires grandalf).
      2. Mermaid source saved to workflow_graph.md for browser rendering.
    """
    print("\n" + "=" * 62)
    print("  BLOYCE'S PROTOCOL — WORKFLOW GRAPH")
    print("=" * 62)

    # ── ASCII diagram ──────────────────────────────────────────────
    try:
        print(app.get_graph().draw_ascii())
    except ImportError:
        print("  (install grandalf for ASCII graph: pip install grandalf)")
        print("""
  [__start__]
       |
   [intake] ---------> END (needs_clarification)
       |
   [validate] -------> END (rejected / escalated)
       |
  [investigate] -----> END (rejected)
       |
   [resolve] --------> END (escalated)
       |
   [close]
       |
   [__end__]
""")

    # ── Mermaid diagram saved to file ─────────────────────────────
    mermaid_src = app.get_graph().draw_mermaid()
    mermaid_path = os.path.join(os.path.dirname(__file__), "workflow_graph.md")
    with open(mermaid_path, "w") as f:
        f.write("# NormalObjects — Bloyce's Protocol Workflow\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_src)
        f.write("\n```\n")
    print(f"  Mermaid diagram saved → {mermaid_path}")
    print("=" * 62)


def visualize_execution(final: ComplaintState) -> None:
    """
    Print a step-by-step execution trace showing which nodes ran,
    what each one decided, and the final outcome.
    """
    path: list[str] = final.get("workflow_path", [])
    path_set = set(path)

    # Build a quick lookup: step → the assistant message logged by that node
    msg_by_step: dict[str, str] = {
        m["step"]: m["content"]
        for m in final.get("messages", [])
        if "step" in m
    }

    print("\n" + "╔" + "═" * 60 + "╗")
    print(f"║  EXECUTION TRACE  ·  {final['complaint_id']}  ·  {final['complainant_name']:<30}║")
    print("╠" + "═" * 60 + "╣")
    print(f"║  Category : {(final.get('category') or 'unknown'):<48}║")
    print(f"║  Path     : {(' → '.join(path) or '(none)'):<48}║")
    print("╚" + "═" * 60 + "╝")

    for i, step in enumerate(_NODE_ORDER):
        ran = step in path_set
        marker = "►" if ran else "○"
        label = _STEP_LABELS[step]

        print(f"\n  {marker}  {label}")

        if ran:
            raw_msg = msg_by_step.get(step, "")
            # Show a trimmed preview of what the node logged
            preview = raw_msg.replace("\n", " ")
            if len(preview) > 70:
                preview = preview[:67] + "..."
            if preview:
                print(f"      └─ {preview}")

            # Per-step detail callouts
            if step == "intake":
                missing = final.get("missing_fields") or []
                print(f"      └─ missing fields : {missing or 'none'}")

            elif step == "validate":
                print(f"      └─ valid          : {final.get('is_valid')}")

            elif step == "investigate":
                angles = _INVESTIGATION_ANGLES.get(final.get("category") or "", [])
                print(f"      └─ angles         : {angles}")
                for pf in (final.get("partial_findings") or []):
                    print(f"      └─ {pf[:72]}")

            elif step == "resolve":
                print(f"      └─ protocol       : {final.get('resolution_protocol', 'N/A')}")
                print(f"      └─ rating         : {final.get('effectiveness_rating', 'N/A')}")
                print(f"      └─ escalate       : {final.get('requires_escalation', False)}")
                print(f"      └─ HITL required  : {final.get('human_approval_required', False)}")

            elif step == "human_approval":
                decision = final.get("human_decision") or "(pending)"
                approved  = decision.lower() == "approve"
                rejected  = decision.lower() in ("reject", "no")
                tag = "APPROVED" if approved else ("REJECTED" if rejected else "OVERRIDE")
                print(f"      └─ decision       : {decision[:60]}")
                print(f"      └─ outcome        : {tag}")

            elif step == "close":
                follow = final.get("follow_up_required", False)
                print(f"      └─ follow-up      : {'YES — 30-day checkpoint required' if follow else 'no'}")
                print(f"      └─ closed at      : {final.get('closed_at', 'N/A')}")
        else:
            print(f"      └─ (not reached)")

        # Draw connector arrow if not the last node
        if i < len(_NODE_ORDER) - 1:
            print(f"           │")
            print(f"           ▼")

    # ── Final outcome banner ───────────────────────────────────────
    outcome_key = final["current_step"]
    outcome_label = _OUTCOME_LABELS.get(outcome_key, outcome_key.upper())
    print("\n  " + "─" * 58)
    print(f"  OUTCOME ▸  {outcome_label}")

    if outcome_key in ("rejected", "escalated"):
        reason = (final.get("error_message") or final.get("validation_notes") or "")
        if reason:
            print(f"  DETAIL  ▸  {reason[:70]}")

    elif outcome_key == "needs_clarification":
        print(f"  MISSING ▸  {final.get('missing_fields', [])}")

    if final.get("human_decision"):
        print(f"  HUMAN   ▸  decision = {final['human_decision'][:60]}")

    print("  " + "─" * 58 + "\n")


# ─── Step 9: Persistence Functions ───────────────────────────────────────────

def save_complaint_summary(final: ComplaintState) -> None:
    """Upsert a one-row human-readable summary into complaint_summaries."""
    _db_conn.execute("""
        INSERT OR REPLACE INTO complaint_summaries
          (complaint_id, complainant_name, submitted_at, category,
           final_step, effectiveness, protocol, human_decision,
           closed_at, follow_up, error_message, saved_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        final["complaint_id"],
        final["complainant_name"],
        final.get("submitted_at"),
        final.get("category"),
        final["current_step"],
        final.get("effectiveness_rating"),
        final.get("resolution_protocol"),
        final.get("human_decision"),
        final.get("closed_at"),
        int(bool(final.get("follow_up_required"))),
        final.get("error_message"),
        datetime.now(timezone.utc).isoformat(),
    ))
    _db_conn.commit()
    print(f"[DB] Saved summary for {final['complaint_id']} → {os.path.basename(DB_PATH)}")


def list_saved_complaints() -> list[dict]:
    """Return all rows from complaint_summaries, newest first."""
    cur = _db_conn.execute("""
        SELECT complaint_id, complainant_name, category,
               final_step, effectiveness, protocol, closed_at, saved_at
        FROM   complaint_summaries
        ORDER  BY saved_at DESC
    """)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def print_saved_complaints() -> None:
    """Print a table of all persisted complaints from the database."""
    rows = list_saved_complaints()
    print("\n" + "=" * 72)
    print(f"  PERSISTED COMPLAINTS  ·  {os.path.basename(DB_PATH)}")
    print("=" * 72)
    if not rows:
        print("  (no records saved yet)")
    else:
        print(f"  {'ID':<10} {'Name':<18} {'Category':<14} {'Step':<22} {'Rating'}")
        print(f"  {'─'*9} {'─'*17} {'─'*13} {'─'*21} {'─'*8}")
        for r in rows:
            print(
                f"  {r['complaint_id']:<10}"
                f" {(r['complainant_name'] or ''):<18}"
                f" {(r['category'] or 'N/A'):<14}"
                f" {r['final_step']:<22}"
                f" {r['effectiveness'] or '—'}"
            )
    print("=" * 72)


def load_complaint_state(complaint_id: str) -> ComplaintState | None:
    """
    Retrieve the latest checkpointed state for a complaint from SQLite.
    Returns None if no checkpoint exists for that thread_id.
    """
    config = {"configurable": {"thread_id": complaint_id}}
    snapshot = app.get_state(config)
    if snapshot and snapshot.values:
        return snapshot.values  # type: ignore[return-value]
    return None


# ─── Step 4: Test the Workflow ────────────────────────────────────────────────

def run_complaint(
    complaint_id: str,
    complainant: str,
    complaint: str,
    *,
    auto_approve: bool = False,
) -> ComplaintState:
    """
    Run a single complaint through the full graph and return the final state.

    When a resolution requires human sign-off:
    - auto_approve=False (default): blocks and reads a decision from stdin.
    - auto_approve=True           : silently approves (for batch / non-interactive use).
    """
    initial_state: ComplaintState = {
        "complaint_id":           complaint_id,
        "raw_complaint":          complaint,
        "complainant_name":       complainant,
        "submitted_at":           datetime.utcnow().isoformat() + "Z",
        # intake fields
        "category":               None,
        "parsed_details":         None,
        "missing_fields":         None,
        "duplicate_of":           None,
        # validation fields
        "is_valid":               None,
        "validation_notes":       None,
        # investigation fields (parallel)
        "investigation_angle":    None,
        "partial_findings":       [],
        "investigation_findings": None,
        "investigation_complete": None,
        # resolution fields
        "resolution":             None,
        "resolution_protocol":    None,
        "effectiveness_rating":   None,
        "requires_escalation":    None,
        # HITL fields
        "human_approval_required": None,
        "human_decision":          None,
        # closure fields
        "resolution_applied":     None,
        "customer_satisfaction":  None,
        "closed_at":              None,
        "follow_up_required":     None,
        # workflow control
        "current_step":           "intake",
        "workflow_path":          [],
        "error_message":          None,
        "retry_counts":           {},
        "messages":               [],
    }
    config = {"configurable": {"thread_id": complaint_id}}

    # ── First pass: run until completion or human-approval interrupt ───────────
    final: ComplaintState | None = None
    for chunk in app.stream(initial_state, config, stream_mode="values"):
        final = chunk

    # ── Check for a pending human-approval interrupt ───────────────────────────
    snapshot = app.get_state(config)
    if snapshot.next:
        if auto_approve:
            decision = "approve"
            print("[APPROVAL] Auto-approved (batch mode).")
        else:
            decision = (
                input("\n  Decision (approve / reject / override text) → ").strip()
                or "approve"
            )

        # ── Resume graph with the human decision ───────────────────────────────
        for chunk in app.stream(Command(resume=decision), config, stream_mode="values"):
            final = chunk

    save_complaint_summary(final)  # type: ignore[arg-type]
    return final  # type: ignore[return-value]


def print_result(final: ComplaintState) -> None:
    """Print a concise summary of the final complaint state."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  ID          : {final['complaint_id']}")
    print(f"  Complainant : {final['complainant_name']}")
    print(f"  Complaint   : {final['raw_complaint'][:70]}...")
    print(f"  Category    : {final.get('category', 'N/A')}")
    print(f"  Path taken  : {' → '.join(final.get('workflow_path', []))}")
    print(f"  Final step  : {final['current_step']}")

    if final["current_step"] == "close":
        print(f"  Protocol    : {final.get('resolution_protocol', 'N/A')}")
        print(f"  Effectiveness: {final.get('effectiveness_rating', 'N/A')}")
        if final.get("human_decision"):
            tag = {"approve": "APPROVED", "reject": "REJECTED"}.get(
                final["human_decision"].lower(), "OVERRIDE"
            )
            print(f"  Human review: {tag} — '{final['human_decision'][:50]}'")
        print(f"  Follow-up   : {final.get('follow_up_required', False)}")
        print(f"  Closed at   : {final.get('closed_at', 'N/A')}")
        print(f"  Satisfaction: {final.get('customer_satisfaction', 'N/A')}")
    elif final["current_step"] == "needs_clarification":
        print(f"  Missing     : {final.get('missing_fields', [])}")
        print(f"  Action      : Complainant must supply missing details before proceeding.")
    elif final["current_step"] == "rejected":
        print(f"  Reason      : {final.get('error_message', 'N/A')}")
    elif final["current_step"] == "escalated":
        print(f"  Note        : {final.get('validation_notes') or final.get('error_message', 'N/A')}")
    elif final["current_step"] == "failed":
        print(f"  Error       : {final.get('error_message', 'N/A')}")
    retry_counts = final.get("retry_counts") or {}
    if retry_counts:
        retried = {k: v for k, v in retry_counts.items() if v}
        if retried:
            print(f"  Retries     : {retried}")
    print(sep)


# ─── Step 8: Human-in-the-Loop Demo ──────────────────────────────────────────

def run_hitl_demo() -> None:
    """
    Interactive demonstration of the HITL checkpoint.
    Runs Chief Hopper's monster complaint and pauses at the approval node so
    the user can type a decision in the terminal.

    Try each option to see different outcomes:
      • approve             → complaint closes with the LLM's resolution
      • reject              → complaint is rejected by the reviewer
      • <custom text>       → your text replaces the LLM resolution, then closes
    """
    print("\n" + "=" * 62)
    print("  STEP 8 — HUMAN-IN-THE-LOOP DEMO")
    print("=" * 62)
    print("  Complaint : C-002-hitl  (Chief Hopper / monster)")
    print("  The graph will pause at 'human_approval' and wait for your input.")
    print("  Options   : 'approve' | 'reject' | type an override resolution")
    print("=" * 62)

    complaint = (
        "On October 31st in the Hawkins National Laboratory tunnel network "
        "I observed two demogorgons that alternated between coordinated "
        "pack-hunting behaviour and violent in-fighting within the same hour. "
        "One creature pinned a lab technician while the other patrolled the "
        "perimeter, which suggests a hierarchy I have not seen documented before."
    )
    # auto_approve=False → will block at the interrupt and read from stdin
    final = run_complaint("C-002-hitl", "Chief Hopper", complaint, auto_approve=False)
    print_result(final)
    visualize_execution(final)


if __name__ == "__main__":
    # ── Show static graph structure first ─────────────────────────────────────
    print_graph_structure()

    test_complaints = [
        # ── Happy-path complaints (full who/what/when/where) ──────────────────
        (
            "C-001", "Joyce Byers",
            "On the night of November 6th at our home on Maple Street, Hawkins, "
            "the portal to the Downside Up opened three times at completely "
            "unpredictable intervals — 9 PM, 2 AM, and 5 AM. Each opening lasted "
            "roughly 4 minutes and left scorch marks on the living-room wall. "
            "I need to know how to predict when it will open next so I can keep "
            "my family safe.",
        ),
        (
            "C-002", "Chief Hopper",
            "On October 31st in the Hawkins National Laboratory tunnel network "
            "I observed two demogorgons that alternated between coordinated "
            "pack-hunting behaviour and violent in-fighting within the same hour. "
            "One creature pinned a lab technician while the other patrolled the "
            "perimeter, which suggests a hierarchy I have not seen documented before.",
        ),
        (
            "C-003", "Mike Wheeler",
            "Yesterday afternoon at Hawkins Middle School gymnasium, Eleven "
            "attempted to remotely view a target in Russia using her psychic "
            "abilities. She could establish contact for roughly 10 seconds before "
            "experiencing severe nosebleeds and complete ability shutdown. She "
            "has never hit this range limitation before and cannot lift objects "
            "heavier than approximately 50 lbs since the incident.",
        ),
        (
            "C-004", "Bob Newby",
            "This past Monday evening at the Hawkins Power Station on Route 6, "
            "every transformer on the north grid tripped simultaneously the moment "
            "a pack of demodogs passed beneath the high-voltage lines. The outage "
            "lasted 22 minutes, and a 10-metre dead zone of vegetation appeared "
            "around each pylon. This has happened twice in the last two weeks.",
        ),
        # ── Deliberately vague — should trigger needs_clarification ───────────
        (
            "C-005", "Billy Hargrove",
            "Something weird happened and I want someone to fix it.",
        ),
    ]

    print("\n" + "=" * 60)
    print("  NormalObjects — Bloyce's Protocol Workflow Test")
    print("=" * 60)
    print(f"\nRunning {len(test_complaints)} test complaints...\n")

    results = []
    for cid, name, text in test_complaints:
        print(f"\n{'━' * 60}")
        print(f"  [{cid}] {name}")
        print(f"  \"{text[:65]}...\"")
        print(f"{'━' * 60}")
        # auto_approve=True keeps the batch non-interactive;
        # any HITL checkpoint is silently approved.
        final = run_complaint(cid, name, text, auto_approve=True)
        print_result(final)
        visualize_execution(final)
        results.append(final)

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'ID':<8} {'Category':<15} {'Final Step':<22} {'Rating'}")
    print(f"  {'─'*7} {'─'*14} {'─'*21} {'─'*8}")
    for r in results:
        hitl = "✓" if r.get("human_decision") else " "
        print(
            f"  {r['complaint_id']:<8}"
            f" {(r.get('category') or 'N/A'):<15}"
            f" {r['current_step']:<22}"
            f" {r.get('effectiveness_rating') or '—':<8}"
            f" HITL:{hitl}"
        )
    print("=" * 60)

    # ── Interactive HITL demo ──────────────────────────────────────────────────
    run_hitl_demo()

    # ── Step 9: Persistence demo ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  STEP 9 — PERSISTENCE DEMO")
    print("=" * 62)
    print(f"  All complaints persisted to: {DB_PATH}")

    # Show every record saved to the summaries table
    print_saved_complaints()

    # Reload one complaint directly from the SQLite checkpoint store
    sample_id = "C-001"
    print(f"\n  Reloading state for [{sample_id}] from SQLite checkpoint...")
    reloaded = load_complaint_state(sample_id)
    if reloaded:
        print(f"  complaint_id    : {reloaded['complaint_id']}")
        print(f"  complainant     : {reloaded['complainant_name']}")
        print(f"  category        : {reloaded.get('category')}")
        print(f"  final_step      : {reloaded['current_step']}")
        print(f"  workflow_path   : {' → '.join(reloaded.get('workflow_path', []))}")
        print(f"  effectiveness   : {reloaded.get('effectiveness_rating', 'N/A')}")
        print(f"  closed_at       : {reloaded.get('closed_at', 'N/A')}")
    else:
        print(f"  No checkpoint found for [{sample_id}].")
    print("=" * 62)
