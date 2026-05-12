"""
NormalObjects Strict Complaint Processor
Bloyce's Protocol — LangGraph Implementation
"""

import os
import json
import re
import operator
from typing import TypedDict, Optional, Literal, Annotated
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Send

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

    # ── Closure ────────────────────────────────────────────────────────────────
    resolution_applied: Optional[bool]
    customer_satisfaction: Optional[str]    # verified response from complainant
    closed_at: Optional[str]               # ISO-8601 timestamp
    follow_up_required: Optional[bool]     # true when effectiveness_rating == "low"

    # ── Workflow control ───────────────────────────────────────────────────────
    current_step: WorkflowStatus
    workflow_path: list[str]               # ordered list of completed steps
    error_message: Optional[str]           # populated on rejection or error
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

    response = llm.invoke([HumanMessage(content=prompt)])
    result = _parse_json(response.content)

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

    response = llm.invoke([HumanMessage(content=prompt)])
    result = _parse_json(response.content)

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

    response = llm.invoke([HumanMessage(content=prompt)])
    finding = f"[{angle}] {response.content.strip()}"

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

    response = llm.invoke([HumanMessage(content=prompt)])
    merged: str = response.content.strip()

    print(f"[MERGE] Synthesis complete.")

    return {
        "investigation_findings": merged,
        "investigation_complete": True,
        "current_step": "resolve",
        "workflow_path": state["workflow_path"] + ["investigate"],
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

    response = llm.invoke([HumanMessage(content=prompt)])
    result = _parse_json(response.content)

    resolution: str = result.get("resolution", "No resolution generated.")
    protocol: str = result.get("resolution_protocol", "GIP-1")
    rating: str = result.get("effectiveness_rating", "medium")
    escalation: bool = bool(result.get("requires_escalation", False))

    print(f"[RESOLVE] Protocol  : {protocol}")
    print(f"[RESOLVE] Rating    : {rating}")
    print(f"[RESOLVE] Escalate  : {escalation}")

    next_step: WorkflowStatus = "escalated" if escalation else "close"

    return {
        **_base(state),
        "resolution": resolution,
        "resolution_protocol": protocol,
        "effectiveness_rating": rating,
        "requires_escalation": escalation,
        "current_step": next_step,
        "workflow_path": state["workflow_path"] + ["resolve"],
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

    response = llm.invoke([HumanMessage(content=prompt)])
    result = _parse_json(response.content)

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
        "messages": state["messages"] + [{
            "role": "assistant", "step": "close",
            "content": outcome,
        }],
    }


# ─── Conditional Routers ──────────────────────────────────────────────────────
# Each router reads current_step (set by the node that just ran) and returns
# the name of the next node — or END for terminal states.

def _route_after_intake(state: ComplaintState) -> str:
    """
    Intake → validate (happy path)
           → END      (missing fields flagged for clarification)
    """
    if state["current_step"] == "needs_clarification":
        print("[ROUTER] Intake → needs clarification (END)")
        return END
    print("[ROUTER] Intake → validate")
    return "validate"


def _route_after_validate(state: ComplaintState):
    """
    Validate → fan-out to N parallel investigate_angle tasks via Send
             → END (rejected or escalated)
    """
    step = state["current_step"]
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
                  → END      (merge failed — data fundamentally insufficient)
    """
    if state.get("investigation_complete"):
        print("[ROUTER] Merge → resolve")
        return "resolve"
    print("[ROUTER] Merge → rejected (END)")
    return END


def _route_after_resolve(state: ComplaintState) -> str:
    """
    Resolve → close    (standard path)
            → END      (escalated to specialized team)
    """
    if state["current_step"] == "close":
        print("[ROUTER] Resolve → close")
        return "close"
    print("[ROUTER] Resolve → escalated (END)")
    return END


# ─── Graph Construction ───────────────────────────────────────────────────────

workflow = StateGraph(ComplaintState)

# ── Add nodes ─────────────────────────────────────────────────────────────────
workflow.add_node("intake",            intake_node)
workflow.add_node("validate",          validate_node)
workflow.add_node("investigate_angle", investigate_angle_node)   # parallel sub-task
workflow.add_node("merge_findings",    merge_findings_node)      # fan-in
workflow.add_node("resolve",           resolve_node)
workflow.add_node("close",             close_node)

# ── Entry point ───────────────────────────────────────────────────────────────
workflow.set_entry_point("intake")

# ── Edges ─────────────────────────────────────────────────────────────────────
#
#   intake ──► validate ──► Send(×N) ──► investigate_angle ──► merge_findings
#       │           │                                                  │
#       ▼           ▼                                             resolve ──► close ──► END
#      END         END                                                │
#  (clarif.)  (reject/esc.)                                          END
#                                                                 (escalated)
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
    {"close": "close", END: END},
)

# close is always terminal
workflow.add_edge("close", END)

# ── Compile ───────────────────────────────────────────────────────────────────
app = workflow.compile()

print("NormalObjects complaint graph compiled successfully.")
print(f"Nodes : {list(workflow.nodes.keys())}")


# ─── Step 5: Visualization ────────────────────────────────────────────────────

# Node display order used when rendering the execution trace
_NODE_ORDER = ["intake", "validate", "investigate", "merge_findings", "resolve", "close"]

_STEP_LABELS = {
    "intake":          "INTAKE          — Parse & categorize",
    "validate":        "VALIDATE        — Check against rules",
    "investigate":     "INVESTIGATE (×N)— Parallel sub-investigations",
    "merge_findings":  "MERGE           — Synthesise findings",
    "resolve":         "RESOLVE         — Apply fix",
    "close":           "CLOSE           — Confirm & log",
}

_OUTCOME_LABELS = {
    "close":               "CLOSED",
    "escalated":           "ESCALATED — forwarded to specialist team",
    "rejected":            "REJECTED  — insufficient detail",
    "needs_clarification": "NEEDS CLARIFICATION — awaiting more info",
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

    print("  " + "─" * 58 + "\n")


# ─── Step 4: Test the Workflow ────────────────────────────────────────────────

def run_complaint(complaint_id: str, complainant: str, complaint: str) -> ComplaintState:
    """Run a single complaint through the full graph and return the final state."""
    initial_state: ComplaintState = {
        "complaint_id":          complaint_id,
        "raw_complaint":         complaint,
        "complainant_name":      complainant,
        "submitted_at":          datetime.utcnow().isoformat() + "Z",
        # intake fields
        "category":              None,
        "parsed_details":        None,
        "missing_fields":        None,
        "duplicate_of":          None,
        # validation fields
        "is_valid":              None,
        "validation_notes":      None,
        # investigation fields (parallel)
        "investigation_angle":   None,
        "partial_findings":      [],
        "investigation_findings": None,
        "investigation_complete": None,
        # resolution fields
        "resolution":            None,
        "resolution_protocol":   None,
        "effectiveness_rating":  None,
        "requires_escalation":   None,
        # closure fields
        "resolution_applied":    None,
        "customer_satisfaction": None,
        "closed_at":             None,
        "follow_up_required":    None,
        # workflow control
        "current_step":          "intake",
        "workflow_path":         [],
        "error_message":         None,
        "messages":              [],
    }
    return app.invoke(initial_state)


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
    print(sep)


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
        final = run_complaint(cid, name, text)
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
        print(
            f"  {r['complaint_id']:<8}"
            f" {(r.get('category') or 'N/A'):<15}"
            f" {r['current_step']:<22}"
            f" {r.get('effectiveness_rating') or '—'}"
        )
    print("=" * 60)
