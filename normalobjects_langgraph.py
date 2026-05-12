"""
NormalObjects Strict Complaint Processor
Bloyce's Protocol — LangGraph Implementation
"""

import os
import json
import re
from typing import TypedDict, Optional, Literal
from datetime import datetime

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()


# ─── LLM Setup ────────────────────────────────────────────────────────────────

llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)


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
    investigation_findings: Optional[str]    # documented evidence
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
        **state,
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
            **state,
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
        **state,
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


# ─── Node 3: Investigation ────────────────────────────────────────────────────

_INVESTIGATION_FOCUS = {
    "portal":        "Investigate temporal patterns, location consistency, and environmental factors.",
    "monster":       "Gather behavioral data, interaction patterns, and environmental triggers.",
    "psychic":       "Document ability specifications, tested limitations, and contextual factors.",
    "environmental": "Analyze power line activity, atmospheric conditions, and anomaly correlation.",
}

def investigate_node(state: ComplaintState) -> ComplaintState:
    """Step 3: Investigation — Gather and document evidence per Bloyce's Protocol."""
    print("\n[INVESTIGATE] Gathering evidence...")

    category = state["category"]
    complaint = state["raw_complaint"]
    parsed = state.get("parsed_details", {})
    focus = _INVESTIGATION_FOCUS.get(category, "Perform a general investigation.")

    prompt = f"""You are a NormalObjects field investigator applying Bloyce's Protocol.

Investigation focus for '{category}' complaints:
{focus}

Original complaint: {complaint}
Parsed details: {json.dumps(parsed)}

Produce a thorough investigation report. Respond ONLY with a JSON object:
- "findings": documented evidence narrative (2–4 sentences)
- "key_factors": list of 2–4 key factors identified
- "investigation_complete": true, or false only if data is fundamentally insufficient"""

    response = llm.invoke([HumanMessage(content=prompt)])
    result = _parse_json(response.content)

    findings: str = result.get("findings", "Investigation inconclusive.")
    complete: bool = bool(result.get("investigation_complete", True))
    key_factors: list = result.get("key_factors", [])

    print(f"[INVESTIGATE] Complete    : {complete}")
    print(f"[INVESTIGATE] Key factors : {key_factors}")
    print(f"[INVESTIGATE] Findings    : {findings[:80]}...")

    next_step: WorkflowStatus = "resolve" if complete else "rejected"
    error = None if complete else "Investigation could not be completed: insufficient data."

    return {
        **state,
        "investigation_findings": findings,
        "investigation_complete": complete,
        "error_message": error,
        "current_step": next_step,
        "workflow_path": state["workflow_path"] + ["investigate"],
        "messages": state["messages"] + [{
            "role": "assistant", "step": "investigate",
            "content": findings,
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
        **state,
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
        **state,
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
