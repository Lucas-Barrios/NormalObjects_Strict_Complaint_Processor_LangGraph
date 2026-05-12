"""
NormalObjects Strict Complaint Processor
Bloyce's Protocol — LangGraph Implementation
"""

from typing import TypedDict, Optional, Literal
from datetime import datetime


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
    duplicate_of: Optional[str]              # complaint_id of the original if duplicate

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
    error_message: Optional[str]           # populated on rejection or error
    messages: list[dict]                   # full LLM message history
