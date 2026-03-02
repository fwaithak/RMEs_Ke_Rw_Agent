"""
Agent state and Pydantic models.
"""

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class AgentState(TypedDict):
    """Full state passed through the LangGraph."""

    # identity
    user_id: str
    thread_id: str

    # profile & history
    user_profile: Dict[str, Any]
    conversation_history: List[Dict[str, str]]

    # current turn
    current_query: str

    # reasoning
    jurisdiction: str
    compliance_type: str
    language: str
    ambiguity_flag: bool
    missing_info_type: Optional[str]
    missing_info_description: Optional[str]
    clarification_question: Optional[str]
    pending_clarification: Optional[str]
    needs_checklist: bool
    needs_calendar: bool

    # retrieval
    retrieved_docs: List[Dict]
    confidence: float
    retrieval_strategy: str  # "normal" | "broad" | "no_filter"

    # decision & response
    action: str
    response: str

    # tool outputs (Phase 3)
    generated_checklist: Optional[Dict]
    scheduled_events: Optional[List[Dict]]

    # evaluation & adaptation (Phase 4)
    groundedness_score: Optional[float]
    tool_accuracy_score: Optional[float]
    evaluation_feedback: Optional[str]
    eval_error: Optional[str]
    needs_revision: bool
    revision_count: int
    max_revisions: int

    episodic_memories: List[Dict]


class ReasoningOutput(BaseModel):
    """Structured output from the LLM reasoning step."""

    jurisdiction: str = Field(
        description="Country or region, e.g. Kenya, Rwanda, Uganda"
    )
    compliance_type: str = Field(
        description="Tax type: VAT, PAYE, CIT, WHT, Business License"
    )
    language: str = Field(description="ISO 639-1 code: en, sw, fr, rw")
    ambiguity_flag: bool = Field(
        description="True if query is ambiguous or missing critical info"
    )
    missing_info_type: Optional[str] = Field(
        default=None,
        description="'jurisdiction' | 'compliance_type' | 'business_details' | etc.",
    )
    missing_info_description: Optional[str] = Field(
        default=None, description="Explanation of what is missing"
    )
    clarification_question: Optional[str] = Field(
        default=None,
        description="Natural-language question to send the user",
    )
    needs_checklist: bool = Field(default=False, description="User wants a compliance checklist")
    needs_calendar: bool = Field(default=False, description="User wants scheduling/reminders")
