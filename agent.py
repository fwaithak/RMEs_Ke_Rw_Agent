"""
AgentRunner: high-level wrapper for the compiled LangGraph.

Handles thread management and state initialisation.
"""

import uuid
from typing import Any, Dict, List, Optional

from config import Config
from memory import DEFAULT_PROFILE, UserProfileManager
from state import AgentState


class AgentRunner:
    """
    Wrapper for the compiled graph.
    Handles thread management and state initialisation.
    """

    def __init__(
        self,
        graph: Any,
        profile_manager: UserProfileManager,
        config: Config,
        user_id: str,
        thread_id: Optional[str] = None,
    ) -> None:
        self.graph = graph
        self.profile_manager = profile_manager
        self.config = config
        self.user_id = user_id
        self.thread_id = thread_id or str(uuid.uuid4())

    @property
    def _cfg(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def run(self, query: str) -> str:
        snapshot = self.graph.get_state(self._cfg)

        # Fields that must NEVER carry over between turns
        reset_each_turn = {
            "current_query": query,
            "confidence": 0.0,
            "retrieved_docs": [],
            "action": "",
            "response": "",
            "retrieval_strategy": "normal",
            "needs_revision": False,
            "revision_count": 0,
            "groundedness_score": None,
            "tool_accuracy_score": None,
            "evaluation_feedback": None,
            "eval_error": None,
            "generated_checklist": None,
            "scheduled_events": None,
            "ambiguity_flag": False,
            "jurisdiction": "unknown",
            "compliance_type": "unknown",
        }

        if snapshot.values:
            input_state = {
                **snapshot.values,
                **reset_each_turn,
            }
        else:
            input_state = AgentState(
                user_id=self.user_id,
                thread_id=self.thread_id,
                user_profile={},
                conversation_history=[],
                current_query=query,
                jurisdiction="unknown",
                compliance_type="unknown",
                language="en",
                ambiguity_flag=False,
                missing_info_type=None,
                missing_info_description=None,
                clarification_question=None,
                pending_clarification=None,
                retrieved_docs=[],
                confidence=0.0,
                retrieval_strategy="normal",
                action="",
                response="",
                generated_checklist=None,
                scheduled_events=None,
                groundedness_score=None,
                tool_accuracy_score=None,
                evaluation_feedback=None,
                eval_error=None,
                needs_revision=False,
                revision_count=0,
                max_revisions=self.config.MAX_REVISIONS,
                episodic_memories=[],
            )

        result = self.graph.invoke(input_state, config=self._cfg)
        return result.get("response", "[No response]")

    def get_last_state(self) -> dict:
        snap = self.graph.get_state(self._cfg)
        return snap.values if snap.values else {}

    def get_profile(self) -> dict:
        return self.profile_manager.load(self.user_id)

    def get_history(self) -> List[Dict]:
        return self.get_last_state().get("conversation_history", [])
