"""
LangGraph node functions: the complete ORDAEU cycle.

Observe → Reason → Decide → Act → Evaluate → Update
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config import Config
from state import AgentState, ReasoningOutput
from tools import document_automation_api, productivity_scheduler_api

logger = logging.getLogger("ComplianceAdvisor")

# ─── Constants ─────────────────────────────────────────────────────────

MAX_CLARIFICATION_ATTEMPTS = 3

REASONING_SYSTEM = """You are a tax compliance assistant for East and Central Africa.

Your job is to extract structured intent from the user's message.

Follow these rules in order:

1. **Off‑topic detection**: If the query is completely unrelated to taxes, compliance, or business regulations (e.g., sports, entertainment, general knowledge), set `ambiguity_flag=False` and leave `jurisdiction` and `compliance_type` as `"unknown"`. Do NOT ask for clarification.

2. **On‑topic, fully specified**: If you can determine both jurisdiction and compliance_type from the query OR from the conversation history / user profile, set `ambiguity_flag=False` and fill in the fields.

3. **On‑topic, ambiguous**: If the query is clearly about taxes/compliance but you cannot determine jurisdiction or compliance_type (e.g., "What taxes do I need to pay?"), set `ambiguity_flag=True`. Provide a specific `clarification_question` asking for the missing information (e.g., "Which country are you in?" or "What type of tax are you asking about?"). Set the corresponding `missing_info_type` and `missing_info_description`.

4. Never ask for information already present in the user profile.

Respond ONLY with a valid JSON object matching this schema:
{
  "jurisdiction":             "<country name or 'unknown'>",
  "compliance_type":          "<VAT | PAYE | CIT | WHT | Business License | unknown>",
  "language":                 "<en | sw | fr | rw>",
  "ambiguity_flag":           <true | false>,
  "missing_info_type":        "<jurisdiction | compliance_type | business_details | null>",
  "missing_info_description": "<explanation or null>",
  "clarification_question":   "<question string or null>"
}"""

ANSWER_SYSTEM = """You are a knowledgeable tax compliance assistant for African markets.
Answer clearly and cite sources using [Source: <name>].
If ANSWER_WITH_CAVEAT, prepend a one-sentence disclaimer."""

ANSWER_PROMPT = """\
User profile:
{profile}

Conversation history:
{history}

Retrieved documents:
{docs}

Query: {query}
Action: {action}
{caveat_note}

Provide a structured, accurate answer."""

EVALUATION_SYSTEM = """You are a strict evaluator of tax compliance answers.

Given:
- The user's query
- The assistant's answer
- The retrieved regulatory documents

Compute:
1. groundedness (0.0–1.0): how well every claim in the answer is
   directly supported by the provided source documents.
   1.0 = every statement citable; 0.5 = half supported; 0.0 = none.
2. feedback: one or two sentences explaining the score and
   identifying any unsupported claims.

Return ONLY valid JSON:
{"groundedness": <float>, "feedback": "<string>"}"""

_ACTION_STRIP_PATTERNS = [
    r"\s+and can you (?:also )?(?:schedule|set|create|add|remind).*",
    r"\s+(?:also )?(?:schedule|remind me|set a reminder|create a reminder).*",
    r"\s+can you (?:also )?.*(?:reminder|deadline|calendar).*",
]


def _extract_factual_query(query: str) -> str:
    """Strip action/tool requests from query before retrieval scoring."""
    cleaned = query
    for pattern in _ACTION_STRIP_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned or query


# ─── Node factory ──────────────────────────────────────────────────────
# Nodes that need LLM / retrieval / memory references receive them
# through the factory function below.


def create_nodes(
    llm: Any,
    llm_json: Any,
    retrieval_module: Any,
    profile_manager: Any,
    episodic_manager: Any,
    config: Config,
) -> Dict[str, Any]:
    """
    Create all graph-node functions with shared dependencies injected.

    Returns a dict of node-name → callable.
    """

    # ------------------------------------------------------------------
    # Node 1: load_profile (OBSERVE)
    # ------------------------------------------------------------------
    def load_profile_node(state: AgentState) -> dict:
        profile = profile_manager.load(state["user_id"])
        return {"user_profile": profile}

    # ------------------------------------------------------------------
    # Node 2: retrieve_episodic_memories (OBSERVE)
    # ------------------------------------------------------------------
    def retrieve_episodic_node(state: AgentState) -> dict:
        episodes = episodic_manager.search(
            query=state["current_query"],
            user_id=state["user_id"],
            top_k=3,
        )
        return {"episodic_memories": episodes}

    # ------------------------------------------------------------------
    # Node 3: reasoning (LLM-powered intent extraction)
    # ------------------------------------------------------------------
    def reasoning_node(state: AgentState) -> dict:
        profile = state.get("user_profile", {})
        history = state.get("conversation_history", [])
        query = state["current_query"]
        pending = state.get("pending_clarification")
        attempts = profile.get("clarification_attempts", 0)

        if pending:
            combined_query = (
                f"[Clarification context]\n"
                f"We previously asked: {pending}\n"
                f"User's answer: {query}"
            )
        else:
            combined_query = query

        if attempts >= MAX_CLARIFICATION_ATTEMPTS:
            return {
                "ambiguity_flag": False,
                "jurisdiction": profile.get("jurisdiction", "unknown"),
                "compliance_type": profile.get("tax_category", "unknown"),
                "language": profile.get("language", "en"),
                "missing_info_type": None,
                "missing_info_description": None,
                "clarification_question": None,
                "pending_clarification": None,
            }

        recent_history = (
            json.dumps(history[-4:], indent=2) if history else "[]"
        )

        user_prompt = (
            f"User profile:\n{json.dumps(profile, indent=2)}\n\n"
            f"Recent conversation (last 4 messages):\n{recent_history}\n\n"
            f"Current query:\n{combined_query}\n\n"
            f"Extract the required information as JSON."
        )

        try:
            response = llm_json.invoke([
                SystemMessage(content=REASONING_SYSTEM),
                HumanMessage(content=user_prompt),
            ])
            raw = json.loads(response.content)
            for field in (
                "missing_info_type",
                "missing_info_description",
                "clarification_question",
            ):
                if raw.get(field) in ("null", "none", ""):
                    raw[field] = None
            validated = ReasoningOutput(**raw)

            return {
                "jurisdiction": validated.jurisdiction,
                "compliance_type": validated.compliance_type,
                "language": validated.language,
                "ambiguity_flag": validated.ambiguity_flag,
                "missing_info_type": validated.missing_info_type,
                "missing_info_description": validated.missing_info_description,
                "clarification_question": validated.clarification_question,
                "pending_clarification": None,
            }

        except Exception as e:
            logger.warning(f"reasoning_node error: {e}")
            return {
                "ambiguity_flag": True,
                "jurisdiction": profile.get("jurisdiction", "unknown"),
                "compliance_type": profile.get("tax_category", "unknown"),
                "language": profile.get("language", "en"),
                "missing_info_type": "unknown",
                "missing_info_description": "Could not parse query.",
                "clarification_question": (
                    "I couldn't understand your question. "
                    "Could you rephrase it?"
                ),
                "pending_clarification": None,
            }

    # ------------------------------------------------------------------
    # Node 4: retrieval (ACT – data fetch)
    # ------------------------------------------------------------------
    def retrieval_node(state: AgentState) -> dict:
        if state.get("ambiguity_flag", False):
            return {
                "retrieved_docs": [],
                "confidence": 0.0,
                "retrieval_strategy": "skipped",
            }

        revision = state.get("revision_count", 0)
        raw_query = state["current_query"]

        retrieval_query = _extract_factual_query(raw_query)
        if retrieval_query != raw_query:
            logger.info(f"Query stripped for retrieval: '{retrieval_query}'")

        feedback = state.get("evaluation_feedback", "")

        if revision == 0:
            strategy = "normal"
            result = retrieval_module.search(
                query=retrieval_query,
                country=state.get("jurisdiction"),
                category=state.get("compliance_type"),
            )
        elif revision == 1:
            strategy = "broad"
            enriched = (
                f"{feedback} {retrieval_query}".strip()
                if feedback
                else retrieval_query
            )
            result = retrieval_module.search(
                query=enriched,
                country=state.get("jurisdiction"),
                category=None,
            )
        else:
            strategy = "no_filter"
            result = retrieval_module.search(
                query=retrieval_query,
                country=None,
                category=None,
            )

        logger.info(
            f"Retrieval strategy='{strategy}' revision={revision} "
            f"confidence={result.get('confidence', 0):.2f}"
        )
        return {
            "retrieved_docs": result.get("documents", []),
            "confidence": float(result.get("confidence", 0.0)),
            "retrieval_strategy": strategy,
        }

    # ------------------------------------------------------------------
    # Node 5: decide (DECIDE)
    # ------------------------------------------------------------------
    def decide_node(state: AgentState) -> dict:
        if state.get("ambiguity_flag", False):
            action = (
                "ASK_CLARIFICATION"
                if state.get("clarification_question")
                else "ESCALATE"
            )
        else:
            confidence = state.get("confidence", 0.0)
            if confidence >= config.HIGH_CONF:
                action = "ANSWER"
            elif confidence >= config.MEDIUM_CONF:
                action = "ANSWER_WITH_CAVEAT"
            else:
                action = "ESCALATE"

        return {"action": action}

    # ------------------------------------------------------------------
    # Node 6: act (ACT – generate response)
    # ------------------------------------------------------------------
    def act_node(state: AgentState) -> dict:
        action = state.get("action", "ANSWER")

        if action == "ASK_CLARIFICATION":
            question = state.get(
                "clarification_question",
                "Could you please provide more details about your query?",
            )
            return {
                "response": question,
                "pending_clarification": question,
            }

        if action == "ESCALATE":
            count = (
                state["user_profile"].get("escalation_count", 0) + 1
            )
            response = (
                f"⚠️ This query requires expert review "
                f"(confidence: {state.get('confidence', 0):.0%}). "
                f"I've flagged it for a tax specialist — escalation #{count}."
            )
            return {
                "response": response,
                "user_profile": {
                    **state["user_profile"],
                    "escalation_count": count,
                },
            }

        caveat_note = (
            "Include a one-sentence disclaimer that confidence is moderate "
            "and the user should verify with official sources."
            if action == "ANSWER_WITH_CAVEAT"
            else ""
        )

        docs_text = (
            "\n\n".join(
                f"[Source: {d.get('metadata', {}).get('citation', '?')}] "
                f"(Updated: {d.get('metadata', {}).get('last_updated', '?')}) "
                f"[Score: {d.get('score', 0):.2f}]\n"
                f"{d.get('content', '[empty]')}"
                for d in state.get("retrieved_docs", [])
            )
            or "No documents retrieved."
        )

        history_text = (
            "\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in state.get("conversation_history", [])[-6:]
            )
            or "No prior history."
        )

        prompt = ANSWER_PROMPT.format(
            profile=json.dumps(state.get("user_profile", {}), indent=2),
            history=history_text,
            docs=docs_text,
            query=state["current_query"],
            action=action,
            caveat_note=caveat_note,
        )

        llm_response = llm.invoke(
            [HumanMessage(content=ANSWER_SYSTEM + "\n\n" + prompt)]
        )
        return {"response": llm_response.content.strip()}

    # ------------------------------------------------------------------
    # Node 7: schedule_reminders
    # ------------------------------------------------------------------
    def schedule_reminders_node(state: AgentState) -> dict:
        checklist = state.get("generated_checklist")
        if not checklist:
            return {"scheduled_events": None}

        profile = state.get("user_profile", {})
        user_contact = {
            "email": profile.get(
                "email", profile.get("contact_email", "not_set")
            ),
            "name": profile.get("name", "User"),
        }

        scheduled: List[Dict] = []
        deadlines = checklist.get("deadlines", [])

        for deadline_str in deadlines:
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", deadline_str)
            deadline_date = date_match.group(0) if date_match else None
            recurrence = (
                "monthly" if "monthly" in deadline_str.lower() else None
            )
            try:
                event = productivity_scheduler_api(
                    event_type=(
                        f"{state.get('compliance_type', 'Tax')} Filing"
                    ),
                    jurisdiction=state.get("jurisdiction", "unknown"),
                    deadline_date=deadline_date,
                    recurrence=recurrence,
                    user_contact=user_contact,
                )
                scheduled.append(event)
                logger.info(
                    f"Reminder scheduled: {event['event_details']['title']}"
                )
            except Exception as exc:
                logger.warning(f"Scheduler API call failed: {exc}")
                scheduled.append(
                    {"error": str(exc), "deadline": deadline_str}
                )

        return {"scheduled_events": scheduled if scheduled else None}

    # ------------------------------------------------------------------
    # Node 8: generate_documents
    # ------------------------------------------------------------------
    def generate_documents_node(state: AgentState) -> dict:
        try:
            checklist = document_automation_api(
                jurisdiction=state.get("jurisdiction", "unknown"),
                compliance_type=state.get("compliance_type", "unknown"),
                retrieved_docs=state.get("retrieved_docs", []),
                business_profile=state.get("user_profile", {}),
            )
            logger.info(
                f"Checklist generated for "
                f"{state.get('jurisdiction')} / "
                f"{state.get('compliance_type')}"
            )
            return {
                "generated_checklist": checklist,
                "scheduled_events": None,
            }
        except Exception as exc:
            logger.warning(f"generate_documents_node failed: {exc}")
            return {
                "generated_checklist": None,
                "scheduled_events": None,
            }

    # ------------------------------------------------------------------
    # Node 9: update_memory (UPDATE)
    # ------------------------------------------------------------------
    def update_memory_node(state: AgentState) -> dict:
        action = state.get("action", "ANSWER")

        updated_history = list(
            state.get("conversation_history", [])
        ) + [
            {"role": "user", "content": state["current_query"]},
            {"role": "assistant", "content": state.get("response", "")},
        ]

        updated_profile = dict(state["user_profile"])
        updated_profile["total_queries"] = (
            updated_profile.get("total_queries", 0) + 1
        )

        if state.get("language") not in (None, "unknown"):
            updated_profile["language"] = state["language"]
        if state.get("jurisdiction") not in (None, "unknown"):
            updated_profile["jurisdiction"] = state["jurisdiction"]
        if state.get("compliance_type") not in (None, "unknown"):
            updated_profile["tax_category"] = state["compliance_type"]

        if action == "ASK_CLARIFICATION":
            updated_profile["clarification_attempts"] = (
                updated_profile.get("clarification_attempts", 0) + 1
            )
        else:
            updated_profile["clarification_attempts"] = 0

        profile_manager.save(state["user_id"], updated_profile)

        eval_reset: Dict[str, Any] = {}
        if action in ("ASK_CLARIFICATION", "ESCALATE"):
            eval_reset = {
                "groundedness_score": None,
                "tool_accuracy_score": None,
                "evaluation_feedback": None,
                "eval_error": None,
                "needs_revision": False,
            }

        if action != "ASK_CLARIFICATION":
            try:
                episode_state = {
                    **state,
                    "user_profile": updated_profile,
                    "generated_checklist_json": json.dumps(
                        state.get("generated_checklist") or {}
                    ),
                    "scheduled_events_json": json.dumps(
                        state.get("scheduled_events") or []
                    ),
                }
                episodic_manager.add_episode(episode_state)
            except Exception as exc:
                logger.warning(f"Episode storage failed: {exc}")

        return {
            "conversation_history": updated_history,
            "user_profile": updated_profile,
            "revision_count": 0,
            **eval_reset,
        }

    # ------------------------------------------------------------------
    # Node 10: evaluate
    # ------------------------------------------------------------------
    def evaluate_node(state: AgentState) -> dict:
        action = state.get("action", "")

        if action not in ("ANSWER", "ANSWER_WITH_CAVEAT"):
            return {
                "groundedness_score": None,
                "tool_accuracy_score": None,
                "evaluation_feedback": None,
                "eval_error": None,
                "needs_revision": False,
            }

        answer = state.get("response", "")
        docs = state.get("retrieved_docs", [])
        query = state.get("current_query", "")
        revision = state.get("revision_count", 0)
        max_rev = state.get("max_revisions", config.MAX_REVISIONS)

        if not answer or not docs:
            return {
                "groundedness_score": 0.0,
                "tool_accuracy_score": 0.5,
                "evaluation_feedback": (
                    "No answer or no source documents — "
                    "cannot assess groundedness."
                ),
                "eval_error": None,
                "needs_revision": revision < max_rev,
            }

        docs_text = "\n\n".join(
            f"[Doc {i + 1} | "
            f"{d.get('metadata', {}).get('citation', '?')}]:\n"
            f"{d.get('content', '')[:500]}"
            for i, d in enumerate(docs)
        )

        prompt = (
            f"Query: {query}\n\nAnswer:\n{answer}\n\n"
            f"Retrieved documents:\n{docs_text}\n\n"
            f"Evaluate groundedness based solely on these documents."
        )

        try:
            resp = llm_json.invoke([
                SystemMessage(content=EVALUATION_SYSTEM),
                HumanMessage(content=prompt),
            ])
            ev = json.loads(resp.content)
            groundedness = float(ev.get("groundedness", 0.0))
            feedback = str(ev.get("feedback", ""))

            checklist = state.get("generated_checklist") or {}
            events = state.get("scheduled_events") or []
            deadlines = checklist.get("deadlines", [])
            has_deadline_content = any(
                "deadline" in d.lower()
                or re.search(r"\d{1,2}(st|nd|rd|th)", d)
                for d in deadlines
            )
            if has_deadline_content and events:
                tool_accuracy = 1.0
            elif has_deadline_content and not events:
                tool_accuracy = 0.5
            else:
                tool_accuracy = 0.8

            needs_revision = (
                groundedness < config.MIN_GROUNDEDNESS
            ) and (revision < max_rev)

            logger.info(
                f"Evaluation: groundedness={groundedness:.2f} "
                f"tool_acc={tool_accuracy:.2f} "
                f"needs_revision={needs_revision} "
                f"revision={revision}/{max_rev}"
            )

            return {
                "groundedness_score": groundedness,
                "tool_accuracy_score": tool_accuracy,
                "evaluation_feedback": feedback,
                "eval_error": None,
                "needs_revision": needs_revision,
            }

        except Exception as exc:
            logger.warning(f"evaluate_node LLM call failed: {exc}")
            return {
                "groundedness_score": None,
                "tool_accuracy_score": None,
                "evaluation_feedback": None,
                "eval_error": str(exc),
                "needs_revision": False,
            }

    return {
        "load_profile": load_profile_node,
        "retrieve_episodic": retrieve_episodic_node,
        "reasoning": reasoning_node,
        "retrieval": retrieval_node,
        "decide": decide_node,
        "act": act_node,
        "schedule_reminders": schedule_reminders_node,
        "generate_documents": generate_documents_node,
        "update_memory": update_memory_node,
        "evaluate": evaluate_node,
    }


# ─── ORDAEU Cycle Logger ──────────────────────────────────────────────


def log_ordaeu_cycle(state: dict, turn: int = 0) -> str:
    """
    Build a structured Observe-Reason-Decide-Act-Evaluate-Update
    summary for one completed conversation turn.

    Returns the formatted string (also prints it).
    """
    lines: List[str] = []
    sep = "═" * 68
    lines.append(f"\n{sep}")
    lines.append(f"  ORDAEU CYCLE — Turn {turn}")
    lines.append(sep)
    lines.append(
        f"  O  bserve  │ query     : "
        f"{state.get('current_query', '')[:80]}"
    )
    lines.append(
        f"  R  eason   │ jurisdict : {state.get('jurisdiction', '?')}  "
        f"type: {state.get('compliance_type', '?')}  "
        f"ambiguous: {state.get('ambiguity_flag', '?')}"
    )
    lines.append(
        f"  D  ecide   │ action    : {state.get('action', '?')}  "
        f"confidence: {state.get('confidence', 0):.2f}  "
        f"strategy: {state.get('retrieval_strategy', '?')}"
    )
    resp = state.get("response", "")
    lines.append(
        f"  A  ct      │ response  : "
        f"{resp[:100]}{'…' if len(resp) > 100 else ''}"
    )
    gs = state.get("groundedness_score")
    ta = state.get("tool_accuracy_score")
    fb = state.get("evaluation_feedback", "")
    err = state.get("eval_error")
    if err:
        lines.append(f"  E  valuate │ ERROR     : {err}")
    else:
        lines.append(
            f"  E  valuate │ ground    : "
            f"{f'{gs:.2f}' if gs is not None else 'N/A'}  "
            f"tool_acc: {f'{ta:.2f}' if ta is not None else 'N/A'}  "
            f"revised: {state.get('revision_count', 0)}x"
        )
        if fb:
            lines.append(f"             │ feedback  : {fb[:100]}")
    docs = state.get("retrieved_docs", [])
    chk = state.get("generated_checklist")
    evts = state.get("scheduled_events") or []
    lines.append(
        f"  U  pdate   │ docs: {len(docs)}  "
        f"checklist: {'✓' if chk else '✗'}  "
        f"reminders: {len(evts)}  "
        f"profile_saved: ✓"
    )
    lines.append(sep)
    output = "\n".join(lines)
    print(output)
    return output
