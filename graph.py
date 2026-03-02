"""
LangGraph state-graph builder.

Constructs the full ORDAEU graph with conditional edges.
"""

import sqlite3
from typing import Any, Dict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from config import Config
from state import AgentState

CHECKPOINT_DB = "./agent_checkpoints.db"


def _route_after_act(state: AgentState) -> str:
    """
    On revision passes, skip tool nodes to avoid duplicate calendar events.
    Normal first-pass ANSWER routes through the tool chain.
    """
    action = state.get("action", "")
    revision = state.get("revision_count", 0)

    if action in ("ANSWER", "ANSWER_WITH_CAVEAT"):
        if revision == 0:
            return "generate_documents"
        return "evaluate"
    return "update_memory"


def _route_after_evaluate(state: AgentState) -> str:
    """If revision needed, loop back to retrieval with broader strategy."""
    if state.get("needs_revision", False):
        return "retrieval"
    return "update_memory"


def build_graph(
    nodes: Dict[str, Any],
    checkpoint_db: str = CHECKPOINT_DB,
) -> tuple:
    """
    Build and compile the LangGraph state graph.

    Returns (compiled_graph, sqlite_connection).
    """
    conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    builder = StateGraph(AgentState)

    builder.add_node("load_profile", nodes["load_profile"])
    builder.add_node("reasoning", nodes["reasoning"])
    builder.add_node("retrieval", nodes["retrieval"])
    builder.add_node("decide", nodes["decide"])
    builder.add_node("act", nodes["act"])
    builder.add_node("generate_documents", nodes["generate_documents"])
    builder.add_node("schedule_reminders", nodes["schedule_reminders"])
    builder.add_node("evaluate", nodes["evaluate"])
    builder.add_node("update_memory", nodes["update_memory"])

    # Linear backbone
    builder.set_entry_point("load_profile")
    builder.add_edge("load_profile", "reasoning")
    builder.add_edge("reasoning", "retrieval")
    builder.add_edge("retrieval", "decide")
    builder.add_edge("decide", "act")

    # After act: tool chain OR evaluate (revision) OR memory
    builder.add_conditional_edges(
        "act",
        _route_after_act,
        {
            "generate_documents": "generate_documents",
            "evaluate": "evaluate",
            "update_memory": "update_memory",
        },
    )

    # Tool chain → evaluate
    builder.add_edge("generate_documents", "schedule_reminders")
    builder.add_edge("schedule_reminders", "evaluate")

    # After evaluate: re-retrieve OR finish
    builder.add_conditional_edges(
        "evaluate",
        _route_after_evaluate,
        {
            "retrieval": "retrieval",
            "update_memory": "update_memory",
        },
    )

    builder.add_edge("update_memory", END)

    graph = builder.compile(checkpointer=checkpointer)
    return graph, conn
