"""
Streamlit chat interface for the RAG-Powered Compliance Advisor.

Run with:
    streamlit run app.py
"""

import uuid
from datetime import datetime, timezone

import streamlit as st

from agent import AgentRunner
from config import Config, init_embeddings, init_llm, init_llm_json
from graph import build_graph
from loader import RegulatoryDocumentLoader
from memory import (
    DEFAULT_PROFILE,
    EpisodicMemoryManager,
    UserProfileManager,
)
from nodes import create_nodes
from retrieval import ConfidenceDecider, ConflictDetector, EnhancedRetrievalModule

#  Page config 

st.set_page_config(
    page_title="🌍 Kenya & Rwanda Tax Compliance Assistant",
    page_icon="🌍",
    layout="centered",
)


#  Initialisation (cached across reruns) 

@st.cache_resource(show_spinner="Loading models and building agent graph …")
def _init_app():
    """One-time heavy initialisation."""
    config = Config.load()

    llm = init_llm(config)
    llm_json = init_llm_json(config)
    embeddings = init_embeddings(config)

    # Retrieval module
    retrieval_module = EnhancedRetrievalModule(embeddings_fn=embeddings)
    loader = RegulatoryDocumentLoader()
    docs = loader.load()
    retrieval_module.ingest(docs)

    # Memory
    profile_manager = UserProfileManager()
    episodic_manager = EpisodicMemoryManager(
        embed_fn=embeddings.embed_query,
    )

    # Graph
    nodes = create_nodes(
        llm=llm,
        llm_json=llm_json,
        retrieval_module=retrieval_module,
        profile_manager=profile_manager,
        episodic_manager=episodic_manager,
        config=config,
    )
    graph, _conn = build_graph(nodes)

    return graph, profile_manager, config


try:
    graph, profile_manager, config = _init_app()
except EnvironmentError as exc:
    st.error(str(exc))
    st.info(
        "Create a `.env` file (or configure Streamlit secrets) with your "
        "`OPENAI_API_KEY`. See the README for details."
    )
    st.stop()


# Session state

INTERACTIVE_USER = "streamlit_user"

if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"st_thread_{uuid.uuid4().hex[:8]}"
    # Reset profile for a clean session
    profile_manager.save(
        INTERACTIVE_USER,
        {
            **DEFAULT_PROFILE,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


def _get_runner() -> AgentRunner:
    return AgentRunner(
        graph=graph,
        profile_manager=profile_manager,
        config=config,
        user_id=INTERACTIVE_USER,
        thread_id=st.session_state.thread_id,
    )


# UI

st.title("🌍 Kenya & Rwanda Tax Compliance Assistant")
st.caption(
    "Ask about VAT, PAYE, penalties, filing deadlines, and more for "
    "Kenya and Rwanda."
)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This assistant uses **RAG** (Retrieval-Augmented Generation) to "
        "answer tax compliance questions for **Kenya** and **Rwanda**.\n\n"
        "**Covered topics:**\n"
        "- VAT rates, registration & filing\n"
        "- PAYE tax bands & obligations\n"
        "- Penalties for late filing/payment\n"
        "- Corporate Income Tax concepts\n"
    )

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"st_thread_{uuid.uuid4().hex[:8]}"
        profile_manager.save(
            INTERACTIVE_USER,
            {
                **DEFAULT_PROFILE,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        st.rerun()


# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "meta" in msg:
            meta = msg["meta"]
            cols = st.columns(3)
            cols[0].metric("Confidence", f"{meta.get('confidence', 0):.0%}")
            cols[1].metric("Action", meta.get("action", "?"))
            gs = meta.get("groundedness")
            cols[2].metric(
                "Groundedness",
                f"{gs:.2f}" if gs is not None else "N/A",
            )


# Chat input
if user_input := st.chat_input("Ask about VAT, PAYE, penalties, deadlines…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    runner = _get_runner()
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                response = runner.run(user_input)
            except Exception as exc:
                response = f"⚠️ Error: {exc}"

        st.markdown(response)

        state = runner.get_last_state()
        checklist = state.get("generated_checklist")
        scheduled_events = state.get("scheduled_events") or []

        if checklist:
            with st.expander("📄 Generated checklist", expanded=False):
                st.json(checklist)

        if scheduled_events:
            with st.expander("⏰ Scheduled reminders", expanded=False):
                st.json(scheduled_events)
        confidence = state.get("confidence", 0.0)
        action = state.get("action", "?")
        groundedness = state.get("groundedness_score")

        meta = {
            "confidence": confidence,
            "action": action,
            "groundedness": groundedness,
        }

        cols = st.columns(3)
        cols[0].metric("Confidence", f"{confidence:.0%}")
        cols[1].metric("Action", action)
        cols[2].metric(
            "Groundedness",
            f"{groundedness:.2f}" if groundedness is not None else "N/A",
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "meta": meta}
    )
