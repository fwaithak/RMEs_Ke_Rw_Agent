# RAG-Powered Compliance Advisor for African SMEs

## HW2: Building the Agent | Team 16

**Course:** 04801-W3 Agentic AI: Foundations and Applications
**Jurisdictions:** Kenya & Rwanda | **Focus:** Monthly VAT, PAYE, and Corporate Tax compliance

Please find our technical brief here: **[Technical Brief](technical_brief.md)**

---

## 📋 Project Overview

This project implements a RAG-enabled agent that assists African SMEs with tax compliance across Kenya and Rwanda. The system combines intelligent retrieval, tool-based actions, and verification guardrails to provide accurate, actionable compliance guidance.

A **Streamlit** web interface provides an interactive chat experience identical to the original Jupyter notebook.

## 🏗️ Agent Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUERY                                 │
│  "What is Kenya's VAT filing deadline?"                       │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 AGENT ORCHESTRATOR                           │
│  (ReAct Pattern with Bounded Reasoning Loop)                 │
│                                                              │
│  1. OBSERVE → 2. REASON → 3. ACT → 4. VERIFY → 5. RESPOND   │
└─────┬────────────────┬────────────────┬─────────────────────┘
      │                │                │
      ▼                ▼                ▼
┌────────────┐  ┌────────────┐  ┌──────────────┐
│ RETRIEVAL  │  │ TOOL-      │  │ VERIFICATION │
│ MODULE     │  │ CALLING    │  │ MODULE       │
│            │  │ MODULE     │  │              │
│ • ChromaDB │  │ • Regulatory│  │ • Groundedness│
│ • Semantic │  │   Search    │  │   Scoring    │
│   Chunking │  │ • Calendar  │  │ • Hallucination│
│ • Embedding│  │   Sync      │  │   Detection  │
│ • Filtered │  │ • Human     │  │ • Auto-      │
│   Search   │  │   Escalation│  │   Escalation │
└────────────┘  └────────────┘  └──────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- An **OpenAI API key** (or compatible endpoint)

### 1. Clone the repository

```bash
git clone https://github.com/Olusamimaths/RAG-for-SMEs.git
cd RAG-for-SMEs
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Copy the example file and fill in your API key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENAI_API_KEY=sk-your-key-here
```

> **Note:** You can also use Streamlit's built-in secrets management by creating `.streamlit/secrets.toml`:
> ```toml
> OPENAI_API_KEY = "sk-your-key-here"
> ```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
├── app.py                  # Streamlit chat interface
├── config.py               # Configuration & LLM initialization
├── loader.py               # Regulatory document loader (live + fallback)
├── retrieval.py            # Hybrid retrieval, reranking, conflict detection
├── memory.py               # User profiles & episodic memory (ChromaDB)
├── state.py                # AgentState TypedDict & ReasoningOutput model
├── helpers.py              # Keyword extraction & summarisation utilities
├── tools.py                # Mock tool APIs (document automation, scheduler)
├── nodes.py                # LangGraph node functions (ORDAEU cycle)
├── graph.py                # LangGraph state-graph builder
├── agent.py                # AgentRunner wrapper class
├── main.ipynb              # Original Jupyter notebook
├── technical_brief.md      # Detailed technical documentation
├── execution_trace.json    # Agent execution logs
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 📊 Core Modules

### 1. **Retrieval Module** ("The Memory") — `retrieval.py`
- **Vector Database**: ChromaDB for fast semantic search
- **Chunking Strategy**: Semantic chunking with regulatory-aware separators (Section/Article boundaries)
- **Embeddings**: OpenAI's `text-embedding-3-small`
- **Hybrid Search**: Dense (embedding) + Sparse (BM25) with Reciprocal Rank Fusion
- **Reranking**: Cross-encoder reranking for precision
- **Features**: Metadata filtering, confidence scoring, citation preservation

### 2. **Tool-Calling Module** ("The Actions") — `tools.py`, `nodes.py`
- **Regulatory Search**: Structured queries for tax rates, deadlines, penalties
- **Calendar Sync**: Simulated deadline scheduling with reminders
- **Human Escalation**: Expert handoff for complex/ambiguous queries
- **ReAct Loop**: Bounded reasoning with maximum 5 iterations

### 3. **Verification Module** ("The Guardrails") — `nodes.py`
- **LLM Evaluation**: Groundedness scoring against source documents
- **Auto-Escalation**: Triggers at <50% groundedness threshold
- **Revision Loop**: Automatic re-retrieval with broader strategies
- **Hallucination Detection**: Flags unverified critical claims

## 🧪 Example Queries

1. **Direct Retrieval**: "What are Kenya's VAT requirements?"
2. **Multi-Step Tools**: "What is Rwanda's VAT rate? Schedule a reminder."
3. **Penalty Lookup**: "What are the penalties for late VAT filing in Kenya?"
4. **Cross-jurisdiction**: "Compare VAT rates between Kenya and Rwanda"
5. **Clarification Flow**: "What taxes do I need to pay?" (triggers clarification)

## 📈 Performance Metrics

- **Groundedness Threshold**: 70% (flags responses for review)
- **Escalation Threshold**: 50% (triggers expert handoff)
- **Retrieval Confidence**: Cosine similarity + cross-encoder scoring
- **Chunk Optimization**: 500 characters with 100-character overlap

## 📚 Documentation

- **[Technical Brief](technical_brief.md)**: Detailed design decisions, failure analysis, and implementation rationale
- **Execution Traces**: Complete logs of agent reasoning, tool calls, and verification steps

## 🔮 Future Enhancements

1. **Real API Integrations**: Connect to KRA iTax, RRA portals, and Google Calendar
2. **Memory Layer**: Short-term session memory and long-term persistent storage for SME profiles
3. **Document Draft Generation**: Automatically create compliance checklists and filing templates
4. **Dual LLM Architecture**: Separate Reasoning and Selector/Scoring LLMs
5. **Natural Language Updates**: Allow administrators to add/update regulations via chat
6. **Audit Trail Generation**: Produce detailed compliance audit reports with citation trails
