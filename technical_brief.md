# Technical Brief: RAG-Enabled Compliance Advisor for African SMEs
![System Architecture](<Agentic AI Architecture.jpg>)

## 1. Implementation Robustness

Our agent successfully navigates between retrieval and tool use for project-specific compliance queries through a structured, three-module pipeline:

- **Retrieval Module** ingests and indexes regulatory documents for Kenya and Rwanda (VAT, PAYE) using ChromaDB with semantic chunking (500 chars, 100 overlap). This preserves legal boundaries (Section/Article) and ensures complete rule retrieval.

- **Tool-Calling Module** implements three custom tools:
  - `regulatory_search`: Queries the vector database with jurisdiction/tax-type filters
  - `calendar_sync`: Schedules compliance deadlines (simulated API)
  - `human_escalation`: Escalates ambiguous/low-confidence queries (A mock)

- **ReAct Orchestrator** uses a bounded reasoning loop (max 5 iterations) to dynamically choose between retrieval, tool use, or final answer generation. The agent demonstrates multi-step reasoning (e.g., Demo 2: retrieves VAT rate → schedules calendar reminder).

**Evidence from Notebook**:  
- Demo 1: Direct retrieval answers VAT deadlines with citations  
- Demo 2: Multi-step tool usage (retrieve → schedule)  
- Demo 3: Auto-escalation on complex cross-border query

---

## 2. Technical Sophistication

### Advanced Chunking Strategy
- **Semantic Chunking** using `RecursiveCharacterTextSplitter` with regulatory-aware separators: `["\nSection", "\nArticle", "\n\n", "\n", ".", " "]`  
- **Justification**: Fixed-size chunking would split tax rates from applicability clauses. Semantic chunking preserves complete regulatory rules while allowing finer splits when needed.  
- **Metadata Preservation**: Each chunk retains `jurisdiction`, `tax_category`, `citation`, `source_url` for traceability.

### Tool Design & Integration
- **Structured Tools** using Pydantic schemas ensure type-safe inputs (e.g., `jurisdiction: Literal["Kenya","Rwanda"]`)  
- **ReAct Loop Integration**: LangGraph’s `create_react_agent` orchestrates tool selection with explicit reasoning steps logged in execution traces.

### Embedding & Retrieval Optimization
- **Embedding Model**: `azure/text-embedding-3-small` via CMU AI Gateway for high-quality semantic representations  
- **Filtered Retrieval**: ChromaDB queries accept `country` and `category` filters to improve precision  
- **Confidence Scoring**: Cosine distance converted to similarity score (0–1) for transparency

---

## 3. Self-Evaluation Logic

### Verification Module Architecture
- **Three-Tier Verification**:
  1. **Exact String Matching** for direct quotes
  2. **Fuzzy Numerical Matching** (±10% tolerance for rates/amounts)
  3. **LLM Semantic Verification** (SUPPORTED/PARTIAL/UNSUPPORTED)

- **Claim Extraction**: Hybrid regex + LLM extraction identifies factual claims (rates, deadlines, penalties)

### Groundedness Scoring & Auto-Escalation
- **Score Calculation**: `verified_claims / total_claims`  
- **Thresholds**:  
  - `GROUNDEDNESS_THRESHOLD = 0.6` (below → flag for review)  
  - `ESCALATION_THRESHOLD = 0.5` (below → auto-escalate)  
- **Critical Claim Flagging**: Deadlines, penalties, obligations trigger warnings if unverified

**Evidence**: Demo 1 shows warnings for unverified deadline claims; Demo 3 triggers auto-escalation due to low groundedness on cross-border query.

---

## 4. Failure Analysis & Adjustments

### Initial Failure
- **Problem**: Early version used fixed-size chunking (512 chars), splitting "VAT rate is 16%" from "applies to all taxable goods" across chunks, causing retrieval failures and hallucinations.

### Technical Adjustment
1. **Switched to Semantic Chunking** with overlap (100 chars) to preserve regulatory context  
2. **Added Verification Module** to catch unsupported claims  
3. **Implemented Auto-Escalation** for low-confidence scenarios

### Result
- **Groundedness Improved**: Demo 2 achieves 83.3% groundedness score  
- **Hallucinations Reduced**: Unverified claims are explicitly flagged in verification report

---

## 5. Execution Trace & Transparency

- **Trace Logging**: All agent decisions, tool calls, and verification results logged with timestamps  
- **Exported Traces**: `execution_trace_demo2.json` shows complete multi-step execution:  
  ```
  [11:43:05.329] RETRIEVAL | Searching regulatory knowledge base...
  [11:43:05.878] RETRIEVAL | ✓ Found 1 relevant documents | Confidence: 92.5%
  [11:43:13.467] DECISION  | Agent selected TOOL: calendar_sync
  ```
- **Verification Reports**: Each demo includes structured report showing groundedness score, verified/unverified claims, and recommendations.

---

## 6. Contribution Statement

| Team Member | Contributions |
|-------------|---------------|
| **Francis Waithaka** (fwaithak) | Configuration Setup, Retrieval Module, Calendar Tool, Escalation Tool(mock) ChromaDB Integration, Documentation |
| **Olusola Samuel** (soolusol) | Verification Module, Agent Orchestrator, Technical Brief, Regulatory Search Tool, Groundedness Scoring & Hallucination Detection |

**Collaboration Approach**: We maintained close collaboration through regular meetings, parallel programming then bring our ideas together, as we decide the best approach from the two different approaches. We later did a review together befor submitting.


## 7. Conclusion

We delivered a fully functional RAG-enabled agent that:

1. **Retrieves** domain-specific compliance information with semantic accuracy  
2. **Acts** via custom tools for search, scheduling, and escalation  
3. **Verifies** its outputs against sources with measurable groundedness scoring  
4. **Traces** all decisions for transparency and debugging  

The system demonstrates robust multi-step reasoning, effective hallucination control, and practical integration with compliance workflows for African SMEs.

## 8. Future Enhancements

To match our proposed design architecture in our previous assignment, next we aim to:

1. **Real API Integrations**: Connect to KRA iTax, RRA portals, and Google Calendar for live data and automated actions.

2. **Memory Layer Implementation**: Add short-term session memory and long-term persistent storage for SME profiles, compliance history, and prior escalation outcomes.

3. **Document Draft Generation**: Automatically create compliance checklists, filing templates, and submission summaries using a document automation API.

4. **Dual LLM Architecture**: Implement separate Reasoning and Selector/Scoring LLMs for improved validation and reduced hallucinations.

5. **Natural Language Updates**: Allow administrators to add/update regulations via chat interface with automatic validation and indexing.

6. **Audit Trail Generation**: Automatically produce detailed compliance audit reports with full citation trails and timestamped decision logs for regulatory inspections.
