"""
Helper functions: keyword extraction, summarisation, and utilities.
"""

import json
from typing import Dict, List

from langchain_core.messages import HumanMessage

from memory import MemoryConfig, memory_config

# ─── Keyword-based extraction ─────────────────────────────────────────

JURISDICTION_KEYWORDS: Dict[str, List[str]] = {
    "Kenya": ["kenya", "ke", "kra", "nairobi", "kes"],
    "Uganda": ["uganda", "ug", "ura", "kampala", "ugx"],
    "Tanzania": ["tanzania", "tz", "tra", "dar es salaam", "tzs"],
    "Rwanda": ["rwanda", "rw", "rra", "kigali", "rwf"],
}

COMPLIANCE_KEYWORDS: Dict[str, List[str]] = {
    "VAT": ["vat", "value added tax", "value-added", "sales tax"],
    "PAYE": [
        "paye", "pay as you earn", "payroll tax",
        "employment tax", "salary tax",
    ],
    "CIT": ["corporate tax", "corporation tax", "cit", "company tax"],
    "WHT": ["withholding", "wht"],
}


def extract_jurisdiction(text: str) -> str:
    lower = text.lower()
    for country, keywords in JURISDICTION_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return country
    return "Kenya"


def extract_compliance_type(text: str) -> str:
    lower = text.lower()
    for ctype, keywords in COMPLIANCE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return ctype
    return "VAT"


def summarize_conversation(history: List[Dict], llm: object) -> str:
    """
    Summarize conversation history using either LLM or simple truncation.
    Called when history exceeds MAX_HISTORY_TURNS.
    """
    if not history:
        return ""

    if memory_config.USE_LLM_SUMMARIZATION:
        history_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}" for msg in history
        )
        prompt = memory_config.SUMMARIZATION_PROMPT.format(
            history=history_text
        )
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception:
            return (
                f"Conversation with {len(history)} messages "
                "(summarization failed)."
            )
    else:
        return (
            f"Last {min(2, len(history))} turns preserved; "
            "earlier messages omitted."
        )
