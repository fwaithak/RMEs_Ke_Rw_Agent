"""
Configuration and LLM/embeddings initialization.

All API credentials are loaded from environment variables or Streamlit secrets.
"""

import os
import logging
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("ComplianceAdvisor")


def _get_secret(key: str) -> str:
    """Retrieve a secret from Streamlit secrets, environment, or raise."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(
            f"Missing required secret: {key}. "
            "Set it in your .env file or Streamlit secrets."
        )
    return value


class Config:
    """Central configuration for the compliance advisor."""

    OPENAI_API_KEY: str = ""
    BASE_URL: str | None = None
    LLM_MODEL: str = "gpt-4o-mini-2024-07-18"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TOP_K_RETRIEVAL: int = 3

    # Recency: ignore documents older than this many days (0 = disabled)
    MAX_DOC_AGE_DAYS: int = 0
    # Confidence thresholds
    HIGH_CONF: float = 0.75
    MEDIUM_CONF: float = 0.50
    LOW_CONF: float = 0.00
    # Conflict detection: max allowed delta for numeric fields (fraction)
    CONFLICT_RATE_TOL: float = 0.01
    # Cross-encoder model
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Evaluation threshold (Phase 4)
    MIN_GROUNDEDNESS: float = 0.70
    MAX_REVISIONS: int = 2

    @classmethod
    def load(cls) -> "Config":
        """Load configuration from environment."""
        cls.OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
        cls.BASE_URL = os.getenv("OPENAI_BASE_URL", None) or None
        return cls()


def init_llm(config: Config) -> ChatOpenAI:
    """Initialize the main LLM instance."""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.BASE_URL,
        temperature=0.1,
    )


def init_llm_json(config: Config) -> ChatOpenAI:
    """Initialize the JSON-mode LLM instance."""
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.BASE_URL,
        model_kwargs={"response_format": {"type": "json_object"}},
        temperature=0,
    )


def init_embeddings(config: Config) -> OpenAIEmbeddings:
    """Initialize the embeddings model."""
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
        base_url=config.BASE_URL,
    )
