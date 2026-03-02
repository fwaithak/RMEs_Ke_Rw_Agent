"""
Memory layer: user profiles, episodic memory, and configuration.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

import chromadb

logger = logging.getLogger("ComplianceAdvisor")


#  Memory Policy Configuration 


class MemoryConfig:
    """Tuneable memory policy parameters."""

    MAX_HISTORY_TURNS: int = 6
    MAX_EPISODES_PER_USER: int = 50
    EPISODE_MAX_AGE_DAYS: int = 2
    USE_LLM_SUMMARIZATION: bool = True

    SUMMARIZATION_PROMPT: str = (
        "Summarize the following conversation in 2-3 sentences.\n"
        "Focus on key facts, user intents, and any decisions made.\n\n"
        "Conversation:\n{history}\n\nSummary:"
    )


memory_config = MemoryConfig()


# Default profile template 

DEFAULT_PROFILE: Dict[str, Any] = {
    "jurisdiction": "unknown",
    "tax_category": "unknown",
    "turnover_tier": "unknown",
    "language": "en",
    "business_type": "unknown",
    "escalation_count": 0,
    "total_queries": 0,
    "created_at": None,
    "updated_at": None,
}

PROFILES_PATH = "./user_profiles.json"


# User Profile Manager 


class UserProfileManager:
    """
    Persistent JSON storage for user profiles (semantic memory).
    """

    def __init__(self, path: str = PROFILES_PATH) -> None:
        self.path = path

    def _read(self) -> Dict:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return {}

    def _write(self, data: Dict) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, user_id: str) -> Dict:
        store = self._read()
        if user_id not in store:
            profile = {
                **DEFAULT_PROFILE,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            store[user_id] = profile
            self._write(store)
        return store[user_id]

    def save(self, user_id: str, profile: Dict) -> None:
        store = self._read()
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        store[user_id] = profile
        self._write(store)

    def increment(self, user_id: str, field: str, by: int = 1) -> None:
        profile = self.load(user_id)
        profile[field] = profile.get(field, 0) + by
        self.save(user_id, profile)


# Episodic Memory Manager 

CHROMA_PATH = "./chroma_episodic"
EPISODIC_COLLECTION = "episodic_memory"


class EpisodicMemoryManager:
    """
    Persistent storage for past interactions using ChromaDB.
    """

    def __init__(
        self,
        embed_fn: Callable,
        persist_path: str = CHROMA_PATH,
    ) -> None:
        self._embed_fn = embed_fn
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(
            name=EPISODIC_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaDB collection '{EPISODIC_COLLECTION}' ready "
            f"({self.collection.count()} episodes stored)."
        )

    def add_episode(self, state: Dict) -> None:
        """Store a completed interaction."""
        query = state.get("current_query", "")
        response = state.get("response", "")
        if not query or not response:
            return

        text = f"{query} {response}"
        embed = self._embed_fn(text)

        citations = [
            d.get("metadata", {}).get("source", "")
            for d in state.get("retrieved_docs", [])
        ]
        episode_id = (
            f"{state['user_id']}_"
            f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        )

        self.collection.add(
            ids=[episode_id],
            embeddings=[embed],
            documents=[text],
            metadatas=[
                {
                    "user_id": state["user_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "query": query[:500],
                    "response": response[:500],
                    "citations": json.dumps(citations),
                    "confidence": float(state.get("confidence", 0.0)),
                    "action_taken": state.get("action", ""),
                }
            ],
        )

        self._prune_user_episodes(state["user_id"])

    def search(self, query: str, user_id: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k relevant episodes for this user."""
        if self.collection.count() == 0:
            return []

        embed = self._embed_fn(query)
        try:
            results = self.collection.query(
                query_embeddings=[embed],
                n_results=min(top_k, self.collection.count()),
                where={"user_id": user_id},
                include=["metadatas", "documents", "distances"],
            )
        except Exception:
            return []

        episodes: List[Dict] = []
        if results and results["metadatas"]:
            for meta, doc, dist in zip(
                results["metadatas"][0],
                results["documents"][0],
                results["distances"][0],
            ):
                similarity = round(1 - dist / 2, 4)
                episodes.append(
                    {**meta, "similarity": similarity, "full_text": doc}
                )
        return episodes

    def _prune_user_episodes(self, user_id: str) -> None:
        """Remove old episodes per pruning policy (FIFO + TTL)."""
        all_user = self.collection.get(where={"user_id": user_id})
        if not all_user["ids"]:
            return

        episodes: List[Dict] = []
        for idx, meta in enumerate(all_user["metadatas"]):
            episodes.append({
                "id": all_user["ids"][idx],
                "timestamp": meta.get("timestamp", "1970-01-01T00:00:00"),
            })

        episodes.sort(key=lambda x: x["timestamp"])

        cutoff = (
            datetime.now(timezone.utc).timestamp()
            - memory_config.EPISODE_MAX_AGE_DAYS * 86400
        )
        to_delete: List[str] = []
        for ep in episodes:
            try:
                ts = datetime.fromisoformat(ep["timestamp"]).timestamp()
                if ts < cutoff:
                    to_delete.append(ep["id"])
            except Exception:
                continue

        keep_count = len(episodes) - len(to_delete)
        if keep_count > memory_config.MAX_EPISODES_PER_USER:
            excess = keep_count - memory_config.MAX_EPISODES_PER_USER
            remaining = [
                ep for ep in episodes if ep["id"] not in to_delete
            ]
            remaining.sort(key=lambda x: x["timestamp"])
            to_delete.extend(ep["id"] for ep in remaining[:excess])

        if to_delete:
            self.collection.delete(ids=to_delete)
            logger.info(
                f"Pruned {len(to_delete)} old episodes for user {user_id}"
            )
