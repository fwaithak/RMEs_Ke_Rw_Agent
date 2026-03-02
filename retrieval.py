"""
Enhanced retrieval module with hybrid dense+sparse search, reranking,
conflict detection, and confidence scoring.
"""

import logging
import math
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import Config

logger = logging.getLogger("ComplianceAdvisor")


def _parse_date(s: str) -> Optional[date]:
    """Convert various date string formats to date object."""
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except (ValueError, TypeError):
            continue
    return None


class EnhancedRetrievalModule:
    """
    Hybrid dense+sparse retrieval with reranking, recency filtering,
    and update pipeline.
    """

    def __init__(
        self,
        embeddings_fn: Any,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP,
    ) -> None:
        self.embeddings_fn = embeddings_fn
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ", "\n### ", "\nSection ", "\nArticle ",
                "\n\n", "\n", ". ", " ", "",
            ],
            length_function=len,
            is_separator_regex=False,
        )

        # ChromaDB
        self.chroma = chromadb.Client()
        try:
            self.chroma.delete_collection("compliance_docs")
        except Exception:
            pass
        self.collection = self.chroma.create_collection(
            name="compliance_docs",
            metadata={"description": "African SME Compliance Regulations"},
        )

        # BM25 state
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: List[str] = []
        self._bm25_meta: List[Dict] = []

        # Cross-encoder reranker
        logger.info(f"Loading cross-encoder: {Config.RERANKER_MODEL} …")
        self._reranker = CrossEncoder(Config.RERANKER_MODEL, max_length=512)
        logger.info("Cross-encoder ready.")

        # Feedback store (in-memory)
        self._feedback_log: List[Dict] = []

    def ingest(self, documents: List[Dict]) -> int:
        """Chunk, embed, index; rebuild BM25."""
        total = 0
        new_corpus: List[str] = []
        new_meta: List[Dict] = []

        for doc in documents:
            if Config.MAX_DOC_AGE_DAYS > 0:
                lu = doc["metadata"].get("last_updated")
                doc_date = _parse_date(lu)
                if doc_date:
                    age = (date.today() - doc_date).days
                    if age > Config.MAX_DOC_AGE_DAYS:
                        logger.warning(
                            f"Skipping stale document ({lu}): "
                            f"{doc['metadata'].get('jurisdiction')} "
                            f"{doc['metadata'].get('tax_category')}"
                        )
                        continue

            chunks = self.splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                cid = (
                    f"{doc['metadata']['jurisdiction']}_"
                    f"{doc['metadata']['tax_category']}_{i}_{len(new_corpus)}"
                )
                embedding = self.embeddings_fn.embed_query(chunk)
                meta = {
                    **doc["metadata"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_id": cid,
                }
                self.collection.add(
                    ids=[cid],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[meta],
                )
                new_corpus.append(chunk)
                new_meta.append(meta)
                total += 1

        self._bm25_corpus.extend(new_corpus)
        self._bm25_meta.extend(new_meta)

        if self._bm25_corpus:
            tokenised = [c.lower().split() for c in self._bm25_corpus]
            self._bm25 = BM25Okapi(tokenised)
            logger.info(
                f"BM25 index built with {len(self._bm25_corpus)} documents"
            )
        else:
            self._bm25 = None
            logger.warning("BM25 corpus empty – sparse retrieval disabled")

        logger.info(
            f"✓ Ingested {total} chunks | BM25 corpus: "
            f"{len(self._bm25_corpus)} chunks"
        )
        return total

    def search(
        self,
        query: str,
        country: Optional[str] = None,
        category: Optional[str] = None,
        top_k: int = Config.TOP_K_RETRIEVAL,
        min_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Hybrid search with RRF fusion, cross-encoder reranking,
        and recency filter.
        """
        conditions: List[Dict] = []
        if country:
            conditions.append({"jurisdiction": {"$eq": country}})
        if category:
            conditions.append({"tax_category": {"$eq": category}})
        where = (
            conditions[0]
            if len(conditions) == 1
            else {"$and": conditions} if conditions else None
        )

        # 1. Dense retrieval
        query_emb = self.embeddings_fn.embed_query(query)
        dense_k = min(top_k * 4, max(20, len(self._bm25_corpus)))
        dense_results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=dense_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        dense_docs: List[Dict] = []
        if dense_results["documents"] and dense_results["documents"][0]:
            for txt, meta, dist in zip(
                dense_results["documents"][0],
                dense_results["metadatas"][0],
                dense_results["distances"][0],
            ):
                dense_docs.append({
                    "content": txt,
                    "metadata": meta,
                    "dense_score": 1 - (dist / 2),
                })

        # 2. BM25 retrieval
        bm25_docs: List[Dict] = []
        if self._bm25 and self._bm25_corpus:
            tokens = query.lower().split()
            scores = self._bm25.get_scores(tokens)
            top_idx = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[:dense_k]
            for idx in top_idx:
                meta = self._bm25_meta[idx]
                if country and meta.get("jurisdiction") != country:
                    continue
                if category and meta.get("tax_category") != category:
                    continue
                bm25_docs.append({
                    "content": self._bm25_corpus[idx],
                    "metadata": meta,
                    "bm25_score": float(scores[idx]),
                })

        # 3. Reciprocal Rank Fusion (RRF)
        def _rrf(
            dense: List[Dict], sparse: List[Dict], k: int = 60
        ) -> List[Dict]:
            rrf_scores: Dict[str, float] = {}
            registry: Dict[str, Dict] = {}
            for rank, d in enumerate(dense, start=1):
                uid = d["metadata"]["chunk_id"]
                rrf_scores[uid] = rrf_scores.get(uid, 0) + 1 / (k + rank)
                registry[uid] = d
            for rank, s in enumerate(sparse, start=1):
                uid = s["metadata"]["chunk_id"]
                rrf_scores[uid] = rrf_scores.get(uid, 0) + 1 / (k + rank)
                if uid not in registry:
                    registry[uid] = s
            ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            return [registry[uid] for uid, _ in ranked]

        fused = _rrf(dense_docs, bm25_docs)[: top_k * 2]

        if not fused:
            logger.warning(f"No documents found for query: {query}")
            return {"documents": [], "confidence": 0.0}

        # 4. Recency post-filter
        if min_date:
            cutoff = _parse_date(min_date)
            if cutoff:
                fused = [
                    d
                    for d in fused
                    if (
                        doc_date := _parse_date(
                            d["metadata"].get("last_updated")
                        )
                    )
                    and doc_date >= cutoff
                ]

        # 5. Cross-encoder reranking
        pairs = [(query, d["content"]) for d in fused]
        ce_scores = self._reranker.predict(pairs)
        for doc, score in zip(fused, ce_scores):
            doc["rerank_score"] = float(score)
        fused.sort(key=lambda d: d["rerank_score"], reverse=True)
        results = fused[:top_k]

        def _sigmoid(x: float) -> float:
            return 1 / (1 + math.exp(-x))

        for doc in results:
            doc["score"] = _sigmoid(doc["rerank_score"])

        confidence = max(d["score"] for d in results) if results else 0.0
        logger.info(
            f"✓ Retrieved {len(results)} docs | Max confidence: {confidence:.2%}"
        )
        return {"documents": results, "confidence": round(confidence, 3)}

    def refresh_document(
        self,
        jurisdiction: str,
        tax_category: str,
        new_content: str,
        new_metadata: Dict,
    ) -> int:
        """Replace all chunks for a given jurisdiction/category."""
        try:
            existing = self.collection.get(
                where={
                    "$and": [
                        {"jurisdiction": {"$eq": jurisdiction}},
                        {"tax_category": {"$eq": tax_category}},
                    ]
                }
            )
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
                logger.info(f"Deleted {len(existing['ids'])} old chunks")
        except Exception as e:
            logger.warning(f"Deletion warning: {e}")

        new_corpus: List[str] = []
        new_meta_list: List[Dict] = []
        for c, m in zip(self._bm25_corpus, self._bm25_meta):
            if not (
                m.get("jurisdiction") == jurisdiction
                and m.get("tax_category") == tax_category
            ):
                new_corpus.append(c)
                new_meta_list.append(m)
        self._bm25_corpus = new_corpus
        self._bm25_meta = new_meta_list

        doc = {
            "content": new_content,
            "metadata": {
                **new_metadata,
                "jurisdiction": jurisdiction,
                "tax_category": tax_category,
                "last_updated": new_metadata.get(
                    "last_updated", date.today().isoformat()
                ),
            },
        }
        return self.ingest([doc])

    def add_feedback(
        self,
        query: str,
        retrieved_doc_ids: List[str],
        correct: bool,
        expert_note: str = "",
    ) -> None:
        """Log expert feedback (in memory)."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_doc_ids": retrieved_doc_ids,
            "correct": correct,
            "expert_note": expert_note,
        }
        self._feedback_log.append(entry)
        logger.info(f"Feedback logged: correct={correct} | '{expert_note}'")

    def export_feedback(self, path: str = "feedback_log.json") -> None:
        import json as _json

        with open(path, "w") as f:
            _json.dump(self._feedback_log, f, indent=2)
        logger.info(f"Feedback saved to {path}")


# ─── Confidence Decider ────────────────────────────────────────────────


class ConfidenceDecider:
    """Maps retrieval confidence to agent actions."""

    LABELS = {
        "ANSWER": "High confidence – answer directly.",
        "ANSWER_WITH_CAVEAT": "Medium confidence – answer with caveat.",
        "ESCALATE": "Low confidence – refuse or escalate to human.",
    }

    def decide(self, search_result: Dict) -> Tuple[str, str]:
        conf = search_result.get("confidence", 0.0)
        if conf >= Config.HIGH_CONF:
            label = "ANSWER"
        elif conf >= Config.MEDIUM_CONF:
            label = "ANSWER_WITH_CAVEAT"
        else:
            label = "ESCALATE"

        explanation = (
            f"Confidence={conf:.2%} | Threshold: "
            f"HIGH≥{Config.HIGH_CONF:.0%}, "
            f"MEDIUM≥{Config.MEDIUM_CONF:.0%}"
        )
        logger.info(f"[Decide] {label} | {explanation}")
        return label, explanation

    def format_response(
        self, action: str, answer: str, sources: List[Dict]
    ) -> str:
        citations = "; ".join(
            m["metadata"].get("citation", "N/A") for m in sources[:3]
        )
        if action == "ANSWER":
            return f"{answer}\n\n📎 Sources: {citations}"
        elif action == "ANSWER_WITH_CAVEAT":
            return (
                f"⚠️ This answer is based on partially matched sources – "
                f"please verify:\n\n{answer}\n\n📎 Sources: {citations}"
            )
        else:
            return (
                "❌ I could not find sufficiently reliable information to "
                "answer this question. Please consult a tax professional "
                "or visit the official portal directly."
            )


# ─── Conflict Detector ─────────────────────────────────────────────────

_FIELD_PATTERNS = {
    "vat_rate": r"(?:VAT|standard)\s+rate[^\d]*(\d+(?:\.\d+)?)\s*%",
    "registration_threshold": r"(?:turnover|threshold)[^\d]*([\d,]+)",
    "filing_deadline_day": r"(?:by the|before the)\s+(\d+)(?:st|nd|rd|th)\s+day",
    "late_payment_penalty": r"late\s+payment[^\d]*(\d+(?:\.\d+)?)\s*%",
}


def _extract_fields(text: str) -> Dict[str, Optional[float]]:
    fields: Dict[str, Optional[float]] = {}
    for field, pattern in _FIELD_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                fields[field] = float(raw)
            except ValueError:
                fields[field] = None
        else:
            fields[field] = None
    return fields


class ConflictDetector:
    """Detects conflicting numeric information across retrieved documents."""

    def detect(self, documents: List[Dict]) -> List[Dict]:
        if len(documents) < 2:
            return []

        extracted: List[Tuple[str, Dict]] = []
        for doc in documents:
            fields = _extract_fields(doc["content"])
            label = (
                f"{doc['metadata'].get('jurisdiction', '?')}/"
                f"{doc['metadata'].get('tax_category', '?')} "
                f"(chunk {doc['metadata'].get('chunk_index', 0)})"
            )
            extracted.append((label, fields))

        conflicts: List[Dict] = []
        for i in range(len(extracted)):
            for j in range(i + 1, len(extracted)):
                label_a, fields_a = extracted[i]
                label_b, fields_b = extracted[j]
                for field in _FIELD_PATTERNS:
                    va, vb = fields_a.get(field), fields_b.get(field)
                    if va is None or vb is None:
                        continue
                    delta = abs(va - vb)
                    threshold = (
                        Config.CONFLICT_RATE_TOL if va < 100 else va * 0.05
                    )
                    if delta > threshold:
                        conflicts.append({
                            "field": field,
                            "doc_a": label_a,
                            "val_a": va,
                            "doc_b": label_b,
                            "val_b": vb,
                            "delta": round(delta, 4),
                        })
        if conflicts:
            logger.warning(f"⚠️  {len(conflicts)} conflict(s) detected")
        else:
            logger.info("✓ No conflicts detected")
        return conflicts

    def format_warning(self, conflicts: List[Dict]) -> str:
        if not conflicts:
            return ""
        lines = ["⚠️ **Conflicting information detected in sources:**"]
        for c in conflicts:
            lines.append(
                f"  • `{c['field']}`: {c['doc_a']} = {c['val_a']} | "
                f"{c['doc_b']} = {c['val_b']} (difference: {c['delta']})"
            )
        lines.append("Please verify with the official portal before acting.")
        return "\n".join(lines)
