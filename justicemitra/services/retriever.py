"""MongoDB Atlas Vector Search retriever with explicit backend identification and metadata support."""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from justicemitra.config import config, AppConfig
from justicemitra.services.embedding import EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)


@dataclass
class RetrievedStatuteChunk:
    """Represents a statutory text chunk retrieved through vector search."""
    chunk_id: str
    statute_name: str
    section: str
    content: str
    source_url: str
    similarity_score: float
    jurisdiction: str = "Central"
    document_version: str = "enacted"


class AtlasVectorRetriever:
    """Executes approximate nearest-neighbor search against MongoDB Atlas Vector Search."""

    def __init__(
        self,
        collection,
        embedder: Optional[EmbeddingService] = None,
        index_name: Optional[str] = None,
        candidates_multiplier: int = 10,
    ):
        self.collection = collection
        self.embedder = embedder or get_embedding_service()
        self.index_name = index_name or config.mongodb_vector_index
        self.candidates_multiplier = candidates_multiplier
        self.backend_name = "MongoDB Atlas Vector Search"

    def retrieve_relevant_chunks(
        self,
        query: str,
        limit: int = 4,
        statute_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedStatuteChunk]:
        """Encodes query and executes $vectorSearch aggregation pipeline in MongoDB Atlas."""
        if self.collection is None:
            raise ConnectionError("MongoDB Atlas collection is not available.")

        # 1. Generate normalized 384-dimensional query embedding
        query_vector = self.embedder.encode(query, normalize=True).tolist()

        # 2. Build $vectorSearch aggregation stage
        search_stage: Dict[str, Any] = {
            "index": self.index_name,
            "path": "embedding",
            "queryVector": query_vector,
            "numCandidates": max(limit * self.candidates_multiplier, 20),
            "limit": limit,
        }

        if statute_filter:
            search_stage["filter"] = {"statute_title": {"$eq": statute_filter}}

        pipeline = [
            {"$vectorSearch": search_stage},
            {
                "$project": {
                    "_id": 0,
                    "chunk_id": 1,
                    "source_name": 1,
                    "source_url": 1,
                    "statute_title": 1,
                    "section": 1,
                    "chunk_text": 1,
                    "jurisdiction": 1,
                    "document_version": 1,
                    "similarity_score": {"$meta": "vectorSearchScore"},
                }
            },
        ]

        try:
            results = list(self.collection.aggregate(pipeline))
        except (PyMongoError, ServerSelectionTimeoutError) as e:
            logger.error(f"Atlas Vector Search aggregation failed: {e}")
            raise

        chunks = []
        for doc in results:
            score = float(doc.get("similarity_score", 0.0))
            if score < min_score:
                continue
            chunks.append(
                RetrievedStatuteChunk(
                    chunk_id=doc.get("chunk_id", ""),
                    statute_name=doc.get("statute_title") or doc.get("source_name", "Statutory Source"),
                    section=doc.get("section", "General"),
                    content=doc.get("chunk_text", ""),
                    source_url=doc.get("source_url", ""),
                    similarity_score=score,
                    jurisdiction=doc.get("jurisdiction", "Central"),
                    document_version=doc.get("document_version", "enacted"),
                )
            )

        return chunks


class LocalFallbackRetriever:
    """Explicit in-memory vector search retriever for offline local development and testing.
    
    Explicitly labeled as 'Local In-Memory Fallback' so callers and UI are aware Atlas is not used.
    """

    def __init__(
        self,
        documents: Optional[List[Dict[str, Any]]] = None,
        embedder: Optional[EmbeddingService] = None,
    ):
        self.embedder = embedder or get_embedding_service()
        self.documents = documents or []
        self.backend_name = "Local In-Memory Fallback (Offline)"

    def retrieve_relevant_chunks(
        self,
        query: str,
        limit: int = 4,
        statute_filter: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[RetrievedStatuteChunk]:
        if not self.documents:
            return []

        import numpy as np

        query_vec = self.embedder.encode(query, normalize=True)
        scored_docs = []

        for doc in self.documents:
            if statute_filter and doc.get("statute_title") != statute_filter:
                continue
            emb = doc.get("embedding")
            if emb is None:
                continue
            doc_vec = np.array(emb)
            # Cosine similarity for normalized vectors is the dot product
            sim = float(np.dot(query_vec, doc_vec))
            if sim >= min_score:
                scored_docs.append((sim, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = scored_docs[:limit]

        chunks = []
        for score, doc in top_docs:
            chunks.append(
                RetrievedStatuteChunk(
                    chunk_id=doc.get("chunk_id", ""),
                    statute_name=doc.get("statute_title") or doc.get("source_name", "Statutory Source"),
                    section=doc.get("section", "General"),
                    content=doc.get("chunk_text", ""),
                    source_url=doc.get("source_url", ""),
                    similarity_score=score,
                    jurisdiction=doc.get("jurisdiction", "Central"),
                    document_version=doc.get("document_version", "enacted"),
                )
            )
        return chunks
