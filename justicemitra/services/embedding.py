"""Singleton embedding service wrapping sentence-transformers for 384-dimensional dense vectors."""

import logging
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)

_GLOBAL_EMBEDDER = None


class EmbeddingService:
    """Singleton service for generating consistent sentence embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, device="cpu", local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name, device="cpu")
        self.dimension = 384

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """Encodes text or list of texts into normalized embedding vectors.
        
        Args:
            texts: Single string or list of strings.
            normalize: If True, vectors are L2-normalized for cosine similarity.
            
        Returns:
            numpy.ndarray of shape (384,) or (N, 384).
        """
        is_single = isinstance(texts, str)
        input_texts = [texts] if is_single else texts
        
        embeddings = self.model.encode(
            input_texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        
        if is_single:
            return embeddings[0]
        return embeddings


def get_embedding_service(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> EmbeddingService:
    """Returns a singleton instance of the EmbeddingService."""
    global _GLOBAL_EMBEDDER
    if _GLOBAL_EMBEDDER is None:
        _GLOBAL_EMBEDDER = EmbeddingService(model_name=model_name)
    return _GLOBAL_EMBEDDER
