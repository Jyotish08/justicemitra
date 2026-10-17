"""Unit tests for the embedding service."""

import numpy as np
import pytest
from justicemitra.services.embedding import get_embedding_service, EmbeddingService


def test_embedding_service_singleton():
    svc1 = get_embedding_service()
    svc2 = get_embedding_service()
    assert svc1 is svc2


def test_embedding_dimensions():
    svc = get_embedding_service()
    vec = svc.encode("Protection of Women from Domestic Violence Act")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    assert svc.dimension == 384


def test_embedding_normalization():
    svc = get_embedding_service()
    vec = svc.encode("Workplace sexual harassment prevention", normalize=True)
    norm = np.linalg.norm(vec)
    # Normalized vector has L2 norm close to 1.0
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_batch_embedding():
    svc = get_embedding_service()
    texts = ["Right to reside in shared household", "Dowry demand penalty"]
    vecs = svc.encode(texts, normalize=True)
    assert isinstance(vecs, np.ndarray)
    assert vecs.shape == (2, 384)
