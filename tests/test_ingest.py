"""Unit tests for the ingestion pipeline and deduplication."""

from pathlib import Path
import tempfile
import pytest
from scripts.ingest import compute_chunk_id, compute_file_hash, parse_pdf_file


def test_deterministic_chunk_id():
    id1 = compute_chunk_id("PWDVA 2005", "Section 17", "Right to reside in shared household")
    id2 = compute_chunk_id("PWDVA 2005", "Section 17", "Right to reside in shared household")
    assert id1 == id2
    assert len(id1) == 64  # SHA-256 length


def test_chunk_id_differs_on_content_change():
    id1 = compute_chunk_id("PWDVA 2005", "Section 17", "Right to reside in shared household")
    id2 = compute_chunk_id("PWDVA 2005", "Section 17", "Modified text for residency")
    assert id1 != id2


def test_file_hash_computation():
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"Statutory PDF content")
        tf_path = Path(tf.name)

    try:
        hash1 = compute_file_hash(tf_path)
        hash2 = compute_file_hash(tf_path)
        assert hash1 == hash2
        assert len(hash1) == 64
    finally:
        tf_path.unlink()


def test_empty_pdf_handling():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
        tf.write(b"")
        tf_path = Path(tf.name)

    try:
        # Should gracefully return empty list without crashing
        chunks = parse_pdf_file(tf_path)
        assert chunks == []
    finally:
        tf_path.unlink()
