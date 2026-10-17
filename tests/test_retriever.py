"""Unit tests for Atlas Vector Retriever and Local Fallback Retriever."""

from unittest.mock import MagicMock
import pytest
from justicemitra.services.retriever import AtlasVectorRetriever, LocalFallbackRetriever, RetrievedStatuteChunk
from justicemitra.services.embedding import get_embedding_service


def test_atlas_vector_retriever_pipeline_construction():
    mock_collection = MagicMock()
    mock_collection.aggregate.return_value = [
        {
            "chunk_id": "c1",
            "statute_title": "PWDVA 2005",
            "section": "Section 17",
            "chunk_text": "Right to reside in shared household",
            "source_url": "https://www.indiacode.nic.in/handle/123456789/2021",
            "similarity_score": 0.88,
            "jurisdiction": "Central",
            "document_version": "enacted",
        }
    ]

    embedder = get_embedding_service()
    retriever = AtlasVectorRetriever(
        collection=mock_collection,
        embedder=embedder,
        index_name="vector_index",
        candidates_multiplier=10,
    )

    chunks = retriever.retrieve_relevant_chunks(
        query="Can I stay in my shared home?",
        limit=3,
        statute_filter="PWDVA 2005",
    )

    assert len(chunks) == 1
    assert chunks[0].statute_name == "PWDVA 2005"
    assert chunks[0].section == "Section 17"
    assert chunks[0].similarity_score == 0.88

    # Verify aggregation was called with $vectorSearch
    call_args = mock_collection.aggregate.call_args[0][0]
    vector_stage = call_args[0]["$vectorSearch"]
    assert vector_stage["index"] == "vector_index"
    assert vector_stage["path"] == "embedding"
    assert len(vector_stage["queryVector"]) == 384
    assert vector_stage["limit"] == 3
    assert vector_stage["filter"] == {"statute_title": {"$eq": "PWDVA 2005"}}


def test_local_fallback_retriever():
    embedder = get_embedding_service()
    doc1_vec = embedder.encode("Domestic violence and right to reside in shared household", normalize=True).tolist()
    doc2_vec = embedder.encode("Income tax deduction on home loan interest", normalize=True).tolist()

    mock_docs = [
        {
            "chunk_id": "c1",
            "statute_title": "PWDVA 2005",
            "section": "Section 17",
            "chunk_text": "Domestic violence and right to reside in shared household",
            "source_url": "https://www.indiacode.nic.in/",
            "embedding": doc1_vec,
        },
        {
            "chunk_id": "c2",
            "statute_title": "Income Tax Act",
            "section": "Section 24",
            "chunk_text": "Income tax deduction on home loan interest",
            "source_url": "https://www.incometax.gov.in/",
            "embedding": doc2_vec,
        },
    ]

    retriever = LocalFallbackRetriever(documents=mock_docs, embedder=embedder)
    assert "Fallback" in retriever.backend_name

    results = retriever.retrieve_relevant_chunks(
        query="shared matrimonial home domestic abuse",
        limit=2,
    )
    assert len(results) > 0
    # Top result should be PWDVA 2005
    assert results[0].statute_name == "PWDVA 2005"
    assert results[0].similarity_score > results[1].similarity_score
