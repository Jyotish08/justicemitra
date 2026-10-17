"""Unit tests for the JusticeMitra RAG orchestration pipeline."""

from unittest.mock import MagicMock
import pytest
from justicemitra.config import AppConfig
from justicemitra.core.rag import JusticeMitraRAGPipeline
from justicemitra.services.retriever import RetrievedStatuteChunk
from justicemitra.services.generation import ModelGenerationResult, EVIDENCE_REFUSAL_MESSAGE


def test_rag_process_query_normal():
    mock_retriever = MagicMock()
    mock_retriever.backend_name = "MongoDB Atlas Vector Search"
    mock_retriever.retrieve_relevant_chunks.return_value = [
        RetrievedStatuteChunk(
            chunk_id="c1",
            statute_name="PWDVA 2005",
            section="Section 17",
            content="Every woman has the right to reside in the shared household.",
            source_url="https://www.indiacode.nic.in/handle/123456789/2021",
            similarity_score=0.85,
        )
    ]

    mock_generator = MagicMock()
    mock_generator.generate_legal_answer.return_value = ModelGenerationResult(
        text="Under Section 17 of PWDVA 2005, you cannot be evicted.",
        model_name="gemma2-9b-it",
        prompt_tokens=100,
        completion_tokens=30,
        latency_ms=250.0,
        citations=[{"statute": "PWDVA 2005", "section": "Section 17", "url": "https://www.indiacode.nic.in/"}],
    )

    cfg = AppConfig(similarity_threshold=0.60)
    pipeline = JusticeMitraRAGPipeline(retriever=mock_retriever, generator=mock_generator, app_config=cfg)

    response = pipeline.process_query("Can my in-laws kick me out?")
    assert response.is_refusal is False
    assert "Section 17" in response.answer
    assert len(response.citations) == 1
    assert response.retrieval_backend == "MongoDB Atlas Vector Search"


def test_rag_emergency_detection_and_helplines():
    mock_retriever = MagicMock()
    mock_retriever.retrieve_relevant_chunks.return_value = []
    mock_generator = MagicMock()

    pipeline = JusticeMitraRAGPipeline(retriever=mock_retriever, generator=mock_generator)
    response = pipeline.process_query("My husband has locked me out and is beating me right now help me right now")

    assert response.is_emergency is True
    assert len(response.emergency_helplines) > 0
    helpline_numbers = [h["number"] for h in response.emergency_helplines]
    assert "1091" in helpline_numbers
    assert "112" in helpline_numbers


def test_rag_refusal_on_low_similarity():
    mock_retriever = MagicMock()
    # Chunks returned with similarity score 0.40, below threshold 0.60
    mock_retriever.retrieve_relevant_chunks.return_value = [
        RetrievedStatuteChunk(
            chunk_id="c1",
            statute_name="Income Tax Act",
            section="Section 10",
            content="Exemptions on income",
            source_url="https://www.incometax.gov.in/",
            similarity_score=0.40,
        )
    ]

    mock_generator = MagicMock()
    cfg = AppConfig(similarity_threshold=0.60)
    pipeline = JusticeMitraRAGPipeline(retriever=mock_retriever, generator=mock_generator, app_config=cfg)

    response = pipeline.process_query("How do patents apply to shipping vessels?")
    assert response.is_refusal is True
    assert response.answer == EVIDENCE_REFUSAL_MESSAGE
    # Verify generator was NOT called (preventing hallucination and token waste)
    mock_generator.generate_legal_answer.assert_not_called()


def test_rag_graceful_database_failure():
    mock_retriever = MagicMock()
    mock_retriever.backend_name = "MongoDB Atlas Vector Search"
    mock_retriever.retrieve_relevant_chunks.side_effect = ConnectionError("Atlas timeout")

    mock_generator = MagicMock()
    pipeline = JusticeMitraRAGPipeline(retriever=mock_retriever, generator=mock_generator)

    response = pipeline.process_query("What are my maintenance rights?")
    assert response.is_refusal is True
    assert "currently unavailable" in response.answer
    assert response.is_emergency is True
    assert len(response.emergency_helplines) > 0
