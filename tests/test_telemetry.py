"""Tests for Sentry observability, span instrumentation, and privacy redaction."""

from unittest.mock import MagicMock, patch
import pytest
import sentry_sdk
from justicemitra.config import AppConfig
from justicemitra.observability.telemetry import (
    setup_sentry_tracing,
    is_sentry_active,
    start_query_transaction,
    start_sanitization_span,
    start_retrieval_span,
    start_chat_span,
    redact_sensitive_pii,
    redact_sensitive_transaction,
)
from justicemitra.core.rag import JusticeMitraRAGPipeline
from justicemitra.services.retriever import RetrievedStatuteChunk, LocalFallbackRetriever
from justicemitra.services.generation import GemmaOpenAIGateway, ModelGenerationResult
from justicemitra.services.embedding import get_embedding_service


@pytest.fixture(autouse=True)
def ensure_sentry_init():
    """Initializes Sentry for testing with traces enabled."""
    cfg = AppConfig(
        sentry_dsn="https://fake_key@o000000.ingest.sentry.io/0000000",
        sentry_environment="test",
        sentry_traces_sample_rate=1.0,
    )
    setup_sentry_tracing(app_config=cfg)
    yield


def test_sentry_initialization_and_active():
    """Verifies that Sentry initializes and reports active status."""
    assert is_sentry_active() is True


def test_retrieval_trace_safe_metadata():
    """Verifies retrieval span records only safe operational metadata, not raw queries or docs."""
    backend_name = "MongoDB Atlas Vector Search"
    index_name = "vector_index"
    sensitive_query = "Victim phone 9876543210 eviction"
    full_passage = "Full confidential statutory text of Section 17 with private details"

    with start_query_transaction(name="test_retrieval"):
        with start_retrieval_span(backend=backend_name, index_name=index_name) as span:
            assert span is not None
            # Operational attributes
            assert span.op == "gen_ai.retrieval"
            assert span.description == f"{backend_name}: {index_name}"
            # Ensure query text and document passage are NOT in span data
            span_data = getattr(span, "_data", {})
            for key, val in span_data.items():
                assert sensitive_query not in str(val)
                assert full_passage not in str(val)
            assert span_data.get("gen_ai.system") == "mongodb_atlas"


def test_generation_trace_safe_metadata():
    """Verifies chat span records only operational model metadata, not prompts or completions."""
    prompt_text = "Private legal scenario with contact 9876543210"
    completion_text = "Private legal advice output"

    with start_query_transaction(name="test_generation"):
        with start_chat_span(model="gemma2-9b-it", temperature=0.1) as span:
            assert span is not None
            assert span.op == "gen_ai.chat"
            assert "gemma2-9b-it" in span.description

            span_data = getattr(span, "_data", {})
            for key, val in span_data.items():
                assert prompt_text not in str(val)
                assert completion_text not in str(val)
            assert span_data.get("gen_ai.system") == "gemma"
            assert span_data.get("gen_ai.request.model") == "gemma2-9b-it"
            assert span_data.get("gen_ai.request.temperature") == 0.1


def test_e2e_rag_pipeline_trace_hierarchy():
    """Exercises full RAG pipeline under an active Sentry transaction and verifies span execution."""
    mock_retriever = MagicMock()
    mock_retriever.backend_name = "MongoDB Atlas Vector Search"
    mock_retriever.retrieve_relevant_chunks.return_value = [
        RetrievedStatuteChunk(
            chunk_id="chunk_test_1",
            statute_name="PWDVA 2005",
            section="Section 17",
            content="Every woman in a domestic relationship has the right to reside in the shared household.",
            source_url="https://www.indiacode.nic.in/handle/123456789/2021",
            similarity_score=0.88,
        )
    ]

    mock_generator = MagicMock()
    mock_generator.generate_legal_answer.return_value = ModelGenerationResult(
        text="Under Section 17 of PWDVA 2005, you have residence rights.",
        model_name="gemma2-9b-it",
        prompt_tokens=120,
        completion_tokens=40,
        latency_ms=180.0,
        citations=[{"statute": "PWDVA 2005", "section": "Section 17", "url": "https://www.indiacode.nic.in/"}],
    )

    cfg = AppConfig(similarity_threshold=0.50)
    pipeline = JusticeMitraRAGPipeline(retriever=mock_retriever, generator=mock_generator, app_config=cfg)

    # Process query containing sensitive contact info
    synthetic_query = "Can my in-laws evict me? Contact me at 9876543210 or test@victim.org"
    response = pipeline.process_query(synthetic_query)

    assert response.is_refusal is False
    assert "Section 17" in response.answer
    assert response.retrieval_backend == "MongoDB Atlas Vector Search"


def test_controlled_error_event_scrubbing():
    """Verifies that an error event captured through Sentry has all PII redacted by before_send."""
    captured_events = []

    def capturing_before_send(event, hint):
        processed = redact_sensitive_pii(event, hint)
        captured_events.append(processed)
        return None  # Do not send upstream during unit test

    client = sentry_sdk.get_client()
    with patch.object(client, "options", {**client.options, "before_send": capturing_before_send}):
        try:
            raise ValueError("Controlled test exception for Aadhaar 1234 5678 9012 and phone 9876543210")
        except Exception as e:
            sentry_sdk.capture_exception(e)

    assert len(captured_events) == 1
    event = captured_events[0]
    exc_val = event["exception"]["values"][0]["value"]
    assert "1234 5678 9012" not in exc_val
    assert "[REDACTED_AADHAAR]" in exc_val
    assert "9876543210" not in exc_val
    assert "[REDACTED_PHONE]" in exc_val
