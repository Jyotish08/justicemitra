"""Standardized Legal Query Regression Suite and Evaluation Pipeline.

Validates Q1 through Q5, out-of-corpus refusal, Recall@3, and privacy redaction.
"""

import json
from pathlib import Path
import pytest
from justicemitra.config import AppConfig
from justicemitra.core.rag import JusticeMitraRAGPipeline
from justicemitra.core.sanitizer import scrub_pii
from justicemitra.services.retriever import LocalFallbackRetriever
from justicemitra.services.embedding import get_embedding_service
from justicemitra.services.generation import GemmaOpenAIGateway, EVIDENCE_REFUSAL_MESSAGE
from justicemitra.observability.telemetry import redact_sensitive_pii, redact_sensitive_transaction


@pytest.fixture(scope="module")
def legal_corpus():
    """Loads the pre-ingested legal corpus from data/legal_corpus.json."""
    corpus_path = Path(__file__).resolve().parent.parent / "data" / "legal_corpus.json"
    assert corpus_path.exists(), "legal_corpus.json must exist before running regression tests."
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rag_pipeline(legal_corpus):
    """Sets up the RAG pipeline using the verified legal corpus."""
    embedder = get_embedding_service()
    retriever = LocalFallbackRetriever(documents=legal_corpus, embedder=embedder)
    cfg = AppConfig(similarity_threshold=0.40)
    generator = GemmaOpenAIGateway(app_config=cfg)
    return JusticeMitraRAGPipeline(retriever=retriever, generator=generator, app_config=cfg)


def test_q1_domestic_violence_residence_rights(rag_pipeline):
    """Q1: Can a woman stay in her matrimonial home after filing a domestic violence complaint?
    
    Target: PWDVA 2005, Section 17 & 19 (Recall@3).
    """
    query = "Can a woman stay in her matrimonial home after filing a domestic violence complaint?"
    response = rag_pipeline.process_query(query)

    assert response.is_refusal is False
    assert len(response.retrieved_chunks) > 0

    top_3_chunks = response.retrieved_chunks[:3]
    top_3_statutes = [c.statute_name for c in top_3_chunks]
    top_3_sections = [c.section for c in top_3_chunks]

    # Verify PWDVA 2005 appears in top 3
    assert any("PWDVA" in s or "Domestic Violence" in s for s in top_3_statutes)
    # Verify Section 17 or Section 19 appears in top 3
    assert any("17" in sec or "19" in sec for sec in top_3_sections)
    # Verify cosine similarity is high (> 0.65)
    assert top_3_chunks[0].similarity_score >= 0.65


def test_q2_informal_dowry_demands(rag_pipeline):
    """Q2: My husband's family is demanding cash and vehicles before our wedding.
    
    Target: Dowry Prohibition Act 1961 / IPC 498A (Recall@3).
    """
    query = "My husband's family is demanding cash and vehicles before our wedding."
    response = rag_pipeline.process_query(query)

    assert response.is_refusal is False
    assert len(response.retrieved_chunks) > 0

    top_3_chunks = response.retrieved_chunks[:3]
    top_3_statutes = [c.statute_name for c in top_3_chunks]

    # Verify Dowry Prohibition Act or IPC 498A / cruelty appears in top 3
    assert any("Dowry" in s or "498A" in s or "IPC" in s for s in top_3_statutes)


def test_q3_free_legal_aid(rag_pipeline):
    """Q3: Where can an indigent woman obtain a free advocate for court representation?
    
    Target: Legal Services Authorities Act 1987, Sec 12(c) / NALSA (Recall@3).
    """
    query = "Where can an indigent woman obtain a free advocate for court representation?"
    response = rag_pipeline.process_query(query)

    assert response.is_refusal is False
    assert len(response.retrieved_chunks) > 0

    top_3_chunks = response.retrieved_chunks[:3]
    top_3_texts = [c.content for c in top_3_chunks]

    # Verify free legal aid or Legal Services Authorities Act appears in top 3
    assert any("Legal Services" in t or "12(c)" in t or "free legal" in t for t in top_3_texts)


def test_q4_out_of_corpus_guardrail_refusal(rag_pipeline):
    """Q4: What are the rules for filing a patent infringement claim under maritime law?
    
    Target: Out-of-Corpus -> 100% refusal rate.
    """
    query = "What are the rules for filing a patent infringement claim under maritime law?"
    response = rag_pipeline.process_query(query)

    # Must be an explicit evidence refusal
    assert response.is_refusal is True
    assert response.answer == EVIDENCE_REFUSAL_MESSAGE


def test_q5_workplace_harassment_limitation_period(rag_pipeline):
    """Q5: Can a workplace harassment complaint be filed if the incident occurred five months ago?
    
    Target: POSH Act 2013, Section 9 (Recall@3).
    """
    query = "Can a workplace harassment complaint be filed if the incident occurred five months ago?"
    response = rag_pipeline.process_query(query)

    assert response.is_refusal is False
    assert len(response.retrieved_chunks) > 0

    top_3_chunks = response.retrieved_chunks[:3]
    top_3_statutes = [c.statute_name for c in top_3_chunks]
    top_3_sections = [c.section for c in top_3_chunks]

    # Verify POSH Act appears in top 3
    assert any("POSH" in s or "Sexual Harassment" in s for s in top_3_statutes)
    # Verify Section 9 appears
    assert any("9" in sec for sec in top_3_sections)


def test_sensitive_telemetry_redaction():
    """Validates that personal identifying disclosures are scrubbed from telemetry payloads."""
    sensitive_payload = {
        "breadcrumbs": [
            {"message": "User query: Ramesh (phone 9876543210) harassed me"},
        ],
        "logentry": {
            "formatted": "Error processing user Aadhaar 1234 5678 9012 at runtime",
        },
        "exception": {
            "values": [
                {"value": "Failed on record with contact +91-9876543210"},
            ]
        },
    }

    redacted = redact_sensitive_pii(sensitive_payload, hint={})

    # Assert no phone or Aadhaar remains in breadcrumbs, logs, or exceptions
    assert "9876543210" not in redacted["breadcrumbs"][0]["message"]
    assert "[REDACTED_PHONE]" in redacted["breadcrumbs"][0]["message"]

    assert "1234 5678 9012" not in redacted["logentry"]["formatted"]
    assert "[REDACTED_AADHAAR]" in redacted["logentry"]["formatted"]

    assert "9876543210" not in redacted["exception"]["values"][0]["value"]
    assert "[REDACTED_PHONE]" in redacted["exception"]["values"][0]["value"]


def test_sensitive_telemetry_redaction_sdk_dict_breadcrumbs():
    """Validates that real Sentry SDK breadcrumb dictionary format is scrubbed."""
    payload = {
        "breadcrumbs": {
            "values": [
                {
                    "message": "User query: Victim with Aadhaar 9999 8888 7777 reported incident",
                    "data": {"contact": "+91-9876543210", "email": "help@victimsupport.org"},
                }
            ]
        },
        "logentry": {
            "message": "Processing user email confidential@domain.com",
        },
        "message": "Uncaught alert for 9876543210",
    }

    redacted = redact_sensitive_pii(payload, hint={})

    crumb = redacted["breadcrumbs"]["values"][0]
    assert "9999 8888 7777" not in crumb["message"]
    assert "[REDACTED_AADHAAR]" in crumb["message"]
    assert "9876543210" not in crumb["data"]["contact"]
    assert "[REDACTED_PHONE]" in crumb["data"]["contact"]
    assert "help@victimsupport.org" not in crumb["data"]["email"]
    assert "[REDACTED_EMAIL]" in crumb["data"]["email"]
    assert "confidential@domain.com" not in redacted["logentry"]["message"]
    assert "[REDACTED_EMAIL]" in redacted["logentry"]["message"]
    assert "9876543210" not in redacted["message"]
    assert "[REDACTED_PHONE]" in redacted["message"]


def test_sensitive_transaction_redaction():
    """Validates that transaction span descriptions and metadata are scrubbed of PII."""
    tx_payload = {
        "spans": [
            {
                "description": "User case search for victim phone 9876543210",
                "data": {"query_snippet": "Aadhaar 1234 5678 9012 verification"},
            }
        ]
    }

    redacted = redact_sensitive_transaction(tx_payload, hint={})
    span = redacted["spans"][0]
    assert "9876543210" not in span["description"]
    assert "[REDACTED_PHONE]" in span["description"]
    assert "1234 5678 9012" not in span["data"]["query_snippet"]
    assert "[REDACTED_AADHAAR]" in span["data"]["query_snippet"]
