"""Verification script for JusticeMitra Sentry integration.

Exercises:
1. Synthetic retrieval trace
2. Synthetic generation trace
3. Synthetic end-to-end RAG pipeline trace
4. Controlled test error event capture
5. Sentry network flush to ensure event delivery
"""

import sys
import logging
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sentry_sdk
from justicemitra.config import config
from justicemitra.observability.telemetry import (
    setup_sentry_tracing,
    is_sentry_active,
    start_query_transaction,
    start_sanitization_span,
    start_retrieval_span,
    start_chat_span,
)
from justicemitra.core.rag import JusticeMitraRAGPipeline
from justicemitra.services.embedding import get_embedding_service
from justicemitra.services.retriever import LocalFallbackRetriever
from justicemitra.services.generation import GemmaOpenAIGateway
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentry_verification")


class JusticeMitraControlledVerificationError(Exception):
    """Safe, controlled test exception for Sentry delivery verification."""
    pass


def run_verification():
    logger.info("Step 1: Initializing Sentry SDK...")
    initialized = setup_sentry_tracing()
    if not initialized or not is_sentry_active():
        logger.error("Sentry failed to initialize or is not active. Check SENTRY_DSN configuration.")
        return False
    logger.info("Sentry SDK successfully initialized and active.")

    logger.info("Step 2: Executing synthetic retrieval trace...")
    with start_query_transaction(name="verification_retrieval_trace"):
        with start_retrieval_span(backend="Local In-Memory Fallback", index_name="seed_corpus") as span:
            span.set_data("synthetic_verification", True)
    logger.info("Retrieval trace executed successfully without exposing query or doc contents.")

    logger.info("Step 3: Executing synthetic generation trace...")
    with start_query_transaction(name="verification_generation_trace"):
        with start_chat_span(model=config.gemma_model_name, temperature=config.gemma_temperature) as span:
            span.set_data("synthetic_verification", True)
    logger.info("Generation trace executed successfully without exposing prompts or completions.")

    logger.info("Step 4: Executing synthetic end-to-end RAG pipeline trace...")
    corpus_path = ROOT / "data" / "legal_corpus.json"
    corpus = []
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

    embedder = get_embedding_service()
    retriever = LocalFallbackRetriever(documents=corpus, embedder=embedder)
    generator = GemmaOpenAIGateway(app_config=config)
    pipeline = JusticeMitraRAGPipeline(retriever=retriever, generator=generator, app_config=config)

    synthetic_query = "What are the protection orders available under Section 18 of the PWDVA 2005?"
    response = pipeline.process_query(synthetic_query)
    logger.info(f"RAG trace executed successfully (is_refusal={response.is_refusal}, latency={response.latency_ms:.1f}ms).")

    logger.info("Step 5: Capturing controlled error event...")
    try:
        raise JusticeMitraControlledVerificationError(
            "JusticeMitra Sentry Integration Verification: Controlled test event for issue tracking verification."
        )
    except JusticeMitraControlledVerificationError as exc:
        event_id = sentry_sdk.capture_exception(exc)
        logger.info(f"Controlled exception captured with Event ID: {event_id}")

    logger.info("Step 6: Flushing Sentry event queue...")
    client = sentry_sdk.get_client()
    flushed = client.flush(timeout=10.0)
    logger.info("Sentry event flush completed.")

    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
