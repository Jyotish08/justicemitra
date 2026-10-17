"""Sentry Agent Tracing, privacy redaction hooks, and AI observability."""

import logging
from typing import Optional, Dict, Any, Generator
from contextlib import contextmanager
import sentry_sdk
from sentry_sdk.integrations.logging import LoggingIntegration
from justicemitra.config import config, AppConfig
from justicemitra.core.sanitizer import scrub_pii

logger = logging.getLogger(__name__)

_SENTRY_INITIALIZED = False


def redact_sensitive_pii(event: Dict[str, Any], hint: Dict[str, Any]) -> Dict[str, Any]:
    """Scrubs personal identifiable information and sensitive disclosures from Sentry payloads."""
    # 1. Scrub log breadcrumbs
    if "breadcrumbs" in event:
        for breadcrumb in event.get("breadcrumbs", []):
            if "message" in breadcrumb and isinstance(breadcrumb["message"], str):
                breadcrumb["message"] = scrub_pii(breadcrumb["message"])

    # 2. Scrub log messages and formatted log entries
    if "logentry" in event and isinstance(event["logentry"], dict):
        if "formatted" in event["logentry"] and isinstance(event["logentry"]["formatted"], str):
            event["logentry"]["formatted"] = scrub_pii(event["logentry"]["formatted"])

    # 3. Scrub exception descriptions
    if "exception" in event and "values" in event["exception"]:
        for exc in event["exception"].get("values", []):
            if "value" in exc and isinstance(exc["value"], str):
                exc["value"] = scrub_pii(exc["value"])

    return event


def setup_sentry_tracing(app_config: Optional[AppConfig] = None) -> bool:
    """Initializes the Sentry SDK with strict zero-PII capture and AI Agent Tracing."""
    global _SENTRY_INITIALIZED
    cfg = app_config or config

    if not cfg.sentry_dsn:
        logger.info("SENTRY_DSN not provided; Sentry tracing is disabled.")
        return False

    try:
        sentry_sdk.init(
            dsn=cfg.sentry_dsn,
            environment=cfg.sentry_environment,
            traces_sample_rate=cfg.sentry_traces_sample_rate,
            send_default_pii=False,  # Enforce zero PII capture by default
            before_send=redact_sensitive_pii,
            integrations=[
                LoggingIntegration(
                    level=logging.INFO,
                    event_level=logging.ERROR,
                )
            ],
        )
        _SENTRY_INITIALIZED = True
        logger.info("Sentry SDK successfully initialized for Agent Tracing.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Sentry: {e}")
        return False


def is_sentry_active() -> bool:
    """Returns True if Sentry is active and initialized."""
    return _SENTRY_INITIALIZED and sentry_sdk.Hub.current.client is not None


@contextmanager
def start_query_transaction(name: str = "legal_query_interaction") -> Generator[Any, None, None]:
    """Context manager for the root end-to-end user query transaction."""
    if not is_sentry_active():
        yield None
        return

    with sentry_sdk.start_transaction(op="gen_ai.query", name=name) as transaction:
        yield transaction


@contextmanager
def start_sanitization_span() -> Generator[Any, None, None]:
    """Context manager for PII sanitization and guardrail checks."""
    if not is_sentry_active():
        yield None
        return

    with sentry_sdk.start_span(op="gen_ai.input_sanitization", description="Sanitize PII and validate guardrails") as span:
        span.set_data("privacy_scrubbed", True)
        yield span


@contextmanager
def start_retrieval_span(
    backend: str = "MongoDB Atlas Vector Search",
    index_name: str = "vector_index",
) -> Generator[Any, None, None]:
    """Context manager for vector database retrieval."""
    if not is_sentry_active():
        yield None
        return

    with sentry_sdk.start_span(op="gen_ai.retrieval", description=f"{backend}: {index_name}") as span:
        span.set_data("gen_ai.system", "mongodb_atlas")
        yield span


@contextmanager
def start_chat_span(
    model: str = "gemma2-9b-it",
    temperature: float = 0.1,
) -> Generator[Any, None, None]:
    """Context manager for Gemma LLM inference."""
    if not is_sentry_active():
        yield None
        return

    with sentry_sdk.start_span(op="gen_ai.chat", description=f"ChatCompletion {model}") as span:
        span.set_data("gen_ai.system", "gemma")
        span.set_data("gen_ai.request.model", model)
        span.set_data("gen_ai.request.temperature", temperature)
        yield span
