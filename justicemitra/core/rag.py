"""RAG orchestration: context assembly, grounding rules, emergency handling, and telemetry."""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from justicemitra.config import config, AppConfig
from justicemitra.core.sanitizer import scrub_pii, detect_emergency, EMERGENCY_HELPLINES
from justicemitra.services.retriever import RetrievedStatuteChunk, AtlasVectorRetriever
from justicemitra.services.generation import (
    LegalGenerationGateway,
    GemmaOpenAIGateway,
    ModelGenerationResult,
    EVIDENCE_REFUSAL_MESSAGE,
)
from justicemitra.observability.telemetry import (
    start_query_transaction,
    start_sanitization_span,
    start_retrieval_span,
    start_chat_span,
)

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Consolidated response object returned by the JusticeMitra RAG pipeline."""
    answer: str
    citations: List[Dict[str, str]] = field(default_factory=list)
    retrieved_chunks: List[RetrievedStatuteChunk] = field(default_factory=list)
    is_emergency: bool = False
    emergency_helplines: List[Dict[str, str]] = field(default_factory=list)
    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    retrieval_backend: str = "MongoDB Atlas Vector Search"
    model_name: str = "gemma2-9b-it"
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class JusticeMitraRAGPipeline:
    """Coordinates PII scrubbing, vector retrieval, grounded generation, and telemetry."""

    def __init__(
        self,
        retriever,
        generator: Optional[LegalGenerationGateway] = None,
        app_config: Optional[AppConfig] = None,
    ):
        self.cfg = app_config or config
        self.retriever = retriever
        self.generator = generator or GemmaOpenAIGateway(app_config=self.cfg)

    def process_query(
        self,
        query: str,
        statute_filter: Optional[str] = None,
    ) -> RAGResponse:
        """Executes the full RAG workflow for a user inquiry."""
        start_overall = time.perf_counter()

        with start_query_transaction(name="legal_query_interaction"):
            # 1. PII Sanitization and Emergency Guardrails
            with start_sanitization_span():
                sanitized_query = scrub_pii(query)
                is_emergency, _ = detect_emergency(query)
                helplines = EMERGENCY_HELPLINES if is_emergency else []

            # 2. Retrieval Execution
            retrieval_backend = getattr(self.retriever, "backend_name", "Retriever")
            retrieved_chunks: List[RetrievedStatuteChunk] = []

            with start_retrieval_span(backend=retrieval_backend):
                try:
                    retrieved_chunks = self.retriever.retrieve_relevant_chunks(
                        query=sanitized_query,
                        limit=self.cfg.retrieval_top_k,
                        statute_filter=statute_filter,
                        min_score=0.0,
                    )
                except Exception as e:
                    logger.error(f"Retrieval error: {e}")
                    # Return graceful failure without exposing stack trace
                    latency = (time.perf_counter() - start_overall) * 1000.0
                    return RAGResponse(
                        answer=(
                            "The legal database service is currently unavailable. Please verify connectivity "
                            "or reach out to the emergency helpline numbers below for immediate assistance."
                        ),
                        is_emergency=is_emergency or True,
                        emergency_helplines=EMERGENCY_HELPLINES,
                        is_refusal=True,
                        refusal_reason="Database connection failure",
                        retrieval_backend=retrieval_backend,
                        model_name=self.cfg.gemma_model_name,
                        latency_ms=latency,
                    )

            # 3. Grounding & Refusal Evaluation
            # Filter chunks based on similarity threshold
            valid_chunks = [
                c for c in retrieved_chunks
                if c.similarity_score >= self.cfg.similarity_threshold
            ]

            if not valid_chunks:
                logger.info(
                    f"No chunks exceeded threshold ({self.cfg.similarity_threshold}). "
                    f"Top score: {retrieved_chunks[0].similarity_score if retrieved_chunks else 0.0}. Refusing generation."
                )
                latency = (time.perf_counter() - start_overall) * 1000.0
                return RAGResponse(
                    answer=EVIDENCE_REFUSAL_MESSAGE,
                    citations=[],
                    retrieved_chunks=retrieved_chunks,
                    is_emergency=is_emergency,
                    emergency_helplines=helplines,
                    is_refusal=True,
                    refusal_reason="No relevant statutory provisions found in database",
                    retrieval_backend=retrieval_backend,
                    model_name=self.cfg.gemma_model_name,
                    latency_ms=latency,
                )

            # 4. Gemma Model Generation
            with start_chat_span(
                model=self.cfg.gemma_model_name,
                temperature=self.cfg.gemma_temperature,
            ):
                gen_result: ModelGenerationResult = self.generator.generate_legal_answer(
                    query=sanitized_query,
                    context=valid_chunks,
                )

            total_latency = (time.perf_counter() - start_overall) * 1000.0

            return RAGResponse(
                answer=gen_result.text,
                citations=gen_result.citations,
                retrieved_chunks=valid_chunks,
                is_emergency=is_emergency,
                emergency_helplines=helplines,
                is_refusal=gen_result.is_refusal,
                refusal_reason=gen_result.refusal_reason,
                retrieval_backend=retrieval_backend,
                model_name=gen_result.model_name,
                latency_ms=total_latency,
                prompt_tokens=gen_result.prompt_tokens,
                completion_tokens=gen_result.completion_tokens,
            )
