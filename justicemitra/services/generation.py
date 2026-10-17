"""Generation service providing provider-agnostic interface and concrete Gemma gateway."""

import os
import time
import logging
from typing import Protocol, List, Dict, Any, Optional
from dataclasses import dataclass, field
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError
from justicemitra.config import config, AppConfig
from justicemitra.services.retriever import RetrievedStatuteChunk

logger = logging.getLogger(__name__)


@dataclass
class ModelGenerationResult:
    """Encapsulates the output of legal answer generation along with usage and citations."""
    text: str
    model_name: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    citations: List[Dict[str, str]] = field(default_factory=list)
    is_refusal: bool = False
    refusal_reason: Optional[str] = None


class LegalGenerationGateway(Protocol):
    """Protocol defining the generation interface decoupled from any specific vendor."""

    def generate_legal_answer(
        self,
        query: str,
        context: List[RetrievedStatuteChunk],
        system_instructions: Optional[str] = None,
    ) -> ModelGenerationResult:
        """Executes grounded legal synthesis using the configured model gateway."""
        ...


STANDARD_SYSTEM_PROMPT = (
    "You are JusticeMitra, a specialized legal information assistant for women in India.\n"
    "Your mandate is to provide accurate, empathetic, and strictly grounded legal information based solely "
    "on the provided statutory context.\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "1. Base your answer strictly on the statutory excerpts provided in the context.\n"
    "2. If the provided context does not contain sufficient evidence or does not address the question, "
    "you MUST explicitly state that statutory evidence is unavailable in the database, and advise consulting "
    "a qualified advocate or visiting a District Legal Services Authority (DLSA) clinic.\n"
    "3. Never invent legal rights, sections, acts, remedies, or procedural timelines not found in the context.\n"
    "4. Cite the relevant Acts and Sections accurately wherever mentioned.\n"
    "5. Clarify that this information is for educational/informational purposes and does not substitute for formal legal counsel."
)

EVIDENCE_REFUSAL_MESSAGE = (
    "I could not locate sufficient statutory provisions or legal evidence in the database to answer your specific inquiry. "
    "To ensure your rights are properly protected, please consult a qualified advocate or reach out to the "
    "National Legal Services Authority (NALSA Helpline: 1516 / DLSA) for free and confidential legal assistance."
)


class GemmaOpenAIGateway:
    """Concrete generation gateway targeting Google Gemma instruction-tuned models via OpenAI-compatible endpoints."""

    def __init__(self, app_config: Optional[AppConfig] = None):
        self.cfg = app_config or config
        self.model = self.cfg.gemma_model_name
        self.api_base = self.cfg.gemma_api_base
        self.api_key = self.cfg.gemma_api_key
        self.timeout = self.cfg.gemma_timeout_seconds

        self._client: Optional[OpenAI] = None
        if self.api_key:
            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
                timeout=self.timeout,
            )

    def generate_legal_answer(
        self,
        query: str,
        context: List[RetrievedStatuteChunk],
        system_instructions: Optional[str] = None,
    ) -> ModelGenerationResult:
        """Constructs grounded prompt and invokes Gemma model with error handling."""
        start_time = time.perf_counter()

        # Build citations list from retrieved context
        citations = [
            {
                "statute": c.statute_name,
                "section": c.section,
                "url": c.source_url,
                "score": f"{c.similarity_score:.3f}",
            }
            for c in context
        ]

        # Guardrail: Check for empty or insufficient context
        if not context:
            logger.info("Empty context passed to generation service. Returning evidence refusal.")
            latency = (time.perf_counter() - start_time) * 1000.0
            return ModelGenerationResult(
                text=EVIDENCE_REFUSAL_MESSAGE,
                model_name=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency,
                citations=[],
                is_refusal=True,
                refusal_reason="Insufficient context retrieved",
            )

        # Context assembly with explicit statutory citation boundaries
        context_blocks = []
        for i, c in enumerate(context, 1):
            context_blocks.append(
                f"[Source {i}: {c.statute_name} - {c.section} (Jurisdiction: {c.jurisdiction})]\n{c.content}"
            )
        context_string = "\n\n".join(context_blocks)

        system_prompt = system_instructions or STANDARD_SYSTEM_PROMPT
        user_prompt = f"Statutory Context:\n{context_string}\n\nUser Legal Inquiry: {query}"

        # If client is not configured (e.g. no API key provided in dev environment)
        if not self._client:
            logger.warning("Gemma client has no API key configured. Returning mock/offline synthesis.")
            latency = (time.perf_counter() - start_time) * 1000.0
            mock_text = (
                f"**Information retrieved from {context[0].statute_name} ({context[0].section}):**\n\n"
                f"{context[0].content[:400]}...\n\n"
                f"*(Note: Gemma API key is not configured in local environment; displaying retrieved statutory excerpt directly.)*"
            )
            return ModelGenerationResult(
                text=mock_text,
                model_name=self.model,
                prompt_tokens=len(user_prompt.split()),
                completion_tokens=len(mock_text.split()),
                latency_ms=latency,
                citations=citations,
                is_refusal=False,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.cfg.gemma_temperature,
                max_tokens=self.cfg.gemma_max_tokens,
            )
            latency = (time.perf_counter() - start_time) * 1000.0
            usage = response.usage
            content = response.choices[0].message.content or ""

            return ModelGenerationResult(
                text=content,
                model_name=self.model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
                latency_ms=latency,
                citations=citations,
            )

        except RateLimitError as e:
            logger.error(f"Gemma inference rate limit exceeded (HTTP 429): {e}")
            latency = (time.perf_counter() - start_time) * 1000.0
            return ModelGenerationResult(
                text="The legal inference service is currently experiencing high demand. Please wait a few moments and try your request again.",
                model_name=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency,
                citations=citations,
                is_refusal=True,
                refusal_reason="Rate limit exceeded",
            )
        except APIConnectionError as e:
            logger.error(f"Failed to connect to Gemma inference endpoint: {e}")
            latency = (time.perf_counter() - start_time) * 1000.0
            return ModelGenerationResult(
                text="Unable to connect to the legal reasoning engine. Please verify network connectivity or check if the inference endpoint is active.",
                model_name=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency,
                citations=citations,
                is_refusal=True,
                refusal_reason="Connection error",
            )
        except APIStatusError as e:
            logger.error(f"Gemma endpoint returned API error ({e.status_code}): {e.message}")
            latency = (time.perf_counter() - start_time) * 1000.0
            return ModelGenerationResult(
                text=f"An error occurred while synthesizing the legal answer (Status code {e.status_code}). Please try again shortly.",
                model_name=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency,
                citations=citations,
                is_refusal=True,
                refusal_reason=f"API error {e.status_code}",
            )
        except Exception as e:
            logger.error(f"Unexpected error during Gemma generation: {e}")
            latency = (time.perf_counter() - start_time) * 1000.0
            return ModelGenerationResult(
                text="An unexpected error occurred during answer generation. Please check the emergency resources below if urgent legal assistance is needed.",
                model_name=self.model,
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=latency,
                citations=citations,
                is_refusal=True,
                refusal_reason=str(e),
            )
