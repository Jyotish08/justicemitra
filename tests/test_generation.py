"""Unit tests for Gemma generation gateway and provider abstraction."""

from unittest.mock import MagicMock, patch
import pytest
from openai import RateLimitError
from justicemitra.config import AppConfig
from justicemitra.services.retriever import RetrievedStatuteChunk
from justicemitra.services.generation import (
    GemmaOpenAIGateway,
    EVIDENCE_REFUSAL_MESSAGE,
    STANDARD_SYSTEM_PROMPT,
)


def test_empty_context_refusal():
    gateway = GemmaOpenAIGateway(app_config=AppConfig(gemma_api_key="mock-key"))
    result = gateway.generate_legal_answer(
        query="What are the patent rules for maritime vessels?",
        context=[],
    )
    assert result.is_refusal is True
    assert result.text == EVIDENCE_REFUSAL_MESSAGE
    assert len(result.citations) == 0


def test_generation_offline_mode():
    # When api_key is empty, it returns the mock/offline synthesis gracefully
    gateway = GemmaOpenAIGateway(app_config=AppConfig(gemma_api_key=""))
    chunk = RetrievedStatuteChunk(
        chunk_id="test1",
        statute_name="PWDVA 2005",
        section="Section 17",
        content="Right to reside in shared household.",
        source_url="https://www.indiacode.nic.in/",
        similarity_score=0.91,
    )
    result = gateway.generate_legal_answer(
        query="Can I stay in my shared home?",
        context=[chunk],
    )
    assert result.is_refusal is False
    assert "PWDVA 2005" in result.text
    assert len(result.citations) == 1
    assert result.citations[0]["statute"] == "PWDVA 2005"
    assert result.citations[0]["section"] == "Section 17"


def test_generation_with_mocked_openai_client():
    gateway = GemmaOpenAIGateway(app_config=AppConfig(gemma_api_key="valid-key"))
    
    # Mock client and chat completion
    mock_choice = MagicMock()
    mock_choice.message.content = "Under Section 17 of PWDVA 2005, a woman has the right to reside in the shared household."
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 150
    mock_usage.completion_tokens = 45

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_response.usage = mock_usage

    gateway._client = MagicMock()
    gateway._client.chat.completions.create.return_value = mock_response

    chunk = RetrievedStatuteChunk(
        chunk_id="test1",
        statute_name="PWDVA 2005",
        section="Section 17",
        content="Right to reside in shared household.",
        source_url="https://www.indiacode.nic.in/",
        similarity_score=0.91,
    )

    result = gateway.generate_legal_answer(
        query="Can I stay in my home?",
        context=[chunk],
    )

    assert result.is_refusal is False
    assert "Section 17" in result.text
    assert result.prompt_tokens == 150
    assert result.completion_tokens == 45
    assert len(result.citations) == 1
