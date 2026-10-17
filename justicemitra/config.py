"""Configuration management for JusticeMitra adhering to Twelve-Factor principles."""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()


@dataclass
class AppConfig:
    # MongoDB Atlas Configuration
    mongodb_uri: Optional[str] = os.getenv("MONGODB_URI")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "justicemitra_db")
    mongodb_collection: str = os.getenv("MONGODB_COLLECTION", "legal_chunks")
    mongodb_vector_index: str = os.getenv("MONGODB_VECTOR_INDEX", "vector_index")
    mongodb_connect_timeout_ms: int = int(os.getenv("MONGODB_CONNECT_TIMEOUT_MS", "5000"))
    mongodb_server_selection_timeout_ms: int = int(os.getenv("MONGODB_SERVER_TIMEOUT_MS", "5000"))

    # Gemma Inference Configuration
    gemma_api_base: str = os.getenv("GEMMA_API_BASE", "https://api.groq.com/openai/v1")
    gemma_api_key: str = os.getenv("GEMMA_API_KEY", "")
    gemma_model_name: str = os.getenv("GEMMA_MODEL_NAME", "gemma2-9b-it")
    gemma_temperature: float = float(os.getenv("GEMMA_TEMPERATURE", "0.1"))
    gemma_max_tokens: int = int(os.getenv("GEMMA_MAX_TOKENS", "1024"))
    gemma_timeout_seconds: float = float(os.getenv("GEMMA_TIMEOUT_SECONDS", "30.0"))

    # Embedding Service Configuration
    embedding_model_name: str = os.getenv(
        "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_dimension: int = 384

    # Sentry Observability Configuration
    sentry_dsn: Optional[str] = os.getenv("SENTRY_DSN")
    sentry_environment: str = os.getenv("SENTRY_ENVIRONMENT", "production")
    sentry_traces_sample_rate: float = float(
        os.getenv("SENTRY_TRACES_SAMPLE_RATE", "1.0")
    )

    # Retrieval and Grounding Parameters
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    retrieval_candidates_multiplier: int = int(os.getenv("RETRIEVAL_CANDIDATES_MULTIPLIER", "10"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.40"))

    # Web & Hosting
    port: int = int(os.getenv("PORT", "8501"))

    @classmethod
    def load(cls) -> "AppConfig":
        """Loads and returns the current configuration."""
        return cls()


config = AppConfig.load()
