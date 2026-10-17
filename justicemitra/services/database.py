"""MongoDB Atlas connection service, lifecycle management, and index definitions."""

import logging
from typing import Optional, Dict, Any, List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from pymongo.operations import SearchIndexModel
from justicemitra.config import config, AppConfig

logger = logging.getLogger(__name__)

_GLOBAL_MONGO_CLIENT: Optional[MongoClient] = None


def _configure_dns_resilience():
    """Ensures DNS resolution does not hang on inactive/unreachable network adapters on Windows."""
    try:
        import dns.resolver
        res = dns.resolver.get_default_resolver()
        # On Windows, disconnected adapters can register unreachable private DNS servers.
        # Ensure fast, reliable DNS resolution using standard public DNS.
        res.nameservers = ["8.8.8.8", "8.8.4.4", "1.1.1.1"]
        res.timeout = 2.0
        res.lifetime = 10.0
    except Exception:
        pass


def get_mongo_client(app_config: Optional[AppConfig] = None) -> Optional[MongoClient]:
    """Returns a pooled MongoClient singleton or None if connection string is unconfigured."""
    global _GLOBAL_MONGO_CLIENT
    cfg = app_config or config

    if not cfg.mongodb_uri:
        logger.warning("MONGODB_URI not set. Atlas database client cannot be initialized.")
        return None

    if _GLOBAL_MONGO_CLIENT is None:
        try:
            _configure_dns_resilience()
            logger.info("Initializing MongoClient connection pool...")
            _GLOBAL_MONGO_CLIENT = MongoClient(
                cfg.mongodb_uri,
                serverSelectionTimeoutMS=cfg.mongodb_server_selection_timeout_ms,
                connectTimeoutMS=cfg.mongodb_connect_timeout_ms,
                maxPoolSize=10,
                retryWrites=True,
            )
        except Exception as e:
            logger.error(f"Failed to create MongoClient: {e}")
            return None

    return _GLOBAL_MONGO_CLIENT


def check_mongo_connection(client: Optional[MongoClient]) -> bool:
    """Verifies that the MongoDB client can connect and ping the cluster within timeout."""
    if client is None:
        return False
    try:
        client.admin.command("ping")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.warning(f"MongoDB connection ping failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error pinging MongoDB: {e}")
        return False


def get_legal_chunks_collection(
    client: Optional[MongoClient], app_config: Optional[AppConfig] = None
):
    """Returns the legal_chunks collection from the configured database."""
    if client is None:
        return None
    cfg = app_config or config
    return client[cfg.mongodb_database][cfg.mongodb_collection]


def get_document_registry_collection(
    client: Optional[MongoClient], app_config: Optional[AppConfig] = None
):
    """Returns the document_registry collection for deduplication."""
    if client is None:
        return None
    cfg = app_config or config
    return client[cfg.mongodb_database]["document_registry"]


def get_vector_search_index_spec() -> Dict[str, Any]:
    """Returns the Atlas Vector Search index definition as specified in the architecture plan."""
    return {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "cosine",
            },
            {
                "type": "filter",
                "path": "statute_title",
            },
            {
                "type": "filter",
                "path": "jurisdiction",
            },
        ]
    }


def ensure_vector_search_index(
    collection,
    index_name: str = "vector_index",
) -> Dict[str, Any]:
    """Ensures that the vector search index exists on the given collection.
    
    Uses Atlas create_search_index API when supported.
    """
    definition = get_vector_search_index_spec()
    try:
        # Check existing search indexes
        existing = list(collection.list_search_indexes())
        for idx in existing:
            if idx.get("name") == index_name:
                logger.info(f"Vector search index '{index_name}' already exists.")
                return {"status": "exists", "name": index_name}

        # Create search index if not found
        model = SearchIndexModel(
            definition=definition,
            name=index_name,
            type="vectorSearch",
        )
        created_name = collection.create_search_index(model=model)
        logger.info(f"Initiated creation of vector search index: {created_name}")
        return {"status": "created", "name": created_name}
    except Exception as e:
        logger.warning(
            f"Note: Vector index creation via driver API encountered: {e}. "
            f"If on an Atlas tier requiring manual search index setup, ensure '{index_name}' is defined in Atlas UI."
        )
        return {"status": "unsupported_or_manual", "error": str(e), "spec": definition}
