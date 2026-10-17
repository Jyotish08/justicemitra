"""Unit tests for database connection and index definitions."""

import pytest
from justicemitra.services.database import (
    get_vector_search_index_spec,
    check_mongo_connection,
    get_mongo_client,
)
from justicemitra.config import AppConfig


def test_vector_search_index_specification():
    spec = get_vector_search_index_spec()
    assert "fields" in spec
    fields = spec["fields"]

    # Verify vector path, dimension (384), and cosine similarity
    vector_field = next((f for f in fields if f.get("type") == "vector"), None)
    assert vector_field is not None
    assert vector_field["path"] == "embedding"
    assert vector_field["numDimensions"] == 384
    assert vector_field["similarity"] == "cosine"

    # Verify filter fields
    filter_paths = [f["path"] for f in fields if f.get("type") == "filter"]
    assert "statute_title" in filter_paths
    assert "jurisdiction" in filter_paths


def test_mongo_client_without_uri():
    cfg = AppConfig(mongodb_uri="")
    client = get_mongo_client(cfg)
    assert client is None
    assert check_mongo_connection(None) is False


def test_mongo_client_with_invalid_uri():
    cfg = AppConfig(
        mongodb_uri="mongodb://non_existent_host:27017",
        mongodb_server_selection_timeout_ms=500,
        mongodb_connect_timeout_ms=500,
    )
    client = get_mongo_client(cfg)
    # Ping should gracefully return False without crashing
    assert check_mongo_connection(client) is False
