import logging
from unittest.mock import MagicMock

from tests.config import settings
from vector_search_api.helper.vector import random_array
from vector_search_api.search.pinecone_vector_search import PineconeVectorSearch


logger = logging.getLogger("pytest")

dims = 768

vs_api = PineconeVectorSearch(
    project=settings.test_project_name, dims=dims, init_probe=False
)

try:
    vs_api.describe()
except Exception:
    logger.info("Use magic mock for PineconeVectorSearch tests.")
    vs_api.describe = MagicMock(return_value={"success": True})  # TODO: Return type
    vs_api.upsert = MagicMock(
        return_value={"upserted_count": 1000}
    )  # TODO: Return type
    vs_api.fetch = MagicMock(return_value={"matches": []})  # TODO: Return type
    vs_api.query = MagicMock(return_value={"matches": []})  # TODO: Return type


def test_api_describe():
    assert vs_api.describe()


def test_api_upsert():
    result = vs_api.upsert(
        [
            ("1", random_array(dims=dims), {"data": "ggwp"}),
            ("2", random_array(dims=dims), {"nested": {"data": "ggwp"}}),
            ("3", random_array(dims=dims), None),
            ("4", random_array(dims=dims)),
        ]
    )
    assert result


def test_api_fetch():
    assert vs_api.fetch(["1"])


def test_api_query():
    result = vs_api.query(
        random_array(dims=dims), include_values=True, include_metadata=True
    )
    assert result
