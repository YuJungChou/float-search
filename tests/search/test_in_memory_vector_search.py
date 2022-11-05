import logging

from tests.config import settings
from vector_search_api.helper.vector import random_array
from vector_search_api.search.in_memory_vector_search import InMemoryVectorSearch


logger = logging.getLogger("pytest")


dims = 8


vs_api = InMemoryVectorSearch(project=settings.test_project_name, dims=dims)


def test_api_describe():
    assert vs_api.describe()


def test_api_upsert():
    result = vs_api.upsert(
        [
            ("1", random_array(dims=dims), {"data": "ggwp"}),
            ("1", random_array(dims=dims), {"nested": {"data": "ggwp"}}),
            ("2", random_array(dims=dims), None),
            ("3", random_array(dims=dims)),
        ]
    )
    assert result


def test_api_query():
    result = vs_api.query(
        random_array(dims=dims), include_values=True, include_metadata=True
    )
    assert result
