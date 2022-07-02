import logging
import random
import uuid

import pytest

from vector_search_api.helper.vector import random_array
from vector_search_api.searcher.in_memory_vector_search import InMemoryVectorSearch

logger = logging.getLogger('pytest')


search_field = 'test_text'
metadata_field = 'test_metadata'
vector_field = 'test_vector'
similarity_field = 'test_similarity'


def get_test_documents(num: int = 100):
    test_documents = [
        {
            search_field: str(uuid.uuid4()),
            metadata_field: {'GGWP': 'ABC'},
            vector_field: random_array(8),
        }
        for _ in range(num)
    ]
    return test_documents


test_documents = get_test_documents()


@pytest.fixture(autouse=True, scope="module")
def vector_search_api():
    vs_api = InMemoryVectorSearch(
        project_name='test_vector_search_api',
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field,
        similarity_field=similarity_field,
    )
    yield vs_api


def test_in_memory_vector_search_project_create(
    vector_search_api: 'InMemoryVectorSearch'
):

    vector_search_api.create_project_if_not_exists()
    assert vector_search_api.get_project_or_none is not None


def test_in_memory_vector_search_insert_documents(
    vector_search_api: 'InMemoryVectorSearch'
):

    vector_search_api.insert_documents(test_documents)
    assert vector_search_api.count_documents() == len(test_documents)


def test_in_memory_vector_search_search_documents(
    vector_search_api: 'InMemoryVectorSearch'
):

    result = vector_search_api.search_documents(
        random.choice(test_documents)[vector_field]
    )
    assert len(result) > 0
    assert (
        search_field in result[0]
        and metadata_field in result[0]
        and vector_field in result[0]
        and similarity_field in result[0]
    )
    assert round(result[0][similarity_field], 2) == 1.0


def test_in_memory_vector_search_search_documents(
    vector_search_api: 'InMemoryVectorSearch'
):

    new_documents = get_test_documents(num=213)
    vector_search_api.refresh_documents(new_documents)
    assert vector_search_api.count_documents() == len(new_documents)

    result = vector_search_api.search_documents(
        random.choice(new_documents)[vector_field]
    )
    assert len(result) > 0
    assert (
        search_field in result[0]
        and metadata_field in result[0]
        and vector_field in result[0]
        and similarity_field in result[0]
    )
    assert round(result[0][similarity_field], 2) == 1.0
