import logging
import random
import uuid

import pytest

from tests.utils.dummy_documents import get_test_documents
from vector_search_api.helper.vector import random_array
from vector_search_api.search.in_memory_vector_search import InMemoryVectorSearch
from vector_search_api.vectorizer.random_vectorizer import RandomVectorizer

logger = logging.getLogger('pytest')


dims = 10
search_field = 'test_text'
metadata_field = 'test_metadata'
vector_field = 'test_vector'
similarity_field = 'test_similarity'


@pytest.fixture(autouse=True, scope="module")
def vector_search_api():
    vs_api = InMemoryVectorSearch(
        project_name='test_vector_search_api',
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field,
        similarity_field=similarity_field,
        dims=dims,
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
    test_documents = get_test_documents(
        dims=dims,
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field
    )
    vector_search_api.insert_documents(test_documents)
    assert vector_search_api.count_documents() == len(test_documents)


def test_in_memory_vector_search_search_documents(
    vector_search_api: 'InMemoryVectorSearch'
):

    test_case_idx = random.randint(0, len(vector_search_api._data[vector_field]) - 1)
    result = vector_search_api.search_documents(
        vector_search_api._data[vector_field][test_case_idx]
    )
    assert len(result) > 0
    assert (
        search_field in result[0]
        and metadata_field in result[0]
        and vector_field in result[0]
        and similarity_field in result[0]
    )
    assert round(result[0][similarity_field], 2) == 1.0


def test_in_memory_vector_search_refresh_documents(
    vector_search_api: 'InMemoryVectorSearch'
):

    new_documents = get_test_documents(
        num=213,
        dims=dims,
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field
    )
    vector_search_api.refresh_documents(new_documents)
    assert vector_search_api.count_documents() == len(new_documents)

    test_case_idx = random.randint(0, len(vector_search_api._data[vector_field]) - 1)
    result = vector_search_api.search_documents(
        vector_search_api._data[vector_field][test_case_idx]
    )
    assert len(result) > 0
    assert (
        search_field in result[0]
        and metadata_field in result[0]
        and vector_field in result[0]
        and similarity_field in result[0]
    )
    assert round(result[0][similarity_field], 2) == 1.0


def test_in_memory_vector_search_insert_documents_with_dynamic_encoding():

    dims = 8
    vectorizer = RandomVectorizer(dims=dims)
    fs_api = InMemoryVectorSearch(
        project_name='test_vector_search_api',
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field,
        similarity_field=similarity_field,
        vectorizer=vectorizer,
        dims=dims
    )
    fs_api.create_project_if_not_exists()

    test_documents = get_test_documents(
        dims=dims,
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field,
        with_vector=True
    )
    fs_api.insert_documents(test_documents)
    assert fs_api.count_documents() == len(test_documents)

    result = fs_api.search_documents(random.choice(test_documents)[vector_field])
    assert len(result) > 0
    assert (
        search_field in result[0]
        and metadata_field in result[0]
        and vector_field in result[0]
        and similarity_field in result[0]
    )
