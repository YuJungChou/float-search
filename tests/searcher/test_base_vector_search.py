import logging
import random

from tests.utils.dummy_documents import get_test_documents
from vector_search_api.searcher.base_vector_search import DummyTestVectorSearch

logger = logging.getLogger('pytest')


dims = 10
search_field = 'test_text'
metadata_field = 'test_metadata'
vector_field = 'test_vector'
similarity_field = 'test_similarity'


def test_base_project_create():

    fs_api = DummyTestVectorSearch(
        dims=dims,
        project_name='test_vector_search_api',
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field,
        similarity_field=similarity_field,
    )
    fs_api.create_project_if_not_exists()
    assert fs_api.get_project_or_none is not None


def test_base_operate_documents():

    fs_api = DummyTestVectorSearch(
        project_name='test_vector_search_api',
        dims=dims,
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field,
        similarity_field=similarity_field,
    )
    fs_api.create_project_if_not_exists()

    test_documents = get_test_documents(
        dims=dims,
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field
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

    new_documents = get_test_documents(
        num=213,
        dims=dims,
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field
    )
    fs_api.refresh_documents(new_documents)
    assert fs_api.count_documents() == len(new_documents)
