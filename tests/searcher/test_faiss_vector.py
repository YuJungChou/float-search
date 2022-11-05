import copy
import logging

import pytest

from tests.config import settings
from tests.utils.dummy_documents import default_metadata, test_text_examples
from tests.utils.embedding import get_texts_embeddings
from vector_search_api.search.faiss_vector_search import FaissVectorSearch

logger = logging.getLogger('pytest')


search_field = 'test_text'
metadata_field = 'test_metadata'
vector_field = 'test_vector'
similarity_field = 'test_similarity'


@pytest.fixture(autouse=True, scope="module")
def vs_api():
    vs_api = FaissVectorSearch(
        project_name=settings.PROJECT_NAME,
        search_field=search_field,
        metadata_field=metadata_field,
        vector_field=vector_field,
        similarity_field=similarity_field,
        dims=settings.EMBEDDING_DIMS,
    )
    yield vs_api


def test_vs_api_project_create(
    vs_api: 'FaissVectorSearch'
):
    vs_api.create_project_if_not_exists()
    assert vs_api.get_project_or_none is not None


def test_vs_api_common_usages(
    vs_api: 'FaissVectorSearch'
):
    embeddings = get_texts_embeddings(
        texts=test_text_examples, dims=settings.EMBEDDING_DIMS
    )
    documents = [{
        search_field: text,
        metadata_field: copy.deepcopy(default_metadata),
        vector_field: emb,
    } for text, emb in zip(test_text_examples, embeddings)]

    insert_count = vs_api.insert_documents(documents=documents)
    assert insert_count == len(documents)

    logger.error(vs_api._index.ntotal)
    results = vs_api.search_documents(query=documents[0][vector_field])
    assert round(results[0]['test_similarity'], 2) == 1.0
    for res in results:
        print(f"{res['test_similarity']}: {res['test_text']}")
