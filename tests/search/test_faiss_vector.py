import logging

from tests.config import settings
from vector_search_api.helper.vector import random_array
from vector_search_api.search.faiss_vector_search import FaissVectorSearch


logger = logging.getLogger("pytest")

dims = 8

vs_api = FaissVectorSearch(project=settings.test_project_name, dims=dims)


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


def test_api_query():
    result = vs_api.query(
        random_array(dims=dims), include_values=True, include_metadata=True
    )
    assert result


def test_api_fetch():
    result = vs_api.fetch(ids=["1", "3"])
    assert result


def test_api_update():
    new_vector = random_array(dims=dims)
    vs_api.update(id="1", values=new_vector, set_metadata={"update_test": True})
    fetch_result = vs_api.fetch(ids=["1"])
    logger.error(fetch_result.vectors["1"].values)
    assert [round(f, 3) for f in fetch_result.vectors["1"].values] == [
        round(f, 3) for f in new_vector.tolist()
    ]
    assert fetch_result.vectors["1"].metadata
    assert fetch_result.vectors["1"].metadata["update_test"] is True
