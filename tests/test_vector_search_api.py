import vector_search_api
from vector_search_api.version import version_name


def test_version():
    assert vector_search_api.__version__ == version_name
