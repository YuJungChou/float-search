import float_search
from float_search.version import version_name


def test_version():
    assert float_search.__version__ == version_name
