import os

from pydantic import BaseConfig


class Settings(BaseConfig):

    # Testing session
    PROJECT_NAME = os.environ.get('EMBEDDING_URL', 'test_vector_search_api')

    # Resources
    EMBEDDING_URL = os.environ.get('EMBEDDING_URL')
    EMBEDDING_DIMS = eval(os.environ.get('EMBEDDING_DIMS', '8'))


settings = Settings()
