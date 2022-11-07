import logging
import uuid
from typing import Text

from pydantic import BaseSettings


class Settings(BaseSettings):
    logger_name = "vector-search-api"

    # Pinecone
    pinecone_api_key: Text = str(uuid.uuid4())
    pinecone_environment: Text = "us-west1-gcp"


settings = Settings()
logger = logging.getLogger(settings.logger_name)
