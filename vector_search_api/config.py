import logging

from pydantic import BaseSettings


class Settings(BaseSettings):
    logger_name = "vector-search-api"


settings = Settings()
logger = logging.getLogger(settings.logger_name)
