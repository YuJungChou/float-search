import logging

from pydantic import BaseConfig


class Settings(BaseConfig):
    logger_name = 'float-search'
    logger = logging.getLogger(logger_name)


settings = Settings()
