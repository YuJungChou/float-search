import os

from pydantic import BaseSettings


class Settings(BaseSettings):

    # Testing session
    test_project_name = os.environ.get("TEST_PROJECT_NAME", "pytest")


settings = Settings()
