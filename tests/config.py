import os
import copy

from dataclasses import dataclass, fields


@dataclass(init=False)
class Settings:

    # Testing session
    test_project_name = os.environ.get("TEST_PROJECT_NAME", "pytest")

    def __init__(self, **kwargs):
        environ = copy.deepcopy(dict(os.environ))
        environ.update(**kwargs)

        case_insensitive_environ = {}
        for env, value in environ.items():
            case_insensitive_environ[env.casefold()] = value

        environ.update(case_insensitive_environ)

        for self_field in fields(self):
            if self_field.name.casefold() in environ:
                setattr(
                    self,
                    self_field.name,
                    self_field.type(environ[self_field.name.casefold()]),
                )


settings = Settings()
