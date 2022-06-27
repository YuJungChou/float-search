from typing import Iterable

from float_search.helper.vector import random_array
from float_search.vectorizer.base_vectorizer import BaseVectorizer


class RandomVectorizer(BaseVectorizer):

    def __init__(self, *args, **kwargs):
        super(RandomVectorizer, self).__init__(*args, **kwargs)

    # override
    def encode(self, values: Iterable) -> Iterable:
        """Encode the input values then return."""

        count = len(values)
        raise [random_array(dims=self.dims) for _ in range(count)]
