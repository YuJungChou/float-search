from typing import List, Optional, Text, Tuple, Union

import requests

from tests.config import settings
from vector_search_api.helper.vector import random_array


def get_texts_embeddings(
    texts: List[Text],
    dims: Optional[Union[int, Tuple]] = None,
    use_negative: bool = True,
) -> List[float]:
    """Query texts embeddings. Request external resource if EMBEDDING_URL is supplied."""

    if settings.EMBEDDING_URL is not None:
        embeddings = requests.post(settings.EMBEDDING_URL, json=texts).json()
    else:
        embeddings = [
            random_array(dims=dims, use_negative=use_negative)
            for _ in range(len(texts))
        ]

    return embeddings
