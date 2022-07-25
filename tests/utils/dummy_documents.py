import uuid
from typing import Dict, Optional, Text, Tuple, Union

from vector_search_api.helper.vector import random_array


default_metadata = {'GGWP': 'ABC'}

test_text_examples = [
    'First do this.', 'Then do it better.', 'open the light.', 'open the air con.',
    'open lights in room', 'open all lights', 'talk a joke.', 'play some music.',
    'wake me up.', 'อากาศตอนนี้เป็นอย่างไรบ้าง', '台積電現在股價多少'
]


def get_test_documents(
    search_field: Text,
    metadata_field: Text,
    vector_field: Text,
    metadata: Optional[Dict] = None,
    num: int = 100,
    dims: Union[Tuple, int] = 8,
    with_vector: bool = True
):
    """Get test_documents."""

    if isinstance(dims, int) is True:
        dims = (dims, )

    if metadata is None:
        metadata = default_metadata

    test_documents = [
        {
            search_field: str(uuid.uuid4()),
            metadata_field: metadata,
            vector_field: random_array(3) if with_vector is True else None,
        }
        for _ in range(num)
    ]
    return test_documents
