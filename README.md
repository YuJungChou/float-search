# Changing the direction of the Repository...
---

# Vector Search API
Vector-Search API of all resources.

## Quick Start
1. Install package.
```sh
(env) pip install vector-search-api
```

2. Execute code.
```python
import numpy as np
from vector_search_api.search import InMemoryVectorSearch

dims = 8
vs_api = InMemoryVectorSearch(project='vs', dims=dims)

records = [
    ('A', np.random.random(dims), {'text': 'You good.'}),
    ('B', np.random.random(dims), {'text': 'I good.'}),
    ('C', np.random.random(dims), {'text': 'Every one good.'}),
]

vs_api.upsert(records)
result = vs_api.query(np.random.random(8))
{
    "matches": [
        {
            "id": "A",
            "metadata": {
                "text": "You good."
            },
            "score": 0.8790238688925034
        },
        {
            "id": "C",
            "metadata": {
                "text": "Every one good."
            },
            "score": 0.7072865376885351
        },
        {
            "id": "B",
            "metadata": {
                "text": "I good."
            },
            "score": 0.6696056110470604
        }
    ]
}
```

## Basic Usages

## Release Notes
