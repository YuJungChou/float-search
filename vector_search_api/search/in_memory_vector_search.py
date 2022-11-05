from typing import Any, Dict, List, Text, Tuple, Union

import numpy as np

from vector_search_api.helper.vector import cosine_similarity
from vector_search_api.schema import Record
from vector_search_api.search.base_vector_search import BaseVectorSearch


class InMemoryVectorSearch(BaseVectorSearch):
    def __init__(self, project: Text, dims: int, **kwargs):
        """Initialize basic attributes project, dims, also the storage of records."""

        super(InMemoryVectorSearch, self).__init__(project=project, dims=dims, **kwargs)

        self._metadata: Dict[Text, Dict[Text, Any]] = {}
        self._ids = np.array([])
        self._vectors = np.empty((0, self.dims))

    def describe(self) -> Dict:
        """Describe the records."""

        return {"count": self._ids.size}

    def query(
        self,
        vector: List[float],
        top_k: int = 3,
        include_values: bool = False,
        include_metadata: bool = False,
    ) -> Dict:
        """Query vector search.

        Parameters
        ----------
        vector : list[float]
            Query with vector.

        top_k : int
            Top k.

        include_values : bool, default is False
            Return vector or not.

        include_metadata : bool, default is False
            Return metadata or not.

        Returns
        -------
        query_result : dict
            Query result.
        """

        cos_sim = cosine_similarity(np.array(vector), targets=self._vectors)
        top_k_idxs = np.argsort(cos_sim)[-top_k:][::-1]

        result: Dict = {
            "matches": [
                {
                    "id": self._ids[idx],
                    "score": cos_sim[idx],
                    "value": (self._vectors[idx] if include_values is True else None),
                    "metadata": (
                        self._metadata[self._ids[idx]]
                        if include_metadata is True
                        else None
                    ),
                }
                for idx in top_k_idxs
            ]
        }
        return result

    def upsert(self, records: List[Union[Record, Tuple]]) -> Dict:
        """Upsert records."""

        update_ids = []
        update_vectors = []
        update_metadata = {}
        for doc in records:
            record = Record(*doc)
            if not record.id:
                raise ValueError(f"The value of id '{record.id}' is not validated.")
            if len(record.vector) != self.dims:
                raise ValueError(
                    f"The vector dimension {len(record.vector)} is not validated."
                )
            update_ids.append(str(record.id))
            update_vectors.append(record.vector)
            update_metadata[str(record.id)] = record.metadata or {}

        self._metadata.update(**update_metadata)
        self._ids = np.append(self._ids, update_ids)
        self._vectors = np.concatenate((self._vectors, update_vectors), axis=0)

        return {"success": True}
