from typing import Any, Dict, List, Text, Tuple, Union

import numpy as np

from vector_search_api.config import logger, settings
from vector_search_api.helper.vector import distance_to_similarity
from vector_search_api.schema import Record
from vector_search_api.search.base_vector_search import BaseVectorSearch


try:
    import faiss
except ImportError:
    logger.warning("Trying import faiss but uninstalled.")


class FaissVectorSearch(BaseVectorSearch):
    def __init__(self, project: Text, dims: int, **kwargs):

        super(FaissVectorSearch, self).__init__(project=project, dims=dims, **kwargs)

        self._metadata: Dict[Text, Dict[Text, Any]] = {}
        self._ids = np.array([])
        self._vectors = np.empty((0, self.dims))
        self._index = faiss.IndexFlatL2(self.dims)

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
        """Query vector search."""

        vector_np = np.array([vector]).astype("float32")
        top_k = self._index.ntotal if top_k >= self._index.ntotal else top_k

        distances, top_k_idxs = self._index.search(vector_np, top_k)

        result: Dict = {
            "matches": [
                {
                    "id": self._ids[idx],
                    "score": distance_to_similarity(distance),
                    "value": (self._vectors[idx] if include_values is True else None),
                    "metadata": (
                        self._metadata[self._ids[idx]]
                        if include_metadata is True
                        else None
                    ),
                }
                for distance, idx in zip(distances[0], top_k_idxs[0])
            ]
        }
        return result

    def upsert(self, records: List[Union[Record, Tuple]]) -> Dict:
        """Upsert records."""

        update_ids: List[Text] = []
        update_vectors: List[List[float]] = []
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

        self._index.add(np.array(update_vectors).astype("float32"))

        self._metadata.update(**update_metadata)
        self._ids = np.append(self._ids, update_ids)
        self._vectors = np.concatenate((self._vectors, update_vectors), axis=0)

        return {"success": True}
