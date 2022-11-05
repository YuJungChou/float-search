from typing import Dict, List, Optional, Text, Tuple

from vector_search_api.schema import Record


class BaseVectorSearch:
    """Base Vector Search ABC."""

    def __init__(self, project: Text, dims: Optional[int] = None, **kwargs):
        """"""

        self.project: Text = project
        self.dims = int(dims) if dims else None
        self.kwargs = kwargs

    def describe(self) -> Dict:
        """"""

        raise NotImplementedError()

    def query(
        self, vector: List[float], top_k: int = 3, include_values: bool = False
    ) -> Dict:
        """"""

        raise NotImplementedError()

    def upsert(self, records: List["Record"]) -> Dict:
        """"""

        raise NotImplementedError()
