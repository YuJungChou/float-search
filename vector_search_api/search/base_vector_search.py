from typing import Dict, List, Optional, Text, Tuple, Union

from vector_search_api.schema import Record


class BaseVectorSearch:
    """Base Vector Search ABC."""

    def __init__(self, project: Text, dims: Optional[int] = None, **kwargs):
        """Initialize basic attributes project, dims."""

        self.project: Text = project
        self.dims = int(dims) if dims else None
        self.kwargs = kwargs

    def describe(self) -> Dict:
        """Describe the api status."""

        raise NotImplementedError()

    def query(
        self,
        vector: List[float],
        top_k: int = 3,
        include_values: bool = False,
        include_metadata: bool = False,
    ) -> Dict:
        """Query vector search."""

        raise NotImplementedError()

    def upsert(self, records: List[Union[Record, Tuple]]) -> Dict:
        """Upsert records."""

        raise NotImplementedError()
