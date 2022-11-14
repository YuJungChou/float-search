from typing import List, Optional, Text, Tuple, Union

from vector_search_api.schema import Record
from vector_search_api.schema.result import Index, QueryResult, UpsertResult


class BaseVectorSearch:
    """Base Vector Search ABC."""

    def __init__(self, project: Text, dims: Optional[int] = None, **kwargs):
        """Initialize basic attributes project, dims."""

        self.project: Text = project
        self.dims = int(dims) if dims else None
        self.kwargs = kwargs

    def describe(self) -> "Index":
        """Describe the api status."""

        raise NotImplementedError()

    def query(
        self,
        vector: List[float],
        top_k: int = 3,
        include_values: bool = False,
        include_metadata: bool = False,
    ) -> "QueryResult":
        """Query vector search."""

        raise NotImplementedError()

    def upsert(self, records: List[Union[Record, Tuple]]) -> UpsertResult:
        """Upsert records."""

        raise NotImplementedError()
