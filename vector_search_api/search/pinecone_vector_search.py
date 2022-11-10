from typing import Dict, List, Optional, Text, Tuple, Union

from vector_search_api.schema import Record
from vector_search_api.search.base_vector_search import BaseVectorSearch
from vector_search_api.schema.result import (
    Index,
    QueryResult,
    UpsertResult,
)
from vector_search_api.config import settings


class PineconeVectorSearch(BaseVectorSearch):
    """Pinecone Vector Search."""

    def __init__(
        self,
        project: Text,
        index: Text = settings.pinecone_index_name,
        namespace: Text = settings.pinecone_namespace,
        api_key: Text = settings.pinecone_api_key,
        environment: Text = settings.pinecone_environment,
        dims: Optional[int] = None,
        init_probe: bool = True,
        **kwargs
    ):
        """Initialize basic attributes project, dims, also the storage of records."""

        import pinecone

        pinecone.init(api_key=api_key, environment=environment)

        super(PineconeVectorSearch, self).__init__(project=project, dims=dims, **kwargs)

        self.index = index
        self.namespace = namespace
        self._index = pinecone.Index(self.index)

        if init_probe is True:
            pinecone.whoami()
            self.describe()

    def describe(self) -> Union[Dict, Index]:
        """Describe the api status."""

        index_stats = self._index.describe_index_stats()
        self.dims = index_stats["dimension"]
        return index_stats

    def query(
        self,
        vector: List[float],
        top_k: int = 3,
        include_values: bool = False,
        include_metadata: bool = False,
        namespace: Optional[Text] = None,
    ) -> Union[Dict, QueryResult]:
        """Query vector search."""

        namespace = namespace or self.namespace
        query_result = self._index.query(
            vector=vector,
            top_k=top_k,
            include_values=include_values,
            include_metadata=include_metadata,
            namespace=namespace,
        )

        return query_result

    def upsert(
        self, records: List[Union[Record, Tuple]], namespace: Optional[Text] = None
    ) -> Union[Dict, UpsertResult]:
        """Upsert records."""

        namespace = namespace or self.namespace
        upsert_result = self._index.upsert(
            [Record(*doc) for doc in records], namespace=namespace
        )
        return upsert_result
