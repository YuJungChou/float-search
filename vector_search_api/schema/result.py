from typing import Dict, List, Optional, Text

from vector_search_api.schema.base import DataclassBase


class Namespace(DataclassBase):
    vector_count: int


class Index(DataclassBase):
    dimension: int
    index_fullness: float
    namespaces: Dict[Text, Namespace]
    total_vector_count: int


class UpsertResult(DataclassBase):
    upserted_count: int


class Match(DataclassBase):
    id: Text
    score: float
    sparseValues: Dict
    values: List[float]
    metadata: Optional[Dict] = None


class QueryResult(DataclassBase):
    matches: List[Match]
    namespace: Text
