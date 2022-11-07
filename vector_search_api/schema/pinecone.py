from typing import Dict, List, Optional, Text

from pydantic import BaseModel


class PineconeNamespace(BaseModel):
    vector_count: int


class PineconeIndex(BaseModel):
    dimension: int
    index_fullness: float
    namespaces: Dict[Text, PineconeNamespace]
    total_vector_count: int


class PineconeUpsertResult(BaseModel):
    upserted_count: int


class PineconeMatch(BaseModel):
    id: Text
    score: float
    sparseValues: Dict
    values: List[float]
    metadata: Optional[Dict] = None


class PineconeQueryResult(BaseModel):
    matches: List[PineconeMatch]
    namespace: Text
