from typing import Dict, List, NamedTuple, Text


class Record(NamedTuple):
    id: Text
    vector: List[float]
    metadata: Dict = None
