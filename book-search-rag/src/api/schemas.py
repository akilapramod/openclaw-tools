from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    min_score: Optional[float] = 0.0
    overfetch_factor: Optional[int] = 3
    sources: Optional[List[str]] = None

class SearchResult(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    rerank_score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
