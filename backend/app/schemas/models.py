from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# Search models
class Article(BaseModel):
    title: str
    description: str
    url: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None
    scrape_date: Optional[str] = None
    topic: Optional[int] = None
    topic_prob: Optional[float] = None

class SearchResult(BaseModel):
    score: float
    payload: Article

class SearchRequest(BaseModel):
    query: str
    limit: int = Field(10, ge=1, le=50)

class SearchResponse(BaseModel):
    results: List[SearchResult]
