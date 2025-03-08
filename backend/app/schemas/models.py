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

# Topic models
class Keyword(BaseModel):
    word: str
    score: float

class Topic(BaseModel):
    id: int
    name: str
    count: int
    keywords: List[Keyword]

class TopicResponse(BaseModel):
    topics: List[Topic]

class ArticlesResponse(BaseModel):
    articles: List[Article]

# Knowledge graph models
class Node(BaseModel):
    id: str
    group: int
    count: Optional[int] = None

class Link(BaseModel):
    source: str
    target: str
    value: int

class KnowledgeGraph(BaseModel):
    nodes: List[Node]
    links: List[Link]

class EntityConnection(BaseModel):
    entity: str
    strength: int
    count: Optional[int] = None

class EntityDetail(BaseModel):
    entity: str
    count: int
    connections: List[EntityConnection]
