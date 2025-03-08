from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
import json
import pandas as pd

from backend.app.config import get_settings
from backend.app.services.topic_model import get_topics, get_topic_articles

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

class TopicResponse(BaseModel):
    topics: List[Dict[str, Any]]

class ArticlesResponse(BaseModel):
    articles: List[Dict[str, Any]]

@router.get("/", response_model=TopicResponse)
async def list_topics(limit: int = Query(20, ge=1, le=50)):
    """Get all available topics"""
    try:
        topics = await get_topics(limit=limit)
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Error listing topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{topic_id}/articles", response_model=ArticlesResponse)
async def topic_articles(
    topic_id: int,
    limit: int = Query(10, ge=1, le=50)
):
    """Get articles for a specific topic"""
    try:
        articles = await get_topic_articles(topic_id=topic_id, limit=limit)
        return {"articles": articles}
    except Exception as e:
        logger.error(f"Error getting topic articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary", response_model=Dict[str, Any])
async def topic_summary():
    """Get a summary of topics with article counts"""
    try:
        # Load topic info from file
        topic_info_path = f"{settings.DATA_DIR}/topic_info.json"
        
        if os.path.exists(topic_info_path):
            with open(topic_info_path, "r") as f:
                topic_info = json.load(f)
            
            # Get topic counts
            topic_counts = {}
            for topic in topic_info:
                if topic["Topic"] != -1:  # Skip outlier topic
                    topic_counts[str(topic["Topic"])] = {
                        "count": topic.get("Count", 0),
                        "name": topic.get("Name", f"Topic {topic['Topic']}")
                    }
            
            return {
                "total_articles": sum(t.get("count",.0) for t in topic_counts.values()),
                "topic_counts": topic_counts
            }
        else:
            # Try to get topic counts from articles CSV
            articles_path = f"{settings.DATA_DIR}/news_articles.csv"
            if os.path.exists(articles_path):
                df = pd.read_csv(articles_path)
                
                if "topic" in df.columns:
                    # Count articles by topic
                    topic_counts = df["topic"].value_counts().to_dict()
                    
                    return {
                        "total_articles": len(df),
                        "topic_counts": {
                            str(topic): {"count": count, "name": f"Topic {topic}"}
                            for topic, count in topic_counts.items()
                            if topic != -1  # Skip outlier topic
                        }
                    }
            
            # Return empty response if no data
            return {"total_articles": 0, "topic_counts": {}}
    except Exception as e:
        logger.error(f"Error getting topic summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/keywords", response_model=Dict[str, Dict[str, List[Dict[str, Any]]]])
async def topic_keywords():
    """Get keywords for all topics"""
    try:
        # Load topic keywords from file
        keywords_path = f"{settings.DATA_DIR}/topic_keywords.json"
        
        if os.path.exists(keywords_path):
            with open(keywords_path, "r") as f:
                topic_keywords = json.load(f)
            
            return {"topic_keywords": topic_keywords}
        
        # Return empty response if no data
        return {"topic_keywords": {}}
    except Exception as e:
        logger.error(f"Error getting topic keywords: {e}")
        raise HTTPException(status_code=500, detail=str(e))
