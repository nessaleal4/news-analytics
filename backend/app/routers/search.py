
from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import pandas as pd
import os
import re
from backend.app.services.embedding import generate_embedding
from backend.app.services.qdrant import search_qdrant

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

class SearchRequest(BaseModel):
    query: str
    limit: int = 10

class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]

@router.post("/semantic", response_model=SearchResponse)
async def semantic_search(request: SearchRequest):
    """Search articles using semantic similarity"""
    try:
        # Generate embedding for query
        query_vector = await generate_embedding(request.query)
        
        # Search Qdrant
        results = await search_qdrant(query_vector, limit=request.limit)
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/keyword", response_model=SearchResponse)
async def keyword_search(request: SearchRequest):
    """Search articles using keyword matching"""
    try:
        # Load articles from file
        df_path = f"{settings.DATA_DIR}/news_articles.csv"
        if not os.path.exists(df_path):
            return {"results": []}
        
        df = pd.read_csv(df_path)
        
        # Prepare query
        query_lower = request.query.lower()
        
        # Search in title and description
        matches = []
        for _, row in df.iterrows():
            title = str(row.get('title', '')).lower()
            desc = str(row.get('description', '')).lower()
            
            # Calculate score based on keyword matches
            score = 0
            if query_lower in title:
                score += 3 * title.count(query_lower)
            if query_lower in desc:
                score += desc.count(query_lower)
                
            # Add individual terms
            for term in query_lower.split():
                if len(term) > 2:  # Ignore short terms
                    if term in title:
                        score += title.count(term)
                    if term in desc:
                        score += 0.5 * desc.count(term)
            
            if score > 0:
                matches.append({
                    "score": float(score),
                    "payload": row.to_dict()
                })
        
        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top matches
        return {"results": matches[:request.limit]}
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recent", response_model=SearchResponse)
async def get_recent_articles(limit: int = Query(10, ge=1, le=50)):
    """Get recent articles"""
    try:
        # Load articles from file
        df_path = f"{settings.DATA_DIR}/news_articles.csv"
        if not os.path.exists(df_path):
            return {"results": []}
        
        df = pd.read_csv(df_path)
        
        # Sort by scrape date if available
        if 'scrape_date' in df.columns:
            df = df.sort_values('scrape_date', ascending=False)
        
        # Return top articles
        results = []
        for _, row in df.head(limit).iterrows():
            results.append({
                "score": 1.0,  # Default score
                "payload": row.to_dict()
            })
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error getting recent articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categories", response_model=Dict[str, List[str]])
async def get_categories():
    """Get available categories"""
    try:
        # Load articles from file
        df_path = f"{settings.DATA_DIR}/news_articles.csv"
        if not os.path.exists(df_path):
            return {"categories": []}
        
        df = pd.read_csv(df_path)
        
        # Get unique categories
        if 'category' in df.columns:
            categories = df['category'].dropna().unique().tolist()
        else:
            categories = []
        
        # Get unique sources
        if 'source' in df.columns:
            sources = df['source'].dropna().unique().tolist()
        else:
            sources = []
        
        return {
            "categories": categories,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/filter", response_model=SearchResponse)
async def filter_articles(
    category: Optional[str] = None,
    source: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = Query(10, ge=1, le=50)
):
    """Filter articles by category and/or source"""
    try:
        # Load articles from file
        df_path = f"{settings.DATA_DIR}/news_articles.csv"
        if not os.path.exists(df_path):
            return {"results": []}
        
        df = pd.read_csv(df_path)
        
        # Apply filters
        if category and 'category' in df.columns:
            df = df[df['category'].str.lower() == category.lower()]
        
        if source and 'source' in df.columns:
            df = df[df['source'].str.lower() == source.lower()]
        
        if query:
            query_lower = query.lower()
            # Create a mask for filtering
            mask = (
                df['title'].str.lower().str.contains(query_lower, regex=False, na=False) | 
                df['description'].str.lower().str.contains(query_lower, regex=False, na=False)
            )
            df = df[mask]
        
        # Sort by scrape date if available
        if 'scrape_date' in df.columns:
            df = df.sort_values('scrape_date', ascending=False)
        
        # Return top articles
        results = []
        for _, row in df.head(limit).iterrows():
            results.append({
                "score": 1.0,  # Default score
                "payload": row.to_dict()
            })
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error filtering articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))
