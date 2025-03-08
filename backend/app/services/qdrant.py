from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import logging
import uuid
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import aiohttp
import json

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global Qdrant client
_client = None

def get_qdrant_client():
    """Get or initialize Qdrant client"""
    global _client
    if _client is None:
        logger.info(f"Connecting to Qdrant at {settings.QDRANT_URL}")
        _client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        logger.info("Qdrant client initialized")
    
    return _client

async def check_collection_exists() -> bool:
    """Check if the collection exists in Qdrant"""
    client = get_qdrant_client()
    collections = client.get_collections().collections
    return any(collection.name == settings.QDRANT_COLLECTION_NAME for collection in collections)

async def create_collection(vector_size: int = 384):
    """Create a new collection in Qdrant if it doesn't exist"""
    exists = await check_collection_exists()
    if exists:
        logger.info(f"Collection {settings.QDRANT_COLLECTION_NAME} already exists")
        return
    
    client = get_qdrant_client()
    logger.info(f"Creating new collection: {settings.QDRANT_COLLECTION_NAME}")
    client.create_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )
    logger.info(f"Collection {settings.QDRANT_COLLECTION_NAME} created")

async def upload_to_qdrant(df: pd.DataFrame, embeddings: np.ndarray):
    """Upload data points with embeddings to Qdrant"""
    if len(df) == 0 or len(embeddings) == 0:
        logger.warning("No data to upload to Qdrant")
        return
    
    if len(df) != len(embeddings):
        raise ValueError(f"DataFrame length ({len(df)}) must match embeddings length ({len(embeddings)})")
    
    client = get_qdrant_client()
    logger.info(f"Uploading {len(df)} points to Qdrant collection {settings.QDRANT_COLLECTION_NAME}")
    
    # Prepare points for upload
    points = []
    for i, row in df.iterrows():
        # Convert row to dict for payload
        payload = row.to_dict()
        
        # Convert non-serializable values to strings
        for key, value in payload.items():
            if pd.isna(value) or value is None:
                payload[key] = ""
            elif not isinstance(value, (str, int, float, bool, list, dict)):
                payload[key] = str(value)
        
        # Generate a stable UUID based on title to avoid duplicates
        id_str = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(payload.get("title", f"article_{i}"))))
        
        # Convert UUID to integer (required by Qdrant)
        point_id = int(id_str.replace("-", "")[:16], 16)
        
        # Create point
        points.append(models.PointStruct(
            id=point_id,
            vector=embeddings[i].tolist(),
            payload=payload
        ))
    
    # Upload in batches to avoid timeouts
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            points=batch
        )
        logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
    
    logger.info("Upload complete")

async def search_qdrant(query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
    """Search Qdrant using vector similarity"""
    client = get_qdrant_client()
    
    try:
        results = client.search(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
        
        return [
            {
                "id": str(result.id),
                "score": float(result.score),
                "payload": result.payload
            }
            for result in results
        ]
    except Exception as e:
        logger.error(f"Error searching Qdrant: {e}")
        # Try with HTTP request as a fallback
        try:
            return await search_qdrant_http(query_vector, limit)
        except Exception as e2:
            logger.error(f"HTTP fallback also failed: {e2}")
            return []

async def search_qdrant_http(query_vector: List[float], limit: int = 10) -> List[Dict[str, Any]]:
    """Search Qdrant using HTTP API as a fallback"""
    url = f"{settings.QDRANT_URL}/collections/{settings.QDRANT_COLLECTION_NAME}/points/search"
    headers = {
        "Content-Type": "application/json",
        "api-key": settings.QDRANT_API_KEY
    }
    payload = {
        "vector": query_vector,
        "limit": limit
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("result", [])
            else:
                text = await response.text()
                logger.error(f"Qdrant HTTP search error: {text}")
                return []
