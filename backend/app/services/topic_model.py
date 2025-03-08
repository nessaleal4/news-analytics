import logging
from typing import List, Dict, Any, Tuple
import os
import json
from bertopic import BERTopic
import numpy as np
from app.config import get_settings
import asyncio

logger = logging.getLogger(__name__)
settings = get_settings()

_topic_model = None

async def build_topic_model(texts: List[str], nr_topics: int = 10) -> Tuple[BERTopic, List[int], List[float]]:
    """Build and train a BERTopic model"""
    logger.info("Building topic model...")
    
    # Define seed topics for guided topic modeling
    seed_topics = [
        ["politics", "government", "president", "election", "vote"],
        ["business", "market", "economy", "company", "stock"],
        ["sports", "team", "game", "player", "win"],
        ["technology", "tech", "innovation", "digital", "software"],
        ["health", "medical", "doctor", "patient", "disease"],
        ["entertainment", "movie", "music", "celebrity", "film"],
        ["world", "international", "global", "country", "foreign"]
    ]

    # Initialize BERTopic with guided topics
    topic_model = BERTopic(
        nr_topics=nr_topics,
        seed_topic_list=seed_topics,
        verbose=True
    )

    # Train the model (this is CPU-intensive and can be slow)
    # Run in an executor to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    topics, probs = await loop.run_in_executor(None, lambda: topic_model.fit_transform(texts))
    
    # Save the model info
    global _topic_model
    _topic_model = topic_model

    return topic_model, topics, probs

async def get_topics(limit: int = 20) -> List[Dict[str, Any]]:
    """Get the top topics and their keywords"""
    # Load topic info from file if available
    topic_info_path = f"{settings.DATA_DIR}/topic_info.json"
    topic_keywords_path = f"{settings.DATA_DIR}/topic_keywords.json"
    
    if os.path.exists(topic_info_path) and os.path.exists(topic_keywords_path):
        try:
            with open(topic_info_path, "r") as f:
                topic_info = json.load(f)
            
            with open(topic_keywords_path, "r") as f:
                topic_keywords = json.load(f)
            
            # Combine info and keywords
            topics = []
            for topic in topic_info:
                if topic["Topic"] != -1 and str(topic["Topic"]) in topic_keywords:
                    topics.append({
                        "id": topic["Topic"],
                        "name": topic.get("Name", f"Topic {topic['Topic']}"),
                        "count": topic.get("Count", 0),
                        "keywords": topic_keywords[str(topic["Topic"])][:10]  # Top 10 keywords
                    })
            
            # Sort by count
            topics = sorted(topics, key=lambda x: x["count"], reverse=True)[:limit]
            return topics
        except Exception as e:
            logger.error(f"Error loading topic data: {e}")
    
    # Return empty list if no data
    return []

async def get_topic_articles(topic_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """Get articles for a specific topic"""
    # Load articles from file
    articles_path = f"{settings.DATA_DIR}/news_articles.csv"
    
    if os.path.exists(articles_path):
        try:
            df = pd.read_csv(articles_path)
            
            # Filter by topic
            if "topic" in df.columns:
                topic_articles = df[df["topic"] == topic_id]
                
                # Sort by topic probability if available
                if "topic_prob" in df.columns:
                    topic_articles = topic_articles.sort_values("topic_prob", ascending=False)
                
                # Get top articles
                top_articles = topic_articles.head(limit)
                
                # Convert to list of dicts
                return top_articles[["title", "description", "url", "source", "category"]].to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error loading topic articles: {e}")
    
    # Return empty list if no data
    return []

# For importing
import pandas as pd
