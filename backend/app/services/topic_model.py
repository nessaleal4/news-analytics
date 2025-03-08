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
    
    # Check if we can load a cached model
    model_path = f"{settings.DATA_DIR}/topic_model"
    
    # Define seed topics for guided topic modeling
    seed_topics = [
        ["politics", "government", "president", "election", "vote"],
        ["business", "market", "economy", "company", "stock"],
        ["sports", "team", "game", "player", "win"],
        ["technology", "tech", "innovation", "digital", "software"],
        ["health", "medical", "doctor", "patient", "disease"],
        ["entertainment", "movie", "music", "celebrity", "
