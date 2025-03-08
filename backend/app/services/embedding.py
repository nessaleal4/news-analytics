# Replace your embedding.py file with this simplified version
import numpy as np
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

# Dummy embedding function that returns random embeddings
async def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate dummy embeddings for a list of texts"""
    # For demo purposes, use a fixed dimension
    embedding_dim = 384
    
    # Create random embeddings
    embeddings = np.random.randn(len(texts), embedding_dim)
    # Normalize the embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    logger.info(f"Generated dummy embeddings for {len(texts)} texts")
    return embeddings

async def generate_embedding(text: str) -> List[float]:
    """Generate dummy embedding for a single text"""
    embeddings = await generate_embeddings([text])
    return embeddings[0].tolist()
