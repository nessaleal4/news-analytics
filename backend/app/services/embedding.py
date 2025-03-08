from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Union
import torch

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global variable for model instance
_model = None

def get_embedding_model():
    """Get or initialize the embedding model"""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL, device=device)
        logger.info(f"Model loaded, embedding dimension: {_model.get_sentence_embedding_dimension()}")
    
    return _model

async def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts"""
    if not texts:
        return np.array([])
    
    model = get_embedding_model()
    
    # Convert to list of strings if any items are not strings
    texts = [str(text) if not isinstance(text, str) else text for text in texts]
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} texts")
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 10)
    
    return embeddings

async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text"""
    embeddings = await generate_embeddings([text])
    return embeddings[0].tolist()
