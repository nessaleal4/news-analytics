from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import logging
from typing import List, Union

logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
_model = None
_tokenizer = None

def get_embedding_model():
    """Get or initialize the MPNet model"""
    global _model, _tokenizer
    if _model is None:
        logger.info("Loading MPNet model...")
        model_name = "microsoft/mpnet-base"
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModel.from_pretrained(model_name)
        logger.info("MPNet model loaded")
    
    return _model, _tokenizer

def mean_pooling(model_output, attention_mask):
    """Mean Pooling - Take attention mask into account for correct averaging"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

async def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using MPNet"""
    if not texts:
        return np.array([])
    
    model, tokenizer = get_embedding_model()
    
    # Convert to list of strings if any items are not strings
    texts = [str(text) if not isinstance(text, str) else text for text in texts]
    
    # Tokenize texts
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    
    # Perform mean pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    # Normalize embeddings
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    
    return sentence_embeddings.numpy()

async def generate_embedding(text: str) -> List[float]:
    """Generate embedding for a single text"""
    embeddings = await generate_embeddings([text])
    return embeddings[0].tolist()
