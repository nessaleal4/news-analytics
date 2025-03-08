from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

# Global variable for the model
_model = None

def get_sentence_transformer_model():
    """Load and cache the MPNet model for sentence embeddings."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-mpnet-base-v2")
    return _model

def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for a list of texts using MPNet."""
    if not texts:
        return np.array([])
    model = get_sentence_transformer_model()
    # Generate embeddings (this is a synchronous call)
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def generate_embedding(text: str) -> List[float]:
    """Generate an embedding for a single text and return it as a list of floats."""
    embeddings = generate_embeddings([text])
    return embeddings[0].tolist()


# Example usage
if __name__ == "__main__":
    sentences = ["This is an example sentence", "Each sentence is converted"]
    embeddings = generate_embeddings(sentences)
    print(embeddings)
