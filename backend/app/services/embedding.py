# Open backend/app/services/embedding.py
# Replace the first few lines from:
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Union
import torch

# With this alternative implementation:
import numpy as np
import logging
from typing import List, Union
import torch

# Instead of importing SentenceTransformer directly, use a workaround
def get_embedding_model():
    """Get or initialize the embedding model with a workaround for huggingface_hub issues"""
    import os
    # Force huggingface to use local cache only to avoid the problematic API call
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # Now import SentenceTransformer after setting environment variables
    from sentence_transformers import SentenceTransformer
    
    model_name = "all-MiniLM-L6-v2"
    device = torch.device("cpu")  # Force CPU usage on Render
    model = SentenceTransformer(model_name, device=device)
    
    return model
