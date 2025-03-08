import logging
from typing import List, Dict, Any
import os
import json
import asyncio
from collections import Counter
import spacy
import nltk
from tqdm import tqdm
import pandas as pd

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Global NLP model
_nlp = None

def get_nlp_model():
    """Get or initialize the spaCy NLP model"""
    global _nlp
    if _nlp is None:
        try:
            # Try to load the model
            logger.info("Loading spaCy model...")
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
        except Exception as e:
            # If model is not found, download it
            logger.warning(f"Error loading spaCy model: {e}")
            logger.info("Downloading spaCy model...")
            try:
                os.system("python -m spacy download en_core_web_sm")
                _nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model downloaded and loaded")
            except Exception as e2:
                logger.error(f"Error downloading spaCy model: {e2}")
                logger.info("Using NLTK as fallback")
                # Make sure we have NLTK resources
                nltk.download("punkt", quiet=True)
                nltk.download("averaged_perceptron_tagger", quiet=True)
                nltk.download("maxent_ne_chunker", quiet=True)
                nltk.download("words", quiet=True)
                _nlp = None
    
    return _nlp

async def extract_entities_with_spacy(text: str) -> List[tuple]:
    """Extract named entities from text using spaCy"""
    nlp = get_nlp_model()
    if nlp is None:
        return []
    
    try:
        # Process the text with spaCy
        doc = nlp(text)
        
        # Extract named entities
        entities = [(ent.text, ent.label_) for ent in doc.ents
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT']]
        
        return entities
    except Exception as e:
        logger.warning(f"Error extracting entities with spaCy: {e}")
        return []

async def extract_entities_with_nltk(text: str) -> List[tuple]:
    """Extract named entities from text using NLTK as fallback"""
    try:
        # Tokenize and tag
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        
        # Extract named entities
        entities = nltk.ne_chunk(tagged)
        
        # Process entities
        extracted = []
        for subtree in entities:
            if hasattr(subtree, 'label'):
                entity_text = ' '.join([word for word, tag in subtree.leaves()])
                entity_type = subtree.label()
                extracted.append((entity_text, entity_type))
        
        return extracted
    except Exception as e:
        logger.warning(f"Error extracting entities with NLTK: {e}")
        return []

async def extract_entities(texts: List[str], limit: int = 100) -> Dict[str, Any]:
    """Extract named entities from texts for knowledge graph"""
    all_entities = []
    entity_pairs = []

    logger.info("Extracting entities from texts...")

    # Process only a subset for performance
    texts_sample = texts[:limit] if len(texts) > limit else texts

    for text in tqdm(texts_sample):
        try:
            # Try with spaCy first
            entities = await extract_entities_with_spacy(text)
            
            # Fallback to NLTK if spaCy fails or returns nothing
            if not entities and get_nlp_model() is None:
                entities = await extract_entities_with_nltk(text)

            if entities:
                all_entities.extend(entities)

                # Create entity pairs that co-occur in the same document
                for i in range(len(entities)):
                    for j in range(i+1, min(i+5, len(entities))):  # Limit to nearby entities
                        entity_pairs.append((
                            entities[i][0], 
                            entities[j][0], 
                            entities[i][1], 
                            entities[j][1]
                        ))
        except Exception as e:
            logger.warning(f"Error processing text for entities: {e}")
            continue

    # Count entity frequencies
    entity_counts = Counter([entity[0].lower() for entity in all_entities])

    # Get top entities
    top_entities = [entity for entity, count in entity_counts.most_common(50) if count > 1]

    # Filter pairs to only include top entities
    filtered_pairs = []
    for e1, e2, t1, t2 in entity_pairs:
        if e1.lower() in top_entities and e2.lower() in top_entities:
            filtered_pairs.append((e1, e2, t1, t2))

    # Count pair frequencies
    pair_counts = Counter([(p[0].lower(), p[1].lower()) for p in filtered_pairs])

    # Create knowledge graph data
    nodes = [{"id": entity, "group": 1, "count": count} 
             for entity, count in entity_counts.most_common(50) if count > 1]
    
    links = [{"source": source, "target": target, "value": count}
             for (source, target), count in pair_counts.most_common(100) if count > 1]

    return {"nodes": nodes, "links": links}

async def get_knowledge_graph() -> Dict[str, Any]:
    """Get the knowledge graph data"""
    # Load from file if available
    kg_path = f"{settings.DATA_DIR}/knowledge_graph.json"
    
    if os.path.exists(kg_path):
        try:
            with open(kg_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading knowledge graph data: {e}")
    
    # Try to generate from articles if file doesn't exist
    articles_path = f"{settings.DATA_DIR}/news_articles.csv"
    if os.path.exists(articles_path):
        try:
            df = pd.read_csv(articles_path)
            texts = df["text"].tolist()
            return await extract_entities(texts, limit=100)
        except Exception as e:
            logger.error(f"Error generating knowledge graph: {e}")
    
    # Return empty graph if all else fails
    return {"nodes": [], "links": []}
