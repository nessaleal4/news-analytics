from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # Qdrant configuration
    QDRANT_URL: str = "https://637e29f4-1f03-4987-95c7-ba4e8a2d30ba.us-east-1-0.aws.cloud.qdrant.io:6333"
    QDRANT_API_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQyNjE4OTE5fQ.2YsYMECAs5a88i8OSEUKIaECydbrgaSOcs_gHnVucSQ"
    QDRANT_COLLECTION_NAME: str = "news_articles"
    
    # Model configuration
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Scraping configuration
    SCRAPE_INTERVAL_HOURS: int = 12  # Scrape every 12 hours
    MIN_ARTICLES_PER_SOURCE: int = 30
    
    # API configuration
    CORS_ORIGINS: list = ["*"]
    
    # Storage paths
    DATA_DIR: str = "backend/app/data"
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    """Get application settings, using cache for efficiency"""
    return Settings()
