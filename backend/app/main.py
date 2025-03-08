from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
import os
import logging

from app.config import get_settings
from app.routers import search, topics, knowledge
from app.services.scraper import scrape_and_process_news

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="News Analytics API",
    description="API for news search, topic modeling, and knowledge graph",
    version="1.0.0",
)

# Set up CORS
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router, prefix="/api/search", tags=["search"])
app.include_router(topics.router, prefix="/api/topics", tags=["topics"])
app.include_router(knowledge.router, prefix="/api/knowledge", tags=["knowledge"])

# Create data directory if it doesn't exist
os.makedirs(settings.DATA_DIR, exist_ok=True)

# Schedule regular news scraping
scheduler = BackgroundScheduler()

@app.on_event("startup")
async def startup_event():
    # Initial scraping on startup
    logger.info("Performing initial news scraping...")
    try:
        await scrape_and_process_news()
        logger.info("Initial scraping complete")
    except Exception as e:
        logger.error(f"Error during initial scraping: {e}")
    
    # Schedule periodic scraping
    scheduler.add_job(
        scrape_and_process_news, 
        'interval', 
        hours=settings.SCRAPE_INTERVAL_HOURS,
        id='scrape_news'
    )
    scheduler.start()
    logger.info(f"Scheduled news scraping every {settings.SCRAPE_INTERVAL_HOURS} hours")

@app.on_event("shutdown")
def shutdown_event():
    scheduler.shutdown()
    logger.info("Shutting down scheduler")

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
