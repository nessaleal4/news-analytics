# News Analytics

A comprehensive platform for analyzing news articles using NLP techniques, featuring topic modeling, semantic search, and knowledge graph visualization.

## Architecture

This project consists of two main components:

- **Backend**: A FastAPI application that provides API endpoints for news scraping, topic modeling, semantic search, and knowledge graph generation.
- **Frontend**: A Streamlit application that provides a user-friendly interface for exploring the analyzed news data.

## Directory Structure

```
news-analytics/
│
├── backend/                  # FastAPI backend
│   ├── app/                  # Main application code
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI entry point
│   │   ├── config.py         # Configuration settings
│   │   ├── routers/          # API route definitions
│   │   ├── schemas/          # Data models
│   │   └── services/         # Business logic
│   ├── requirements.txt      # Backend dependencies
│   └── railway.json          # Railway deployment configuration
│
├── frontend/                 # Streamlit frontend
│   ├── app.py                # Main Streamlit application
│   ├── utils/                # Utility functions
│   │   └── api.py            # API client for backend communication
│   └── requirements.txt      # Frontend dependencies
│
├── streamlit_app.py          # Entry point for Streamlit Cloud
└── README.md                 # Project documentation
```

## Features

- **Semantic Search**: Find relevant news articles based on meaning, not just keywords
- **Topic Modeling**: Discover themes and topics across news articles using BERTopic
- **Knowledge Graph**: Visualize connections between entities mentioned in the news
- **Real-time Scraping**: Automatically scrape and process news articles every 12 hours

## Technology Stack

- **Backend**:
  - FastAPI: Modern, high-performance web framework
  - Sentence Transformers: For generating embeddings
  - BERTopic: For topic modeling
  - spaCy & NLTK: For NLP tasks and entity extraction
  - Qdrant: Vector database for semantic search
  - Deployed on Railway

- **Frontend**:
  - Streamlit: For interactive data visualization
  - Plotly: For charts and graphs
  - PyVis: For network visualization
  - Deployed on Streamlit Cloud

## Deployment

### Backend (Railway)

The backend is deployed on Railway with the following environment variables:

- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant authentication
- `QDRANT_COLLECTION_NAME`: Name of the collection in Qdrant
- `EMBEDDING_MODEL`: Name of the Sentence Transformer model
- `DATA_DIR`: Directory for storing processed data
- `SCRAPE_INTERVAL_HOURS`: Interval for scraping news (in hours)
- `MIN_ARTICLES_PER_SOURCE`: Minimum number of articles to scrape per source
- `CORS_ORIGINS`: CORS configuration

### Frontend (Streamlit Cloud)

The frontend is deployed on Streamlit Cloud with the following configuration:

- Main file path: `streamlit_app.py`
- Required secret: `BACKEND_URL` (URL of the Railway-deployed backend)

## Local Development

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## API Documentation

When running locally or on Railway, API documentation is available at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`

## Credits

This project uses:
- CNN for news data
- Several open-source libraries for NLP and data visualization
