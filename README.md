# News Analytics

A full-stack news analytics application with semantic search, topic modeling, and knowledge graph visualization.

## Project Overview

This application provides tools for analyzing news articles using modern NLP techniques:

- **Semantic Search**: Find relevant articles based on meaning, not just keywords
- **Topic Modeling**: Discover themes and topics across news articles
- **Knowledge Graph**: Explore connections between entities mentioned in the news

## Architecture

The system consists of two main components:

1. **Backend (FastAPI on Fly.io)**
   - Handles embeddings generation
   - Interfaces with Qdrant for vector search
   - Scrapes news articles
   - Performs topic modeling
   - Extracts entities for knowledge graph
   - Exposes APIs for frontend

2. **Frontend (Streamlit)**
   - Provides user interface for all features
   - Connects to backend API for data

## Repository Structure

news-analytics/
├── backend/
│   ├── app/
│   │   ├── __init__.py                     # Backend __init__.py
│   │   ├── main.py                         # FastAPI entry point
│   │   ├── config.py                       # Configuration settings
│   │   ├── routers/
│   │   │   ├── __init__.py                 # Routers __init__.py
│   │   │   ├── search.py                   # Search endpoints
│   │   │   ├── topics.py                   # Topic modeling endpoints
│   │   │   └── knowledge.py                # Knowledge graph endpoints
│   │   ├── services/
│   │   │   ├── __init__.py                 # Services __init__.py
│   │   │   ├── embedding.py                # Embedding generation service
│   │   │   ├── qdrant.py                   # Qdrant client service
│   │   │   ├── scraper.py                  # News scraping service
│   │   │   ├── topic_model.py              # Topic modeling service
│   │   │   └── knowledge_graph.py          # Knowledge graph service
│   │   └── schemas/
│   │       ├── __init__.py                 # Schemas __init__.py
│   │       └── models.py                   # Pydantic models
│   ├── requirements.txt                    # Backend dependencies
│   └── fly.toml                            # Fly.io configuration
├── frontend/
│   ├── app.py                              # Main Streamlit app
│   ├── utils/
│   │   ├── __init__.py                     # Utils __init__.py
│   │   └── api.py                          # API client
│   └── requirements.txt                    # Frontend dependencies
├── streamlit_app.py                        # Entry point for Streamlit Cloud
└── README.md                               # Project documentation
