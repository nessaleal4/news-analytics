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

```
news-analytics/
├── backend/                     # FastAPI backend (deploy to Fly.io)
│   ├── app/                     # Application code
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI entry point
│   │   ├── config.py            # Configuration
│   │   ├── routers/             # API route definitions
│   │   ├── services/            # Business logic
│   │   └── schemas/             # Data models
│   ├── fly.toml                 # Fly.io configuration
│   └── requirements.txt         # Backend dependencies
├── frontend/                    # Streamlit frontend
│   ├── app.py                   # Main Streamlit app
│   ├── utils/                   # Utility functions
│   │   └── api.py
