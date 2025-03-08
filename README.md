[news-analytics-vanessa-nlp](https://news-analytics-vanessa-nlp.streamlit.app/)

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
├── backend/
│   └── app/                     # FastAPI application
│       ├── __init__.py
│       ├── main.py              # FastAPI entry point
│       ├── config.py            # Configuration
│       ├── Dockerfile           # Docker configuration for Fly.io
│       ├── fly.toml             # Fly.io configuration
│       ├── requirements.txt     # Backend dependencies
│       ├── routers/             # API route definitions
│       │   ├── __init__.py
│       │   ├── search.py
│       │   ├── topics.py
│       │   └── knowledge.py
│       ├── services/            # Business logic
│       │   ├── __init__.py
│       │   ├── embedding.py
│       │   ├── qdrant.py
│       │   ├── scraper.py
│       │   ├── topic_model.py
│       │   └── knowledge_graph.py
│       └── schemas/             # Data models
│           ├── __init__.py
│           └── models.py
├── frontend/
│   ├── app.py                   # Main Streamlit app
│   ├── requirements.txt         # Frontend dependencies
│   └── utils/                   # Utility functions
│       ├── __init__.py
│       └── api.py               # API client for backend
└── streamlit_app.py             # Entry point for Streamlit Cloud
```

## Deployment Instructions

### 1. Deploy Backend to Fly.io

1. **Fork or clone this repository to your GitHub account**

2. **Deploy using Fly.io Dashboard**
   - Log in to [Fly.io](https://fly.io/dashboard)
   - Click "Create App"
   - Choose "Deploy from GitHub repository"
   - Select your news-analytics repository
   - Specify `backend/app` as the Current Working Directory
   - Select "None" when asked about PostgreSQL
   - Create a volume for data storage with:
     ```
     fly volumes create news_analytics_data --size 1 --region <your-region>
     ```

3. **Verify the deployment**
   - Once deployed, Fly.io will provide a URL for your application (e.g., `https://news-analytics.fly.dev`)
   - Test the health endpoint: `https://news-analytics.fly.dev/api/health`

### 2. Deploy Frontend to Streamlit Cloud

1. **Connect Streamlit Cloud to your GitHub repository**
   - Log in to [Streamlit Cloud](https://streamlit.io/cloud)
   - Click "New app"
   - Connect to your GitHub repository
   - Select `streamlit_app.py` as the main app

2. **Configure the deployment**
   - Set the Python version (3.9+ recommended)
   - Add the following secrets in the Streamlit dashboard:
     - `API_URL`: Your Fly.io backend URL (e.g., `https://news-analytics.fly.dev`)

3. **Deploy the application**
   - Click "Deploy"
   - Streamlit will build and deploy your application

## Using the Application

### Search Engine

The search engine provides two methods to find articles:

1. **Semantic Search**: Uses vector embeddings to find articles based on meaning
2. **Keyword Search**: Finds articles containing specific words or phrases

### Topics

The topics feature allows you to:

1. View top topics detected in the news
2. Explore keywords associated with each topic
3. Browse articles related to specific topics

### Knowledge Graph

The knowledge graph visualizes connections between entities mentioned in news articles:

1. Explore important people, organizations, and places
2. View relationship strength between connected entities
3. Analyze the overall knowledge network

## Technologies Used

- **Backend**: FastAPI, BERTopic, spaCy, Qdrant, Sentence Transformers
- **Frontend**: Streamlit, Plotly, Pyvis
- **Data Storage**: Qdrant Cloud (vector database)
- **Deployment**: Fly.io, Streamlit Cloud

## Acknowledgments

This project was developed as part of a data science course assignment.
This project was developed as part of a data science course in Natural Language Processing assignment.
