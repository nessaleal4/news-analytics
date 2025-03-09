import requests
import json
import os
from typing import Dict, List, Any, Optional
import logging
import streamlit as st
import pandas as pd
import re
import numpy as np

# Import for cosine similarity calculation
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

class APIClient:
    """Client for interacting with the News Analytics API"""
    
    def __init__(self, base_url: str):
        """Initialize API client with base URL"""
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
        self.embeddings = None
        self.logger.info(f"API Client initialized with URL: {self.base_url}")
        
        # Try to load embeddings right away
        self._load_local_embeddings()
        
        # Test API connection
        try:
            self._test_api_connection()
        except Exception as e:
            self.logger.warning(f"API connection failed: {e}")
            # Set use_local_data to True if in Streamlit context
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
    
    def _test_api_connection(self):
        """Test connection to the API"""
        try:
            # Try a simple endpoint to check connection
            url = f"{self.base_url}/api/categories"  # Use an existing endpoint
            response = requests.get(url, timeout=3)  # Short timeout for quick check
            response.raise_for_status()
            return True
        except Exception as e:
            self.logger.warning(f"API connection test failed: {e}")
            return False
    
    def _load_local_embeddings(self):
        """Load local embeddings from embeddings.npy"""
        try:
            # Try multiple possible paths for embeddings file
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings.npy'),
                os.path.join('frontend', 'data', 'embeddings.npy'),
                os.path.join('data', 'embeddings.npy'),
                os.path.join('.', 'embeddings.npy')
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.embeddings = np.load(path)
                    self.logger.info(f"Loaded {len(self.embeddings)} local embeddings from {path}")
                    return
            
            self.logger.warning("Embeddings file not found in any of the expected locations")
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
    
    def _semantic_local_search(self, query_embedding, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using local embeddings"""
        if not hasattr(st.session_state, "local_data") or self.embeddings is None:
            return []
        
        df = st.session_state.local_data.get("news_articles")
        if df is None:
            return []
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            try:
                if i < len(df):  # Make sure we don't go beyond dataframe size
                    # Use 1 - cosine distance to get similarity score
                    sim = 1 - cosine(query_embedding, doc_embedding)
                    similarities.append((i, sim))
            except Exception as e:
                self.logger.error(f"Error calculating similarity for doc {i}: {e}")
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:limit]:
            if idx < len(df):  # Safety check
                row = df.iloc[idx]
                results.append({
                    "score": float(score),
                    "payload": row.to_dict()
                })
        
        return results
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and errors"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                self.logger.error(f"API error: {error_data}")
            except:
                self.logger.error(f"API error: {response.text}")
            raise e
        except json.JSONDecodeError:
            self.logger.error(f"Error decoding API response: {response.text}")
            raise ValueError("Invalid response from API")
    
    def _local_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search in local data when API is unavailable"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data.get("news_articles") is None:
            return []
        
        df = st.session_state.local_data["news_articles"]
        query_lower = query.lower()
        results = []
        
        for i, row in df.iterrows():
            # Get text fields to search in
            title = str(row.get('title', '')).lower()
            desc = str(row.get('description', '')).lower()
            
            # Combine text fields if available, otherwise use what we have
            if 'text' in row:
                full_text = str(row['text']).lower()
            else:
                full_text = title + " " + desc
            
            # Calculate score based on keyword matches
            score = 0
            if query_lower in title:
                score += 3
            if query_lower in desc:
                score += 1
            
            # Add score for individual terms
            for term in query_lower.split():
                if len(term) > 2:  # Skip short terms
                    if term in title:
                        score += 1
                    if term in full_text:
                        score += 0.5
            
            if score > 0:
                results.append({
                    "score": float(score),
                    "payload": row.to_dict()
                })
        
        # Sort by score and limit results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for articles semantically"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Attempt to load embedding for query
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode(query)
                
                # Use local embeddings search
                local_results = self._semantic_local_search(query_embedding, limit)
                if local_results:
                    return local_results
                
                # Fallback to keyword search if semantic search returns no results
                return self._local_search(query, limit)
            except Exception as e:
                self.logger.error(f"Error in local semantic search: {e}")
                # Fallback to keyword search
                return self._local_search(query, limit)
        
        # Try API first
        try:
            url = f"{self.base_url}/api/search/semantic"
            payload = {"query": query, "limit": limit}
            
            response = requests.post(url, json=payload, timeout=10)
            data = self._handle_response(response)
            return data.get("results", [])
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            # Set use_local_data to True for future requests in Streamlit context
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            
            # Fallback to local search
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer('all-MiniLM-L6-v2')
                query_embedding = model.encode(query)
                
                local_results = self._semantic_local_search(query_embedding, limit)
                if local_results:
                    return local_results
            except Exception:
                pass
            
            # Final fallback to keyword search
            return self._local_search(query, limit)
    
    def keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for articles by keyword"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._local_search(query, limit)
        
        # Try API first
        try:
            url = f"{self.base_url}/api/search/keyword"
            payload = {"query": query, "limit": limit}
            
            response = requests.post(url, json=payload, timeout=10)
            data = self._handle_response(response)
            return data.get("results", [])
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}")
            # Set use_local_data to True for future requests in Streamlit context
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            
            # Fallback to local search
            return self._local_search(query, limit)
    
    def get_recent_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent articles"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_recent_articles(limit)
        
        # Try API first
        try:
            url = f"{self.base_url}/api/search/recent?limit={limit}"
            
            response = requests.get(url, timeout=10)
            data = self._handle_response(response)
            return data.get("results", [])
        except Exception as e:
            self.logger.error(f"Error getting recent articles: {e}")
            # Set use_local_data to True for future requests in Streamlit context
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            
            # Fallback to local data
            return self._get_local_recent_articles(limit)
    
    def _get_local_recent_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent articles from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data.get("news_articles") is None:
            return []
        
        df = st.session_state.local_data["news_articles"]
        
        # Sort by date if available
        if 'date' in df.columns:
            df = df.sort_values('date', ascending=False)
        
        # Convert top rows to format expected by frontend
        results = []
        for i, row in df.head(limit).iterrows():
            results.append({
                "score": 1.0,  # Placeholder score
                "payload": row.to_dict()
            })
        
        return results
    
    def get_topics(self) -> List[Dict[str, Any]]:
        """Get all topics"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_topics()
        
        # Try API first
        try:
            url = f"{self.base_url}/api/topics"
            
            response = requests.get(url, timeout=10)
            data = self._handle_response(response)
            return data.get("topics", [])
        except Exception as e:
            self.logger.error(f"Error getting topics: {e}")
            # Set use_local_data to True for future requests in Streamlit context
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            
            # Fallback to local data
            return self._get_local_topics()
    
    def _get_local_topics(self) -> List[Dict[str, Any]]:
        """Get topics from local data"""
        if not hasattr(st.session_state, "local_data"):
            return []
        
        topic_info = st.session_state.local_data.get("topic_info")
        topic_keywords = st.session_state.local_data.get("topic_keywords")
        
        if not topic_info:
            return []
        
        return self.process_topic_info(topic_info, topic_keywords)
    
    def process_topic_info(self, topic_info, topic_keywords):
        """Process topic info into a standardized format"""
        topics = []
        try:
            # Handle list format
            if isinstance(topic_info, list):
                for topic in topic_info:
                    topic_id = topic.get("Topic", -1)
                    if topic_id == -1:  # Skip outlier topic
                        continue
                        
                    topic_id_str = str(topic_id)
                    
                    # Get keywords for this topic if available
                    keywords = []
                    if topic_keywords and topic_id_str in topic_keywords:
                        for item in topic_keywords[topic_id_str]:
                            if isinstance(item, dict):
                                keywords.append({"word": item.get("word", ""), "score": item.get("score", 0)})
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                keywords.append({"word": item[0], "score": item[1]})
                    
                    topics.append({
                        "id": topic_id,
                        "count": topic.get("Count", 0),
                        "keywords": keywords,
                        "name": topic.get("Name", f"Topic {topic_id}")
                    })
            elif isinstance(topic_info, dict):
                # Dictionary format processing
                for topic_id, info in topic_info.items():
                    if topic_id == "-1":  # Skip outlier topic
                        continue
                        
                    try:
                        topic_id_int = int(topic_id)
                    except ValueError:
                        continue  # Skip if topic_id is not a valid integer
                    
                    # Get keywords for this topic if available
                    keywords = []
                    if topic_keywords and topic_id in topic_keywords:
                        for item in topic_keywords[topic_id]:
                            if isinstance(item, dict):
                                keywords.append({"word": item.get("word", ""), "score": item.get("score", 0)})
                            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                                keywords.append({"word": item[0], "score": item[1]})
                    
                    count = 0
                    if isinstance(info, dict):
                        count = info.get("count", 0)
                    elif isinstance(info, (int, float)):
                        count = info
                    
                    topics.append({
                        "id": topic_id_int,
                        "count": count,
                        "keywords": keywords,
                        "name": f"Topic {topic_id}"
                    })
        except Exception as e:
            self.logger.error(f"Error processing topic data: {e}")
            return []
        
        return topics
    
    def get_topic_summary(self) -> Dict[str, Any]:
        """Get summary of all topics"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_topic_summary()
        
        # Try API first
        try:
            url = f"{self.base_url}/api/topics/summary"
            
            response = requests.get(url, timeout=10)
            data = self._handle_response(response)
            return data
        except Exception as e:
            self.logger.error(f"Error getting topic summary: {e}")
            # Fallback to local data
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            return self._get_local_topic_summary()
    
    def _get_local_topic_summary(self) -> Dict[str, Any]:
        """Get topic summary from local data"""
        if not hasattr(st.session_state, "local_data"):
            return {"total_articles": 0, "total_topics": 0}
        
        articles = st.session_state.local_data.get("news_articles")
        topic_info = st.session_state.local_data.get("topic_info")
        
        total_articles = len(articles) if articles is not None else 0
        total_topics = 0
        
        if isinstance(topic_info, list):
            # Count topics excluding outlier (-1)
            total_topics = sum(1 for t in topic_info if t.get("Topic", -1) != -1)
        elif isinstance(topic_info, dict):
            # Count topics excluding outlier (-1)
            total_topics = sum(1 for k in topic_info.keys() if k != "-1")
        
        return {
            "total_articles": total_articles,
            "total_topics": total_topics
        }
    
    def get_topic_articles(self, topic_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get articles for a specific topic"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_topic_articles(topic_id, limit)
        
        # Try API first
        try:
            url = f"{self.base_url}/api/topics/{topic_id}/articles"
            params = {"limit": limit}
            
            response = requests.get(url, params=params, timeout=10)
            data = self._handle_response(response)
            return data.get("articles", [])
        except Exception as e:
            self.logger.error(f"Error getting topic articles: {e}")
            # Fallback to local data
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            return self._get_local_topic_articles(topic_id, limit)
    
    def _get_local_topic_articles(self, topic_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get articles for a topic from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["news_articles"] is None:
            return []
        
        df = st.session_state.local_data["news_articles"]
        
        # Filter by topic if available
        if 'topic' in df.columns:
            df = df[df['topic'] == topic_id]
        else:
            # If no topic column, just return some random articles
            import random
            return [row.to_dict() for i, row in df.sample(min(limit, len(df))).iterrows()]
        
        return [row.to_dict() for i, row in df.head(limit).iterrows()]
    
    def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get knowledge graph data"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_knowledge_graph()
        
        # Try API first
        try:
            url = f"{self.base_url}/api/graph/data"
            
            response = requests.get(url, timeout=10)
            data = self._handle_response(response)
            return data
        except Exception as e:
            self.logger.error(f"Error getting knowledge graph: {e}")
            # Fallback to local data
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            return self._get_local_knowledge_graph()
    
    def _get_local_knowledge_graph(self) -> Dict[str, Any]:
        """Get knowledge graph from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["knowledge_graph"] is None:
            return {"nodes": [], "links": []}
        
        try:
            graph = st.session_state.local_data["knowledge_graph"]
            
            # If topic_keywords are available, enhance node labels
            if st.session_state.local_data["topic_keywords"]:
                topic_keywords = st.session_state.local_data["topic_keywords"]
                
                for node in graph.get("nodes", []):
                    node_id = node.get("id", "")
                    if node_id in topic_keywords:
                        # Get the first keyword to use as label
                        first_keyword = topic_keywords[node_id][0]["word"] if topic_keywords[node_id] else ""
                        node["label"] = f"Topic {node_id}: {first_keyword}"
            
            return graph
        except Exception as e:
            self.logger.error(f"Error processing knowledge graph: {e}")
            return {"nodes": [], "links": []}
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_graph_stats()
        
        # Try API first
        try:
            url = f"{self.base_url}/api/graph/stats"
            
            response = requests.get(url, timeout=10)
            data = self._handle_response(response)
            return data
        except Exception as e:
            self.logger.error(f"Error getting graph stats: {e}")
            # Fallback to local data
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            return self._get_local_graph_stats()
    
    def _get_local_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph stats from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["knowledge_graph"] is None:
            return {"total_entities": 0, "total_connections": 0}
        
        graph = st.session_state.local_data["knowledge_graph"]
        
        return {
            "total_entities": len(graph.get("nodes", [])),
            "total_connections": len(graph.get("links", []))
        }
    
    def get_categories(self) -> Dict[str, List[str]]:
        """Get available news categories and sources"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_categories()
        
        # Try API first
        try:
            url = f"{self.base_url}/api/categories"
            
            response = requests.get(url, timeout=10)
            data = self._handle_response(response)
            return data
        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            # Fallback to local data
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            return self._get_local_categories()
    
    def _get_local_categories(self) -> Dict[str, List[str]]:
        """Get categories and sources from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["news_articles"] is None:
            return {"categories": [], "sources": []}
        
        df = st.session_state.local_data["news_articles"]
        categories = []
        sources = []
        
        # Extract categories if category column exists
        if 'category' in df.columns:
            categories = sorted(df['category'].dropna().unique().tolist())
        
        # Extract sources if source column exists
        if 'source' in df.columns:
            sources = sorted(df['source'].dropna().unique().tolist())
        
        # Default values if not available
        if not categories:
            categories = ["Politics", "Business", "Technology", "Science", "Health"]
        if not sources:
            sources = ["Associated Press", "Reuters", "BBC", "CNN", "Fox News"]
        
        return {
            "categories": categories,
            "sources": sources
        }
    
    def get_top_entities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top entities by mention count"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_top_entities(limit)
        
        # Try API first
        try:
            url = f"{self.base_url}/api/graph/entities"
            params = {"limit": limit}
            
            response = requests.get(url, params=params, timeout=10)
            data = self._handle_response(response)
            return data.get("entities", [])
        except Exception as e:
            self.logger.error(f"Error getting top entities: {e}")
            # Fallback to local data
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            return self._get_local_top_entities(limit)
    
    def _get_local_top_entities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top entities from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["knowledge_graph"] is None:
            return []
        
        graph = st.session_state.local_data["knowledge_graph"]
        
        # Sort nodes by count
        nodes = sorted(
            graph.get("nodes", []),
            key=lambda x: x.get("count", 0),
            reverse=True
        )
        
        return nodes[:limit]
    
    def get_entity_connections(self, entity: str) -> Dict[str, Any]:
        """Get connections for a specific entity"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_entity_connections(entity)
        
        # Try API first
        try:
            url = f"{self.base_url}/api/graph/entity/{entity}"
            
            response = requests.get(url, timeout=10)
            data = self._handle_response(response)
            return data
        except Exception as e:
            self.logger.error(f"Error getting entity connections: {e}")
            # Fallback to local data
            if hasattr(st, 'session_state'):
                st.session_state.use_local_data = True
            return self._get_local_entity_connections(entity)
    
    def _get_local_entity_connections(self, entity: str) -> Dict[str, Any]:
        """Get entity connections from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["knowledge_graph"] is None:
            return {"id": entity, "count": 0, "connections": []}
        
        graph = st.session_state.local_data["knowledge_graph"]
