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
        self._load_local_embeddings()
    
    def _load_local_embeddings(self):
        """Load local embeddings from embeddings.npy"""
        try:
            embeddings_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'embeddings.npy')
            if os.path.exists(embeddings_path):
                self.embeddings = np.load(embeddings_path)
                self.logger.info(f"Loaded {len(self.embeddings)} local embeddings")
            else:
                self.logger.warning("Embeddings file not found")
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
    
    def _semantic_local_search(self, query_embedding, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search using local embeddings"""
        if not hasattr(st.session_state, "local_data") or not self.embeddings:
            return []
        
        df = st.session_state.local_data["news_articles"]
        
        # Calculate cosine similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            try:
                # Use 1 - cosine distance to get similarity score
                sim = 1 - cosine(query_embedding, doc_embedding)
                similarities.append((i, sim))
            except Exception as e:
                self.logger.error(f"Error calculating similarity for doc {i}: {e}")
        
        # Sort by similarity and get top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in similarities[:limit]:
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
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["news_articles"] is None:
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
                return self._semantic_local_search(query_embedding, limit)
            except ImportError:
                # Fallback to keyword search if embedding fails
                return self._local_search(query, limit)
        
        # Existing API call logic remains the same
        try:
            url = f"{self.base_url}/api/search/semantic"
            payload = {"query": query, "limit": limit}
            
            response = requests.post(url, json=payload, timeout=10)
            data = self._handle_response(response)
            return data.get("results", [])
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            # Fallback to local search
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
            # Fallback to local data
            return self._get_local_recent_articles(limit)
    
    def _get_local_recent_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent articles from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["news_articles"] is None:
            return []
        
        df = st.session_state.local_data["news_articles"]

```python
                    topic_info:
                        topic_id = topic.get("Topic", -1)
                        topic_id_str = str(topic_id)
                        
                        # Get keywords for this topic if available
                        keywords = []
                        if topic_id_str in topic_keywords:
                            for item in topic_keywords[topic_id_str]:
                                keywords.append({"word": item["word"], "score": item["score"]})
                        
                        topics.append({
                            "id": topic_id,
                            "count": topic.get("Count", 0),
                            "keywords": keywords,
                            "name": topic.get("Name", f"Topic {topic_id}")
                        })
            else:
                # Dictionary format processing
                for topic_id, info in topic_info.items():
                    topic_id_int = int(topic_id)
                    
                    # Get keywords for this topic if available
                    keywords = []
                    if topic_keywords and str(topic_id_int) in topic_keywords:
                        for item in topic_keywords[str(topic_id_int)]:
                            if isinstance(item, dict):
                                # Handle the case where items are already dictionaries
                                keywords.append({"word": item["word"], "score": item["score"]})
                            else:
                                # Handle the case where items are [word, score] pairs
                                word, score = item
                                keywords.append({"word": word, "score": score})
                    
                    topics.append({
                        "id": topic_id_int,
                        "count": info.get("count", 0) if isinstance(info, dict) else info,
                        "keywords": keywords
                    })
        except Exception as e:
            self.logger.error(f"Error processing topic data: {e}")
            st.error(f"Error processing topic data: {e}")
            return []
        
        return topics
    
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
            return self._get_local_top_entities(limit)
    
    def _get_local_top_entities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top entities from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["knowledge_graph"] is None:
            return []
        
        graph = st.session_state.local_data["knowledge_graph"]

graph"]
        
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
            return self._get_local_entity_connections(entity)
    
    def _get_local_entity_connections(self, entity: str) -> Dict[str, Any]:
        """Get entity connections from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["knowledge_graph"] is None:
            return {"id": entity, "count": 0, "connections": []}
        
        graph = st.session_state.local_data["knowledge_graph"]
        
        # Find the entity node
        node = next((n for n in graph.get("nodes", []) if n.get("id") == entity), None)
        
        if not node:
            return {"id": entity, "count": 0, "connections": []}
        
        # Find all links with this entity
        connections = []
        for link in graph.get("links", []):
            if link.get("source") == entity:
                target = link.get("target")
                strength = link.get("value", 1)
                connections.append({"entity": target, "strength": strength})
            elif link.get("target") == entity:
                source = link.get("source")
                strength = link.get("value", 1)
                connections.append({"entity": source, "strength": strength})
        
        # Sort by strength
        connections.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "id": entity,
            "count": node.get("count", 0),
            "connections": connections
        }
```
