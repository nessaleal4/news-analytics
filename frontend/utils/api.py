import requests
import json
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class APIClient:
    """Client for interacting with the News Analytics API"""
    
    def __init__(self, base_url: str):
        """Initialize API client with base URL"""
        self.base_url = base_url.rstrip('/')
    
    def _handle_response(self, response: requests.Response) -> Dict[str, Any]:
        """Handle API response and errors"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            try:
                error_data = response.json()
                logger.error(f"API error: {error_data}")
            except:
                logger.error(f"API error: {response.text}")
            raise e
        except json.JSONDecodeError:
            logger.error(f"Error decoding API response: {response.text}")
            raise ValueError("Invalid response from API")
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for articles semantically"""
        url = f"{self.base_url}/api/search/semantic"
        payload = {"query": query, "limit": limit}
        
        try:
            response = requests.post(url, json=payload)
            data = self._handle_response(response)
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def keyword_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for articles by keyword"""
        url = f"{self.base_url}/api/search/keyword"
        payload = {"query": query, "limit": limit}
        
        try:
            response = requests.post(url, json=payload)
            data = self._handle_response(response)
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def get_recent_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent articles"""
        url = f"{self.base_url}/api/search/recent?limit={limit}"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error getting recent articles: {e}")
            return []
    
    def get_categories(self) -> Dict[str, List[str]]:
        """Get available categories and sources"""
        url = f"{self.base_url}/api/search/categories"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return {"categories": [], "sources": []}
    
    def filter_articles(
        self, 
        category: Optional[str] = None, 
        source: Optional[str] = None, 
        query: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Filter articles by category and/or source"""
        url = f"{self.base_url}/api/search/filter"
        params = {"limit": limit}
        
        if category:
            params["category"] = category
        if source:
            params["source"] = source
        if query:
            params["query"] = query
        
        try:
            response = requests.post(url, params=params)
            data = self._handle_response(response)
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Error filtering articles: {e}")
            return []
    
    def get_topics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get list of topics"""
        url = f"{self.base_url}/api/topics?limit={limit}"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data.get("topics", [])
        except Exception as e:
            logger.error(f"Error getting topics: {e}")
            return []
    
    def get_topic_articles(self, topic_id: int, limit: int = 10) -> List[Dict[str, Any]]:
        """Get articles for a specific topic"""
        url = f"{self.base_url}/api/topics/{topic_id}/articles?limit={limit}"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data.get("articles", [])
        except Exception as e:
            logger.error(f"Error getting topic articles: {e}")
            return []
    
    def get_topic_summary(self) -> Dict[str, Any]:
        """Get topic summary"""
        url = f"{self.base_url}/api/topics/summary"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data
        except Exception as e:
            logger.error(f"Error getting topic summary: {e}")
            return {"total_articles": 0, "topic_counts": {}}
    
    def get_topic_keywords(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Get keywords for all topics"""
        url = f"{self.base_url}/api/topics/keywords"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data
        except Exception as e:
            logger.error(f"Error getting topic keywords: {e}")
            return {"topic_keywords": {}}
    
    def get_knowledge_graph(self) -> Dict[str, Any]:
        """Get knowledge graph data"""
        url = f"{self.base_url}/api/knowledge"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data
        except Exception as e:
            logger.error(f"Error getting knowledge graph: {e}")
            return {"nodes": [], "links": []}
    
    def get_top_entities(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top entities from knowledge graph"""
        url = f"{self.base_url}/api/knowledge/entities?limit={limit}"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data.get("entities", [])
        except Exception as e:
            logger.error(f"Error getting top entities: {e}")
            return []
    
    def get_entity_connections(self, entity_name: str) -> Dict[str, Any]:
        """Get connections for a specific entity"""
        url = f"{self.base_url}/api/knowledge/entity/{entity_name}"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data
        except Exception as e:
            logger.error(f"Error getting entity connections: {e}")
            return {"entity": entity_name, "count": 0, "connections": []}
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        url = f"{self.base_url}/api/knowledge/stats"
        
        try:
            response = requests.get(url)
            data = self._handle_response(response)
            return data
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {
                "total_entities": 0, 
                "total_connections": 0, 
                "entity_types": {}, 
                "strongest_connections": []
            }
