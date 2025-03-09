import requests
import json
from typing import Dict, List, Any, Optional
import logging
import streamlit as st
import pandas as pd
import re

logger = logging.getLogger(__name__)

class APIClient:
    """Client for interacting with the News Analytics API"""
    
    def __init__(self, base_url: str):
        """Initialize API client with base URL"""
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"API Client initialized with URL: {self.base_url}")
    
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
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for articles semantically"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
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
            # Fallback to local search
            return self._local_search(query, limit)
    
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
        
        # Sort by date if available, otherwise just take the first rows
        if 'scrape_date' in df.columns:
            df = df.sort_values('scrape_date', ascending=False)
        
        # Convert to list of results
        results = []
        for i, row in df.head(limit).iterrows():
            results.append({
                "score": 1.0,  # Default score
                "payload": row.to_dict()
            })
        
        return results
    
    def get_categories(self) -> Dict[str, List[str]]:
        """Get available categories and sources"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._get_local_categories()
        
        # Try API first
        try:
            url = f"{self.base_url}/api/search/categories"
            
            response = requests.get(url, timeout=10)
            data = self._handle_response(response)
            return data
        except Exception as e:
            self.logger.error(f"Error getting categories: {e}")
            # Fallback to local data
            return self._get_local_categories()
    
    def _get_local_categories(self) -> Dict[str, List[str]]:
        """Get categories from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["news_articles"] is None:
            return {"categories": [], "sources": []}
        
        df = st.session_state.local_data["news_articles"]
        
        # Extract unique categories and sources
        categories = df['category'].unique().tolist() if 'category' in df.columns else []
        sources = df['source'].unique().tolist() if 'source' in df.columns else []
        
        return {
            "categories": categories,
            "sources": sources
        }
    
    def filter_articles(
        self, 
        category: Optional[str] = None, 
        source: Optional[str] = None, 
        query: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Filter articles by category and/or source"""
        if hasattr(st.session_state, "use_local_data") and st.session_state.use_local_data:
            # Use local data
            return self._filter_local_articles(category, source, query, limit)
        
        # Try API first
        try:
            url = f"{self.base_url}/api/search/filter"
            params = {"limit": limit}
            
            if category:
                params["category"] = category
            if source:
                params["source"] = source
            if query:
                params["query"] = query
            
            response = requests.post(url, params=params, timeout=10)
            data = self._handle_response(response)
            return data.get("results", [])
        except Exception as e:
            self.logger.error(f"Error filtering articles: {e}")
            # Fallback to local data
            return self._filter_local_articles(category, source, query, limit)
    
    def _filter_local_articles(
        self, 
        category: Optional[str] = None, 
        source: Optional[str] = None, 
        query: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Filter articles from local data"""
        if not hasattr(st.session_state, "local_data") or st.session_state.local_data["news_articles"] is None:
            return []
        
        df = st.session_state.local_data["news_articles"]
        
        # Apply filters
        if category and 'category' in df.columns:
            df = df[df['category'].str.lower() == category.lower()]
        
        if source and 'source' in df.columns:
            df = df[df['source'].str.lower() == source.lower()]
        
        if query:
            query_lower = query.lower()
            mask = False
            
            if 'title' in df.columns:
                mask |= df['title'].str.lower().str.contains(query_lower, regex=False)
            
            if 'description' in df.columns:
                mask |= df['description'].str.lower().str.contains(query_lower, regex=False)
            
            if 'text' in df.columns:
                mask |= df['text'].str.lower().str.contains(query_lower, regex=False)
            
            df = df[mask]
        
        # Sort by date if available
        if 'scrape_date' in df.columns:
