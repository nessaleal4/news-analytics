from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import os
import json
import pandas as pd

from backend.app.config import get_settings
from backend.app.services.knowledge_graph import get_knowledge_graph

router = APIRouter()
logger = logging.getLogger(__name__)
settings = get_settings()

class KnowledgeGraphResponse(BaseModel):
    nodes: List[Dict[str, Any]]
    links: List[Dict[str, Any]]

@router.get("/", response_model=KnowledgeGraphResponse)
async def get_graph():
    """Get the knowledge graph data"""
    try:
        graph = await get_knowledge_graph()
        return graph
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entities", response_model=Dict[str, List[Dict[str, Any]]])
async def get_top_entities(limit: int = Query(20, ge=1, le=100)):
    """Get the top entities from the knowledge graph"""
    try:
        graph = await get_knowledge_graph()
        
        # Sort nodes by count
        nodes = sorted(graph.get("nodes", []), key=lambda x: x.get("count", 0), reverse=True)
        
        # Return top entities
        return {"entities": nodes[:limit]}
    except Exception as e:
        logger.error(f"Error getting top entities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/entity/{entity_name}", response_model=Dict[str, Any])
async def get_entity_connections(entity_name: str):
    """Get all connections for a specific entity"""
    try:
        graph = await get_knowledge_graph()
        
        # Find the entity node
        entity_node = None
        for node in graph.get("nodes", []):
            if node.get("id", "").lower() == entity_name.lower():
                entity_node = node
                break
        
        if not entity_node:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
        
        # Find all connections
        connections = []
        for link in graph.get("links", []):
            source = link.get("source", "")
            target = link.get("target", "")
            
            if source.lower() == entity_name.lower():
                # Find the target node
                target_node = None
                for node in graph.get("nodes", []):
                    if node.get("id", "").lower() == target.lower():
                        target_node = node
                        break
                
                if target_node:
                    connections.append({
                        "entity": target,
                        "strength": link.get("value", 1),
                        "count": target_node.get("count", 0)
                    })
            
            elif target.lower() == entity_name.lower():
                # Find the source node
                source_node = None
                for node in graph.get("nodes", []):
                    if node.get("id", "").lower() == source.lower():
                        source_node = node
                        break
                
                if source_node:
                    connections.append({
                        "entity": source,
                        "strength": link.get("value", 1),
                        "count": source_node.get("count", 0)
                    })
        
        # Sort connections by strength
        connections.sort(key=lambda x: x.get("strength", 0), reverse=True)
        
        return {
            "entity": entity_name,
            "count": entity_node.get("count", 0),
            "connections": connections
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity connections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats", response_model=Dict[str, Any])
async def get_graph_stats():
    """Get statistics about the knowledge graph"""
    try:
        graph = await get_knowledge_graph()
        
        nodes = graph.get("nodes", [])
        links = graph.get("links", [])
        
        # Count entity types
        entity_types = {}
        for node in nodes:
            entity_type = node.get("group", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # Get strongest connections
        strongest_links = sorted(links, key=lambda x: x.get("value", 0), reverse=True)[:10]
        
        return {
            "total_entities": len(nodes),
            "total_connections": len(links),
            "entity_types": entity_types,
            "strongest_connections": [
                {
                    "source": link.get("source", ""),
                    "target": link.get("target", ""),
                    "strength": link.get("value", 0)
                }
                for link in strongest_links
            ]
        }
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
