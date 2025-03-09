import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
from datetime import datetime
import time
import logging

from frontend.utils.api import APIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_API_URL = "https://news-analytics-production.up.railway.app"

def load_local_data():
    """Load data from local files in the data directory when API is unavailable"""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    local_data = {
        "news_articles": None,
        "topic_info": None,
        "topic_keywords": None,
        "knowledge_graph": None
    }
    
    # Load news articles CSV
    articles_path = os.path.join(data_dir, "news_articles.csv")
    if os.path.exists(articles_path):
        try:
            local_data["news_articles"] = pd.read_csv(articles_path)
            logger.info(f"Loaded {len(local_data['news_articles'])} articles from local file")
        except Exception as e:
            logger.error(f"Error loading news articles: {e}")
    
    # Load topic info JSON
    topic_info_path = os.path.join(data_dir, "topic_info.json")
    if os.path.exists(topic_info_path):
        try:
            with open(topic_info_path, "r") as f:
                local_data["topic_info"] = json.load(f)
            logger.info(f"Loaded topic info from local file")
        except Exception as e:
            logger.error(f"Error loading topic info: {e}")
    
    # Load topic keywords JSON
    keywords_path = os.path.join(data_dir, "topic_keywords.json")
    if os.path.exists(keywords_path):
        try:
            with open(keywords_path, "r") as f:
                local_data["topic_keywords"] = json.load(f)
            logger.info(f"Loaded topic keywords from local file")
        except Exception as e:
            logger.error(f"Error loading topic keywords: {e}")
    
    # Load knowledge graph JSON
    graph_path = os.path.join(data_dir, "knowledge_graph.json")
    if os.path.exists(graph_path):
        try:
            with open(graph_path, "r") as f:
                local_data["knowledge_graph"] = json.load(f)
            logger.info(f"Loaded knowledge graph from local file")
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
    
    return local_data

# Initialize session state
if "api_client" not in st.session_state:
    # Change API_URL to BACKEND_URL to match the secret name in Streamlit Cloud
    api_url = st.secrets.get("BACKEND_URL", DEFAULT_API_URL)
    st.session_state.api_client = APIClient(api_url)
    
    # Load local data for fallback
    st.session_state.local_data = load_local_data()
    
    # Initialize use_local_data in session state
    if "use_local_data" not in st.session_state:
        st.session_state.use_local_data = False

# Page configuration
st.set_page_config(
    page_title="News Analytics",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Header
    st.title("📰 News Analytics")
    st.markdown("""
    Explore news articles with semantic search, topic modeling, and knowledge graphs.
    """)
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        app_mode = st.radio(
            "Choose a feature",
            ["Search", "Topics", "Knowledge Graph", "About"]
        )
        
        st.markdown("---")
        
        # API URL configuration (for development)
        with st.expander("API Configuration"):
            api_url = st.text_input(
                "API URL",
                value=DEFAULT_API_URL,
                help="URL of the News Analytics API"
            )
            
            if st.button("Update API URL"):
                st.session_state.api_client = APIClient(api_url)
                st.success(f"API URL updated to {api_url}")
            
            # Option to toggle between API and local data
            # Make sure use_local_data exists in session state
            if "use_local_data" not in st.session_state:
                st.session_state.use_local_data = False
                
            use_local = st.checkbox("Use local data", value=st.session_state.use_local_data)
            if use_local != st.session_state.use_local_data:
                st.session_state.use_local_data = use_local
                if use_local:
                    st.info("Using local data files from the data directory")
                else:
                    st.info("Using remote API")
    
    # Main content
    if app_mode == "Search":
        render_search_page()
    elif app_mode == "Topics":
        render_topics_page()
    elif app_mode == "Knowledge Graph":
        render_knowledge_graph_page()
    else:
        render_about_page()

def render_search_page():
    st.header("Search News Articles")
    
    # Search options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input("Enter your search query:", value="")
        
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["Semantic", "Keyword"],
            index=0
        )
    
    # Advanced filters
    with st.expander("Advanced Filters"):
        # Get categories and sources
        categories_data = st.session_state.api_client.get_categories()
        categories = ["All"] + categories_data.get("categories", [])
        sources = ["All"] + categories_data.get("sources", [])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            category = st.selectbox("Category", categories, index=0)
        
        with col2:
            source = st.selectbox("Source", sources, index=0)
        
        with col3:
            limit = st.slider("Max Results", 5, 50, 10)
    
    # Search button
    search_pressed = st.button("Search", type="primary")
    
    # Display results
    if search_pressed and query:
        with st.spinner("Searching..."):
            # Prepare filter parameters
            filter_category = None if category == "All" else category
            filter_source = None if source == "All" else source
            
            # Perform search based on type
            if search_type == "Semantic":
                results = st.session_state.api_client.semantic_search(query, limit=limit)
            else:
                results = st.session_state.api_client.keyword_search(query, limit=limit)
            
            # Apply filters if selected
            if filter_category or filter_source:
                filtered_results = []
                for result in results:
                    payload = result.get("payload", {})
                    if filter_category and payload.get("category") != filter_category:
                        continue
                    if filter_source and payload.get("source") != filter_source:
                        continue
                    filtered_results.append(result)
                results = filtered_results[:limit]
            
            # Display results
            st.subheader(f"Search Results ({len(results)})")
            
            if not results:
                st.info("No results found. Try a different query or search type.")
            
            for result in results:
                score = result.get("score", 0)
                payload = result.get("payload", {})
                
                with st.container():
                    col1, col2 = st.columns([5, 1])
                    
                    with col1:
                        st.markdown(f"### {payload.get('title', 'No Title')}")
                    
                    with col2:
                        st.text(f"Score: {score:.2f}")
                    
                    st.markdown(f"**Source:** {payload.get('source', 'Unknown')} | **Category:** {payload.get('category', 'Unknown')}")
                    st.markdown(f"{payload.get('description', '')[:500]}...")
                    
                    if payload.get("url"):
                        st.markdown(f"[Read more]({payload.get('url')})")
                    
                    st.markdown("---")
    else:
        # Show recent articles
        st.subheader("Recent Articles")
        
        with st.spinner("Loading recent articles..."):
            recent = st.session_state.api_client.get_recent_articles(limit=5)
            
            if not recent:
                st.info("No recent articles available. The API may be unavailable or still initializing.")
            
            for result in recent:
                payload = result.get("payload", {})
                
                with st.container():
                    st.markdown(f"### {payload.get('title', 'No Title')}")
                    st.markdown(f"**Source:** {payload.get('source', 'Unknown')} | **Category:** {payload.get('category', 'Unknown')}")
                    st.markdown(f"{payload.get('description', '')[:300]}...")
                    
                    if payload.get("url"):
                        st.markdown(f"[Read more]({payload.get('url')})")
                    
                    st.markdown("---")

def render_topics_page():
    st.header("News Topics")
    
    # Get topic summary
    with st.spinner("Loading topic data..."):
        topic_summary = st.session_state.api_client.get_topic_summary()
        topics = st.session_state.api_client.get_topics()
    
    # Display summary metrics
    total_articles = topic_summary.get("total_articles", 0)
    total_topics = len(topics)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Articles", total_articles)
    
    with col2:
        st.metric("Total Topics", total_topics)
    
    # Topic visualization
    if topics:
        # Prepare data for chart
        chart_data = []
        for topic in topics:
            chart_data.append({
                "Topic": f"Topic {topic.get('id')}",
                "Count": topic.get("count", 0),
                "ID": topic.get("id")
            })
        
        df_chart = pd.DataFrame(chart_data)
        
        # Sort by count
        df_chart = df_chart.sort_values("Count", ascending=False)
        
        # Create bar chart
        fig = px.bar(
            df_chart, 
            x="Topic", 
            y="Count",
            title="Articles by Topic",
            color="Count",
            color_continuous_scale="Viridis"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Topic details
        st.subheader("Topic Details")
        
        # Select topic
        topic_options = [f"Topic {t.get('id')}: {t.get('keywords', [])[0]['word'] if t.get('keywords') else ''}" 
                         for t in topics]
        selected_topic = st.selectbox("Select a topic to explore", topic_options)
        
        if selected_topic:
            # Extract topic ID
            topic_id = int(selected_topic.split(':')[0].replace('Topic ', '').strip())
            
            # Find selected topic
            topic = next((t for t in topics if t.get('id') == topic_id), None)
            
            if topic:
                # Display keywords
                st.markdown("#### Top Keywords")
                
                keywords = topic.get("keywords", [])
                keyword_data = []
                
                for kw in keywords:
                    keyword_data.append({
                        "Keyword": kw.get("word", ""),
                        "Score": kw.get("score", 0)
                    })
                
                df_keywords = pd.DataFrame(keyword_data)
                
                # Create horizontal bar chart for keywords
                fig = px.bar(
                    df_keywords,
                    x="Score",
                    y="Keyword",
                    orientation='h',
                    title=f"Keywords for Topic {topic_id}",
                    color="Score",
                    color_continuous_scale="Viridis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display articles for this topic
                st.markdown("#### Articles in this Topic")
                
                with st.spinner("Loading topic articles..."):
                    articles = st.session_state.api_client.get_topic_articles(topic_id)
                
                for article in articles:
                    with st.container():
                        st.markdown(f"### {article.get('title', 'No Title')}")
                        st.markdown(f"**Source:** {article.get('source', 'Unknown')} | **Category:** {article.get('category', 'Unknown')}")
                        st.markdown(f"{article.get('description', '')[:300]}...")
                        
                        if article.get("url"):
                            st.markdown(f"[Read more]({article.get('url')})")
                        
                        st.markdown("---")
    else:
        st.info("No topic data available. The API may be unavailable or still initializing.")

def render_knowledge_graph_page():
    st.header("Knowledge Graph")
    
    # Get knowledge graph data
    with st.spinner("Loading knowledge graph..."):
        graph = st.session_state.api_client.get_knowledge_graph()
        graph_stats = st.session_state.api_client.get_graph_stats()
    
    # Display stats
    total_entities = graph_stats.get("total_entities", 0)
    total_connections = graph_stats.get("total_connections", 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Entities", total_entities)
    
    with col2:
        st.metric("Total Connections", total_connections)
    
    # Entity exploration
    st.subheader("Entity Explorer")
    
    # Get top entities
    with st.spinner("Loading entities..."):
        entities = st.session_state.api_client.get_top_entities()
    
    if entities:
        # Prepare entity options
        entity_options = [f"{e.get('id')} ({e.get('count', 0)})" for e in entities]
        selected_entity = st.selectbox("Select an entity to explore", entity_options)
        
        if selected_entity:
            # Extract entity name
            entity_name = selected_entity.split(' (')[0]
            
            # Get entity connections
            with st.spinner(f"Loading connections for {entity_name}..."):
                entity_data = st.session_state.api_client.get_entity_connections(entity_name)
            
            # Display entity info
            st.markdown(f"#### {entity_name}")
            st.markdown(f"**Mentions:** {entity_data.get('count', 0)}")
            
            # Display connections
            connections = entity_data.get("connections", [])
            
            if connections:
                st.markdown("#### Connected Entities")
                
                # Prepare data for chart
                conn_data = []
                for conn in connections:
                    conn_data.append({
                        "Entity": conn.get("entity", ""),
                        "Strength": conn.get("strength", 0)
                    })
                
                df_conn = pd.DataFrame(conn_data)
                
                # Sort and limit
                df_conn = df_conn.sort_values("Strength", ascending=False).head(10)
                
                # Create bar chart
                fig = px.bar(
                    df_conn,
                    x="Entity",
                    y="Strength",
                    title=f"Top Connections for {entity_name}",
                    color="Strength",
                    color_continuous_scale="Viridis"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No connections found for {entity_name}")
    else:
        st.info("No entity data available. The API may be unavailable or still initializing.")
    
    # Knowledge Graph Visualization
    st.subheader("Graph Visualization")
    
    if graph.get("nodes") and graph.get("links"):
        # Create network visualization with pyvis
        from pyvis.network import Network
        import streamlit.components.v1 as components
        
        # Create network
        net = Network(height="600px", width="100%", notebook=True, directed=False)
        
        # Add nodes
        for node in graph.get("nodes", [])[:50]:  # Limit to 50 nodes for performance
            node_id = node.get("id", "")
            net.add_node(
                node_id, 
                label=node_id,
                title=f"{node_id} (Mentions: {node.get('count', 0)})",
                size=10 + (node.get("count", 1) * 2)
            )
        
        # Add edges
        for link in graph.get("links", [])[:100]:  # Limit to 100 edges
            source = link.get("source", "")
            target = link.get("target", "")
            value = link.get("value", 1)
            
            if source in [node.get("id") for node in graph.get("nodes", [])[:50]] and \
               target in [node.get("id") for node in graph.get("nodes", [])[:50]]:
                net.add_edge(source, target, value=value, width=1 + value)
        
        # Set physics layout
        net.barnes_hut(spring_length=200)
        
        # Generate HTML
        html_path = "knowledge_graph.html"
        net.save_graph(html_path)
        
        # Display in Streamlit
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        
        components.html(html, height=600)
    else:
        st.info("No graph data available for visualization. The API may be unavailable or still initializing.")

def render_about_page():
    st.header("About News Analytics")
    
    st.markdown("""
    ### Overview
    
    This application provides tools to analyze news articles using modern NLP techniques:
    
    - **Semantic Search**: Find relevant articles based on meaning, not just keywords
    - **Topic Modeling**: Discover themes and topics across articles
    - **Knowledge Graph**: Explore connections between entities mentioned in the news
    
    ### Technology Stack
    
    - **Backend**: FastAPI on Railway with BERTopic for topic modeling
    - **Frontend**: Streamlit
    - **Vector Database**: Qdrant Cloud for semantic search
    - **Embeddings**: Sentence Transformers
    
    ### Data Sources
    
    News articles are scraped from CNN and processed using NLP pipelines.
    The data is updated every 12 hours to keep the content fresh.
    
    ### Fallback Mechanism
    
    This application includes a fallback to local data when the API is unavailable.
    This ensures you can still explore the interface and functionality even when
    the backend is initializing or offline.
    
    ### Code and Deployment
    
    The source code is available on GitHub and deployed using cloud platforms
    for seamless scaling and accessibility.
    """)
    
    # API status check
    st.subheader("API Status")
    
    try:
        # Simple check if API is responsive
        start_time = time.time()
        _ = st.session_state.api_client.get_recent_articles(limit=1)
        end_time = time.time()
        
        response_time = round((end_time - start_time) * 1000)
        
        st.success(f"API is online. Response time: {response_time}ms")
    except Exception as e:
        st.error(f"API is offline or experiencing issues: {str(e)}")
        
        # Show local data status
        if st.session_state.local_data["news_articles"] is not None:
            st.success(f"Using local data: {len(st.session_state.local_data['news_articles'])} articles available")
        else:
            st.warning("No local data available. Both API and local fallback are unavailable.")

if __name__ == "__main__":
    main()
