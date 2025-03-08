import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
import logging
import os
import json
import re
from typing import List, Dict, Any, Tuple
import asyncio
import aiohttp
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from app.config import get_settings
from app.services.embedding import generate_embeddings
from app.services.qdrant import create_collection, upload_to_qdrant
from app.services.topic_model import build_topic_model
from app.services.knowledge_graph import extract_entities

logger = logging.getLogger(__name__)
settings = get_settings()

# Download NLTK resources
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {e}")

# Initialize the lemmatizer and stop words
try:
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
except Exception as e:
    logger.warning(f"Error initializing NLTK components: {e}")
    stop_words = set()
    lemmatizer = None

# Function to clean text
def clean_text(text):
    """Clean the input text by removing URLs, HTML tags, punctuation, etc."""
    if pd.isna(text):
        return ""

    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove digits
    text = re.sub(r'\d+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Function for advanced text preprocessing
def preprocess_text(text):
    """Tokenize, remove stopwords, and lemmatize text"""
    if not text:
        return ""

    # Tokenize
    try:
        tokens = nltk.word_tokenize(text)

        # Remove stopwords and short tokens
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

        # Lemmatize
        if lemmatizer:
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Join tokens back into a string
        return " ".join(tokens)
    except Exception as e:
        logger.warning(f"Error preprocessing text: {e}")
        return text

async def scrape_cnn(url: str, category: str, min_articles: int = 10) -> List[Dict[str, Any]]:
    """Scrape news articles from CNN"""
    articles = []
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status != 200:
                    logger.error(f"Error fetching CNN {category}: {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')

                # Find article links and headlines
                headlines = soup.find_all("span", class_="container__headline-text")
                count = 0

                for headline in headlines:
                    if count >= min_articles:
                        break

                    title = headline.text.strip()
                    if title and len(title) > 10:  # Filter out short titles
                        # Find parent link element to get the URL
                        link_element = headline.find_parent("a")
                        article_url = ""
                        if link_element and 'href' in link_element.attrs:
                            article_url = link_element['href']
                            if not article_url.startswith("http"):
                                article_url = "https://www.cnn.com" + article_url

                        # Get article content if URL is available
                        description = ""
                        if article_url:
                            try:
                                async with session.get(article_url, headers=headers) as article_response:
                                    if article_response.status == 200:
                                        article_html = await article_response.text()
                                        article_soup = BeautifulSoup(article_html, 'html.parser')

                                        # Extract article paragraphs
                                        paragraphs = article_soup.find_all("div", class_="paragraph")
                                        if paragraphs:
                                            description = " ".join([p.text.strip() for p in paragraphs[:10]])

                                        if not description:
                                            # Try alternative selectors
                                            paragraphs = article_soup.find_all("p", class_="paragraph")
                                            if paragraphs:
                                                description = " ".join([p.text.strip() for p in paragraphs[:10]])
                                            
                                        if not description:
                                            # Try one more selector
                                            paragraphs = article_soup.select("div.article__content p")
                                            if paragraphs:
                                                description = " ".join([p.text.strip() for p in paragraphs[:10]])
                                
                                await asyncio.sleep(0.5)  # Be polite to the server
                            except Exception as e:
                                logger.error(f"Error fetching article content: {e}")

                        if not description:
                            # Use the title as description if content couldn't be retrieved
                            description = title

                        # Create article record
                        article = {
                            "title": title,
                            "description": description,
                            "url": article_url,
                            "source": "CNN",
                            "category": category,
                            "scrape_date": datetime.now().strftime("%Y-%m-%d")
                        }

                        articles.append(article)
                        count += 1

        return articles
    except Exception as e:
        logger.error(f"Error scraping CNN {category}: {e}")
        return []

async def scrape_news(min_articles_per_source: int = 30) -> pd.DataFrame:
    """Scrape news articles from CNN"""
    all_articles = []

    # Define CNN categories
    categories = ["politics", "business", "health", "entertainment", "sports", "tech", "world"]
    
    for category in categories:
        url = f"https://www.cnn.com/{category}"
        try:
            logger.info(f"Scraping CNN {category}...")
            articles = await scrape_cnn(
                url=url,
                category=category,
                min_articles=min_articles_per_source // len(categories)
            )
            
            all_articles.extend(articles)
            logger.info(f"Scraped {len(articles)} articles from CNN {category}")
            await asyncio.sleep(2)  # Be polite to the server
        except Exception as e:
            logger.error(f"Error scraping CNN {category}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(all_articles)

    # Handle empty DataFrame
    if len(df) == 0:
        logger.warning("No articles were scraped")
        return pd.DataFrame(columns=[
            "title", "description", "url", "source", "category", "scrape_date",
            "clean_title", "clean_description", "text"
        ])

    # Remove duplicates
    df = df.drop_duplicates(subset=["title"])

    # Clean text fields
    df['clean_title'] = df['title'].apply(clean_text)
    df['clean_description'] = df['description'].apply(clean_text)
    
    # Create combined text field for searching
    df['text'] = df['clean_title'] + " " + df['clean_description']

    logger.info(f"Total unique articles scraped: {len(df)}")
    return df

async def scrape_and_process_news():
    """Scrape news, generate embeddings, and upload to Qdrant"""
    try:
        # Step 1: Scrape news
        logger.info("Starting news scraping...")
        df = await scrape_news(min_articles_per_source=settings.MIN_ARTICLES_PER_SOURCE)
        
        if len(df) == 0:
            logger.warning("No articles to process")
            return
        
        # Step 2: Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = await generate_embeddings(df['text'].tolist())
        
        # Step 3: Ensure collection exists
        await create_collection(vector_size=len(embeddings[0]))
        
        # Step 4: Upload to Qdrant
        await upload_to_qdrant(df, embeddings)
        
        # Step 5: Build topic model
        logger.info("Building topic model...")
        topic_model, topics, probs = await build_topic_model(df['text'].tolist())
        
        # Add topic information to the dataframe
        df['topic'] = topics
        df['topic_prob'] = [float(p) for p in probs]
        
        # Save topic model data
        topic_info = topic_model.get_topic_info()
        topic_keywords = {
            int(row['Topic']): [{"word": word, "score": float(score)} for word, score in topic_model.get_topic(row['Topic'])]
            for idx, row in topic_info.iterrows() if row['Topic'] != -1
        }
        
        # Step 6: Extract entities for knowledge graph
        logger.info("Extracting entities for knowledge graph...")
        kg_data = await extract_entities(df['text'].tolist(), limit=500)
        
        # Save processed data
        logger.info("Saving processed data...")
        os.makedirs(settings.DATA_DIR, exist_ok=True)
        
        # Save the dataframe
        df.to_csv(f"{settings.DATA_DIR}/news_articles.csv", index=False)
        
        # Save topic info
        with open(f"{settings.DATA_DIR}/topic_info.json", "w") as f:
            json.dump(topic_info.to_dict(orient="records"), f)
        
        # Save topic keywords
        with open(f"{settings.DATA_DIR}/topic_keywords.json", "w") as f:
            json.dump(topic_keywords, f)
        
        # Save knowledge graph data
        with open(f"{settings.DATA_DIR}/knowledge_graph.json", "w") as f:
            json.dump(kg_data, f)
        
        logger.info("News scraping and processing complete")
        return df
    except Exception as e:
        logger.error(f"Error in scrape_and_process_news: {e}")
        raise
