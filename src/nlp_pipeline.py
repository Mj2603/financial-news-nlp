"""
Financial News Sentiment Analysis Framework

A sophisticated framework for quantitative analysis of financial news sentiment
and its impact on market movements. Designed for institutional investors and
quantitative analysts.
"""

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
from newsapi import NewsApiClient
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, List, Optional, Union
import json
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_news.log'),
        logging.StreamHandler()
    ]
)

class FinancialNewsPipeline:
    """
    A sophisticated pipeline for quantitative analysis of financial news sentiment
    and its impact on market movements.
    """
    
    def __init__(self, api_key: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the pipeline with configuration and API credentials.
        
        Args:
            api_key: NewsAPI key for data collection
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize NLP components
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize NewsAPI client
        if self.api_key:
            self.newsapi = NewsApiClient(api_key=self.api_key)
        
        # Create data directories
        self._setup_directories()
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'data_dir': 'data',
            'cache_dir': 'cache',
            'max_retries': 3,
            'timeout': 30,
            'batch_size': 100
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def _setup_directories(self):
        """Create necessary directories for data storage."""
        for dir_name in ['raw', 'processed', 'market']:
            Path(self.config['data_dir']).joinpath(dir_name).mkdir(parents=True, exist_ok=True)
    
    def collect_news(self, 
                    query: str, 
                    days: int = 7, 
                    sources: Optional[List[str]] = None,
                    save_raw: bool = True) -> pd.DataFrame:
        """
        Collect financial news articles from various sources.
        
        Args:
            query: Search query for financial instruments or companies
            days: Number of days to look back
            sources: List of news sources to include
            save_raw: Whether to save raw data to disk
            
        Returns:
            DataFrame with collected news articles
        """
        try:
            if self.api_key:
                news_data = self._collect_from_api(query, days, sources)
            else:
                news_data = self._scrape_news(query)
            
            df = pd.DataFrame(news_data)
            if not df.empty:
                df['publishedAt'] = pd.to_datetime(df['publishedAt'])
                
                if save_raw:
                    self._save_raw_data(df, query)
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting news: {str(e)}")
            return pd.DataFrame()
    
    def _collect_from_api(self, 
                         query: str, 
                         days: int, 
                         sources: Optional[List[str]]) -> List[Dict]:
        """Collect news from NewsAPI."""
        from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.newsapi.get_everything(
            q=query,
            from_param=from_date,
            language='en',
            sort_by='relevancy',
            sources=','.join(sources) if sources else None
        )['articles']
    
    def _scrape_news(self, query: str) -> List[Dict]:
        """
        Scrape financial news from alternative sources.
        
        Args:
            query: Search query
            
        Returns:
            List of news articles
        """
        articles = []
        try:
            # Implement scraping logic for financial news sources
            # This is a placeholder for demonstration
            pass
                
        except Exception as e:
            self.logger.error(f"Error scraping news: {str(e)}")
        
        return articles
    
    def _save_raw_data(self, df: pd.DataFrame, query: str):
        """Save raw news data to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{query.replace(' ', '_')}_{timestamp}.csv"
        df.to_csv(Path(self.config['data_dir']).joinpath('raw', filename), index=False)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess financial text for analysis.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
        
        return ' '.join(tokens)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[float, List[str]]]:
        """
        Perform sophisticated sentiment analysis on financial text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment metrics
        """
        # TextBlob sentiment
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # spaCy analysis
        doc = self.nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ['ORG', 'MONEY', 'PERCENT']]
        
        return {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'entities': entities
        }
    
    def process_news(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process news articles through the quantitative analysis pipeline.
        
        Args:
            news_df: DataFrame with news articles
            
        Returns:
            Processed DataFrame with sentiment analysis
        """
        if news_df.empty:
            return pd.DataFrame()
        
        # Preprocess text
        news_df['processed_text'] = news_df['title'] + ' ' + news_df['description']
        news_df['processed_text'] = news_df['processed_text'].apply(self.preprocess_text)
        
        # Analyze sentiment
        sentiment_scores = news_df['processed_text'].apply(self.analyze_sentiment)
        news_df['polarity'] = sentiment_scores.apply(lambda x: x['polarity'])
        news_df['subjectivity'] = sentiment_scores.apply(lambda x: x['subjectivity'])
        news_df['entities'] = sentiment_scores.apply(lambda x: x['entities'])
        
        # Save processed data
        self._save_processed_data(news_df)
        
        return news_df
    
    def _save_processed_data(self, df: pd.DataFrame):
        """Save processed news data to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"processed_{timestamp}.csv"
        df.to_csv(Path(self.config['data_dir']).joinpath('processed', filename), index=False)
    
    def get_market_data(self, 
                       symbol: str, 
                       days: int = 7,
                       interval: str = '1d') -> pd.DataFrame:
        """
        Retrieve market data for correlation analysis.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with market data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date, interval=interval)
            
            # Save market data
            self._save_market_data(hist, symbol)
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()
    
    def _save_market_data(self, df: pd.DataFrame, symbol: str):
        """Save market data to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol}_{timestamp}.csv"
        df.to_csv(Path(self.config['data_dir']).joinpath('market', filename), index=True)

if __name__ == "__main__":
    # Example usage
    pipeline = FinancialNewsPipeline()
    
    # Collect and analyze news
    news_df = pipeline.collect_news("AAPL", days=30)
    processed_df = pipeline.process_news(news_df)
    
    # Get market data
    market_data = pipeline.get_market_data("AAPL", days=30)
    
    print("\nProcessed News Sample:")
    print(processed_df[['title', 'polarity', 'subjectivity']].head())
    
    print("\nMarket Data Sample:")
    print(market_data.head()) 