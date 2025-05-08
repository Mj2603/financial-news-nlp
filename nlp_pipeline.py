"""
NLP Pipeline for Financial News Analysis

This module implements a complete pipeline for collecting, processing, and analyzing
financial news sentiment using NLP techniques.
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

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class FinancialNewsPipeline:
    """
    A pipeline for collecting and analyzing financial news sentiment.
    """
    
    def __init__(self, api_key=None):
        """
        Initialize the pipeline.
        
        Args:
            api_key: NewsAPI key (optional)
        """
        self.api_key = api_key
        self.nlp = spacy.load('en_core_web_sm')
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize NewsAPI client if key is provided
        if api_key:
            self.newsapi = NewsApiClient(api_key=api_key)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def collect_news(self, query, days=7, sources=None):
        """
        Collect news articles from various sources.
        
        Args:
            query: Search query
            days: Number of days to look back
            sources: List of news sources (optional)
            
        Returns:
            DataFrame with news articles
        """
        try:
            if self.api_key:
                # Use NewsAPI
                from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                news = self.newsapi.get_everything(
                    q=query,
                    from_param=from_date,
                    language='en',
                    sort_by='relevancy'
                )
                articles = news['articles']
            else:
                # Fallback to web scraping
                articles = self._scrape_news(query)
            
            # Convert to DataFrame
            df = pd.DataFrame(articles)
            df['publishedAt'] = pd.to_datetime(df['publishedAt'])
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting news: {str(e)}")
            return pd.DataFrame()
    
    def _scrape_news(self, query):
        """
        Scrape news from financial websites.
        
        Args:
            query: Search query
            
        Returns:
            List of news articles
        """
        articles = []
        try:
            # Example: Scrape from Yahoo Finance
            url = f"https://finance.yahoo.com/news/{query.replace(' ', '-')}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract news items (adjust selectors based on website structure)
            news_items = soup.find_all('div', class_='news-item')
            
            for item in news_items:
                articles.append({
                    'title': item.find('h3').text,
                    'description': item.find('p').text,
                    'publishedAt': item.find('time')['datetime'],
                    'url': item.find('a')['href']
                })
                
        except Exception as e:
            self.logger.error(f"Error scraping news: {str(e)}")
        
        return articles
    
    def preprocess_text(self, text):
        """
        Preprocess text for analysis.
        
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
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
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
    
    def process_news(self, news_df):
        """
        Process news articles through the pipeline.
        
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
        
        return news_df
    
    def get_market_data(self, symbol, days=7):
        """
        Get market data for correlation analysis.
        
        Args:
            symbol: Stock symbol
            days: Number of days to look back
            
        Returns:
            DataFrame with market data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)
            
            return hist
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {str(e)}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    pipeline = FinancialNewsPipeline()
    
    # Collect news
    news_df = pipeline.collect_news("AAPL stock")
    
    # Process news
    processed_df = pipeline.process_news(news_df)
    
    # Get market data
    market_data = pipeline.get_market_data("AAPL")
    
    print("\nProcessed News Sample:")
    print(processed_df[['title', 'polarity', 'subjectivity']].head())
    
    print("\nMarket Data Sample:")
    print(market_data.head()) 