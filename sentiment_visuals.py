"""
Financial News Sentiment Analysis Dashboard

A professional-grade dashboard for visualizing and analyzing financial news sentiment
and its impact on market movements. Designed for institutional investors and
quantitative analysts.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import logging
from typing import Dict, List, Optional, Union
import json
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_dashboard.log'),
        logging.StreamHandler()
    ]
)

class SentimentVisualizer:
    """
    A professional-grade visualization engine for financial news sentiment analysis.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the visualization engine.
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.app = dash.Dash(
            __name__,
            suppress_callback_exceptions=True,
            title="Financial News Sentiment Analysis"
        )
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'theme': 'plotly_white',
            'colors': {
                'primary': '#1f77b4',
                'secondary': '#ff7f0e',
                'background': '#ffffff',
                'text': '#2c3e50'
            },
            'layout': {
                'width': 1200,
                'height': 800
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return {**default_config, **json.load(f)}
        return default_config
    
    def create_wordcloud(self, 
                        text_series: pd.Series, 
                        title: str = "Key Terms") -> WordCloud:
        """
        Create a professional word cloud from financial text data.
        
        Args:
            text_series: Series of text data
            title: Title for the plot
            
        Returns:
            WordCloud object
        """
        try:
            # Combine all text
            text = ' '.join(text_series)
            
            # Generate word cloud with professional styling
            wordcloud = WordCloud(
                width=1200,
                height=600,
                background_color='white',
                max_words=150,
                contour_width=3,
                contour_color=self.config['colors']['primary'],
                colormap='viridis'
            ).generate(text)
            
            return wordcloud
            
        except Exception as e:
            self.logger.error(f"Error creating word cloud: {str(e)}")
            return None
    
    def plot_sentiment_timeline(self, 
                              df: pd.DataFrame, 
                              title: str = "Sentiment Analysis Timeline") -> go.Figure:
        """
        Create a professional timeline plot of sentiment metrics.
        
        Args:
            df: DataFrame with sentiment data
            title: Title for the plot
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Add polarity line
            fig.add_trace(go.Scatter(
                x=df['publishedAt'],
                y=df['polarity'],
                mode='lines+markers',
                name='Sentiment Polarity',
                line=dict(
                    color=self.config['colors']['primary'],
                    width=2
                ),
                marker=dict(
                    size=8,
                    symbol='circle'
                )
            ))
            
            # Add subjectivity line
            fig.add_trace(go.Scatter(
                x=df['publishedAt'],
                y=df['subjectivity'],
                mode='lines+markers',
                name='Subjectivity',
                line=dict(
                    color=self.config['colors']['secondary'],
                    width=2
                ),
                marker=dict(
                    size=8,
                    symbol='square'
                )
            ))
            
            # Update layout with professional styling
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    y=0.95,
                    font=dict(size=24)
                ),
                xaxis_title='Date',
                yaxis_title='Score',
                hovermode='x unified',
                template=self.config['theme'],
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(l=50, r=50, t=100, b=50)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment timeline: {str(e)}")
            return None
    
    def plot_sentiment_distribution(self, 
                                  df: pd.DataFrame, 
                                  title: str = "Sentiment Distribution") -> go.Figure:
        """
        Create a professional distribution plot of sentiment metrics.
        
        Args:
            df: DataFrame with sentiment data
            title: Title for the plot
            
        Returns:
            Plotly figure
        """
        try:
            fig = go.Figure()
            
            # Add polarity histogram
            fig.add_trace(go.Histogram(
                x=df['polarity'],
                name='Sentiment Polarity',
                opacity=0.7,
                marker_color=self.config['colors']['primary'],
                nbinsx=50
            ))
            
            # Add subjectivity histogram
            fig.add_trace(go.Histogram(
                x=df['subjectivity'],
                name='Subjectivity',
                opacity=0.7,
                marker_color=self.config['colors']['secondary'],
                nbinsx=50
            ))
            
            # Update layout with professional styling
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    y=0.95,
                    font=dict(size=24)
                ),
                xaxis_title='Score',
                yaxis_title='Frequency',
                barmode='overlay',
                template=self.config['theme'],
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ),
                margin=dict(l=50, r=50, t=100, b=50)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment distribution: {str(e)}")
            return None
    
    def create_dashboard(self, df: pd.DataFrame) -> dash.Dash:
        """
        Create a professional interactive dashboard for sentiment analysis.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Dash app
        """
        # Define the layout
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(
                    "Financial News Sentiment Analysis Dashboard",
                    style={
                        'textAlign': 'center',
                        'color': self.config['colors']['text'],
                        'marginBottom': '30px'
                    }
                )
            ]),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Date Range:"),
                    dcc.DatePickerRange(
                        id='date-range',
                        start_date=df['publishedAt'].min(),
                        end_date=df['publishedAt'].max(),
                        style={'marginBottom': '20px'}
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Analysis Type:"),
                    dcc.Dropdown(
                        id='analysis-type',
                        options=[
                            {'label': 'Sentiment Analysis', 'value': 'sentiment'},
                            {'label': 'Entity Analysis', 'value': 'entities'},
                            {'label': 'Market Impact', 'value': 'market'}
                        ],
                        value='sentiment',
                        style={'width': '200px'}
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px'})
            ], style={'marginBottom': '30px'}),
            
            # Main content
            html.Div([
                # Left column
                html.Div([
                    html.H2("Key Terms", style={'textAlign': 'center'}),
                    html.Img(id='word-cloud', style={'width': '100%'})
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                # Right column
                html.Div([
                    html.H2("Sentiment Timeline", style={'textAlign': 'center'}),
                    dcc.Graph(id='sentiment-timeline')
                ], style={'width': '70%', 'display': 'inline-block'})
            ], style={'marginBottom': '30px'}),
            
            # Bottom row
            html.Div([
                html.Div([
                    html.H2("Sentiment Distribution", style={'textAlign': 'center'}),
                    dcc.Graph(id='sentiment-distribution')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    html.H2("News Articles", style={'textAlign': 'center'}),
                    html.Div(id='news-table', style={'height': '400px', 'overflow': 'auto'})
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ], style={'padding': '20px'})
        
        # Define callbacks
        @self.app.callback(
            [Output('word-cloud', 'src'),
             Output('sentiment-timeline', 'figure'),
             Output('sentiment-distribution', 'figure'),
             Output('news-table', 'children')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('analysis-type', 'value')]
        )
        def update_dashboard(start_date, end_date, analysis_type):
            # Filter data by date range
            mask = (df['publishedAt'] >= start_date) & (df['publishedAt'] <= end_date)
            filtered_df = df[mask]
            
            # Create word cloud
            wordcloud = self.create_wordcloud(filtered_df['processed_text'])
            wordcloud_img = wordcloud.to_image()
            
            # Create plots
            timeline_fig = self.plot_sentiment_timeline(filtered_df)
            dist_fig = self.plot_sentiment_distribution(filtered_df)
            
            # Create news table with professional styling
            table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Date", style={'padding': '10px'}),
                    html.Th("Title", style={'padding': '10px'}),
                    html.Th("Sentiment", style={'padding': '10px'}),
                    html.Th("Subjectivity", style={'padding': '10px'})
                ], style={'backgroundColor': self.config['colors']['primary'],
                         'color': 'white'})),
                html.Tbody([
                    html.Tr([
                        html.Td(row['publishedAt'].strftime('%Y-%m-%d %H:%M'),
                               style={'padding': '10px'}),
                        html.Td(row['title'], style={'padding': '10px'}),
                        html.Td(f"{row['polarity']:.2f}", style={'padding': '10px'}),
                        html.Td(f"{row['subjectivity']:.2f}", style={'padding': '10px'})
                    ], style={'backgroundColor': 'white' if i % 2 == 0 else '#f8f9fa'})
                    for i, (_, row) in enumerate(filtered_df.iterrows())
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse'})
            
            return wordcloud_img, timeline_fig, dist_fig, table
        
        return self.app

if __name__ == "__main__":
    # Example usage
    from nlp_pipeline import FinancialNewsPipeline
    
    # Initialize pipeline and get data
    pipeline = FinancialNewsPipeline()
    news_df = pipeline.collect_news("AAPL", days=30)
    processed_df = pipeline.process_news(news_df)
    
    # Create visualizer and dashboard
    visualizer = SentimentVisualizer()
    app = visualizer.create_dashboard(processed_df)
    app.run_server(debug=False, port=8050) 