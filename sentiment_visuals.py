"""
Sentiment Visualization Module

This module provides visualization tools for financial news sentiment analysis.
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
from dash.dependencies import Input, Output
import logging

class SentimentVisualizer:
    """
    A class for visualizing financial news sentiment analysis results.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        self.logger = logging.getLogger(__name__)
    
    def create_wordcloud(self, text_series, title="Word Cloud"):
        """
        Create a word cloud from text data.
        
        Args:
            text_series: Series of text data
            title: Title for the plot
            
        Returns:
            WordCloud object
        """
        try:
            # Combine all text
            text = ' '.join(text_series)
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                contour_width=3,
                contour_color='steelblue'
            ).generate(text)
            
            return wordcloud
            
        except Exception as e:
            self.logger.error(f"Error creating word cloud: {str(e)}")
            return None
    
    def plot_sentiment_timeline(self, df, title="Sentiment Timeline"):
        """
        Create a timeline plot of sentiment scores.
        
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
                name='Polarity',
                line=dict(color='blue')
            ))
            
            # Add subjectivity line
            fig.add_trace(go.Scatter(
                x=df['publishedAt'],
                y=df['subjectivity'],
                mode='lines+markers',
                name='Subjectivity',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Score',
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment timeline: {str(e)}")
            return None
    
    def plot_sentiment_distribution(self, df, title="Sentiment Distribution"):
        """
        Create a distribution plot of sentiment scores.
        
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
                name='Polarity',
                opacity=0.7,
                marker_color='blue'
            ))
            
            # Add subjectivity histogram
            fig.add_trace(go.Histogram(
                x=df['subjectivity'],
                name='Subjectivity',
                opacity=0.7,
                marker_color='red'
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Score',
                yaxis_title='Count',
                barmode='overlay'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating sentiment distribution: {str(e)}")
            return None
    
    def create_dashboard(self, df):
        """
        Create an interactive dashboard for sentiment analysis.
        
        Args:
            df: DataFrame with sentiment data
            
        Returns:
            Dash app
        """
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("Financial News Sentiment Analysis Dashboard"),
            
            # Controls
            html.Div([
                html.Label("Date Range:"),
                dcc.DatePickerRange(
                    id='date-range',
                    start_date=df['publishedAt'].min(),
                    end_date=df['publishedAt'].max()
                )
            ]),
            
            # Word Cloud
            html.Div([
                html.H2("Word Cloud"),
                html.Img(id='word-cloud')
            ]),
            
            # Sentiment Timeline
            html.Div([
                html.H2("Sentiment Timeline"),
                dcc.Graph(id='sentiment-timeline')
            ]),
            
            # Sentiment Distribution
            html.Div([
                html.H2("Sentiment Distribution"),
                dcc.Graph(id='sentiment-distribution')
            ]),
            
            # News Table
            html.Div([
                html.H2("News Articles"),
                html.Div(id='news-table')
            ])
        ])
        
        @app.callback(
            [Output('word-cloud', 'src'),
             Output('sentiment-timeline', 'figure'),
             Output('sentiment-distribution', 'figure'),
             Output('news-table', 'children')],
            [Input('date-range', 'start_date'),
             Input('date-range', 'end_date')]
        )
        def update_dashboard(start_date, end_date):
            # Filter data by date range
            mask = (df['publishedAt'] >= start_date) & (df['publishedAt'] <= end_date)
            filtered_df = df[mask]
            
            # Create word cloud
            wordcloud = self.create_wordcloud(filtered_df['processed_text'])
            wordcloud_img = wordcloud.to_image()
            
            # Create plots
            timeline_fig = self.plot_sentiment_timeline(filtered_df)
            dist_fig = self.plot_sentiment_distribution(filtered_df)
            
            # Create news table
            table = html.Table([
                html.Thead(html.Tr([
                    html.Th("Date"),
                    html.Th("Title"),
                    html.Th("Polarity"),
                    html.Th("Subjectivity")
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(row['publishedAt'].strftime('%Y-%m-%d')),
                        html.Td(row['title']),
                        html.Td(f"{row['polarity']:.2f}"),
                        html.Td(f"{row['subjectivity']:.2f}")
                    ]) for _, row in filtered_df.iterrows()
                ])
            ])
            
            return wordcloud_img, timeline_fig, dist_fig, table
        
        return app

if __name__ == "__main__":
    # Example usage
    from nlp_pipeline import FinancialNewsPipeline
    
    # Initialize pipeline and get data
    pipeline = FinancialNewsPipeline()
    news_df = pipeline.collect_news("AAPL stock")
    processed_df = pipeline.process_news(news_df)
    
    # Create visualizer and dashboard
    visualizer = SentimentVisualizer()
    app = visualizer.create_dashboard(processed_df)
    app.run_server(debug=True) 