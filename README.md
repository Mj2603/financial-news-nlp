# Financial News NLP Pipeline

A comprehensive pipeline for collecting, analyzing, and visualizing sentiment from financial news headlines.

## Features

- News Collection
  - API-based collection (NewsAPI)
  - Web scraping fallback (Yahoo Finance)
  - Customizable date ranges and sources

- Text Processing
  - Tokenization and lemmatization
  - Stop word removal
  - Entity recognition

- Sentiment Analysis
  - Polarity scoring
  - Subjectivity analysis
  - Entity extraction

- Visualization
  - Interactive dashboard
  - Word clouds
  - Sentiment timelines
  - Distribution plots

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Mj2603/financial-news-nlp.git
cd financial-news-nlp
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

### Basic Usage

```python
from nlp_pipeline import FinancialNewsPipeline
from sentiment_visuals import SentimentVisualizer

# Initialize pipeline
pipeline = FinancialNewsPipeline(api_key='YOUR_NEWSAPI_KEY')  # Optional

# Collect and process news
news_df = pipeline.collect_news("AAPL stock")
processed_df = pipeline.process_news(news_df)

# Create visualizations
visualizer = SentimentVisualizer()
app = visualizer.create_dashboard(processed_df)
app.run_server(debug=True)
```

### Running the Dashboard

```bash
python sentiment_visuals.py
```

The dashboard will be available at http://127.0.0.1:8050/

## Project Structure

```
financial-news-nlp/
├── data/
│   └── news_sample.csv
├── nlp_pipeline.py
├── sentiment_visuals.py
├── requirements.txt
└── README.md
```

## Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- nltk >= 3.8.1
- spacy >= 3.7.2
- beautifulsoup4 >= 4.12.0
- requests >= 2.31.0
- newsapi-python >= 0.2.7
- yfinance >= 0.2.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- wordcloud >= 1.9.0
- textblob >= 0.17.1
- scikit-learn >= 1.0.0
- plotly >= 5.13.0
- dash >= 2.9.0

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 