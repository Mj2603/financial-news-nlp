FROM python:3.9-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3
RUN pip install --no-cache-dir nltk==3.8.1 && python -m nltk.downloader punkt vader_lexicon stopwords
RUN pip install --no-cache-dir \
    cymem==2.0.6 \
    preshed==3.0.6 \
    murmurhash==1.0.7 \
    thinc==8.1.10 \
    blis==0.7.9 \
    spacy==3.5.3
RUN pip install --no-cache-dir \
    beautifulsoup4==4.12.2 \
    requests==2.31.0 \
    newsapi-python==0.2.7 \
    yfinance==0.2.31
RUN pip install --no-cache-dir \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    wordcloud==1.9.2
RUN pip install --no-cache-dir \
    textblob==0.17.1 \
    scikit-learn==1.3.0 \
    dash==2.14.2 \
    dash-bootstrap-components==1.5.0 \
    plotly==5.18.0
RUN python -m spacy download en_core_web_sm
RUN mkdir -p /app/data/raw /app/data/processed /app/models
COPY . .
EXPOSE 8050
CMD ["python", "src/sentiment_visuals.py"]
