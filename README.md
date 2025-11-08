# Shodan AI FastAPI

Simple FastAPI app for AI sentiment analysis and text summarization.

## Files
- `app.py` - Main FastAPI application
- `sentiment.py` - Sentiment analysis model
- `summarizer.py` - Text summarization model  

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
python app.py
```

## API Endpoints

### POST /predict
Analyze sentiment of text:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this!"}'
```

### POST /summarize  
Summarize text:
```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Long text here...", "max_length": 100}'
```

## View Docs
Visit http://localhost:8000/docs for interactive API documentation.