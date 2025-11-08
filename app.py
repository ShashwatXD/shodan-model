from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, List
from sentiment import SentimentAnalyzer
from summarizer import TextSummarizer

app = FastAPI(title="Shodan AI API", version="1.0.0")

# Global model instances
sentiment_analyzer = None
text_summarizer = None


class SentimentRequest(BaseModel):
    text: Union[str, List[str]]


class SummaryRequest(BaseModel):
    text: str
    max_length: int = 150
    min_length: int = 30


@app.on_event("startup")
async def startup():
    global sentiment_analyzer, text_summarizer
    
    print("Loading models...")
    sentiment_analyzer = SentimentAnalyzer()
    text_summarizer = TextSummarizer()
    print("Models loaded successfully!")


@app.get("/")
async def root():
    return {
        "message": "Shodan AI API",
        "endpoints": ["/predict", "/summarize"],
        "status": "running"
    }


@app.post("/predict")
async def predict_sentiment(request: SentimentRequest):
    try:
        result = sentiment_analyzer.predict(request.text)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/summarize")
async def summarize_text(request: SummaryRequest):
    try:
        result = text_summarizer.summarize(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)