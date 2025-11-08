import torch
from transformers import pipeline


class TextSummarizer:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.device = 0 if torch.cuda.is_available() else -1
        self._load_model()
    
    def _load_model(self):
        self.summarizer = pipeline(
            "summarization",
            model=self.model_name,
            device=self.device
        )
    
    def summarize(self, text, max_length=150, min_length=30):
        if len(text) < 100:
            return {"error": "Text too short for summarization"}
        
        try:
            result = self.summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            
            summary = result[0]["summary_text"]
            
            return {
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(text)
            }
        
        except Exception as e:
            return {"error": str(e)}