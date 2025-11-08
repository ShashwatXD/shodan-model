import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, text):
        if isinstance(text, list):
            return self._predict_batch(text)
        else:
            return self._predict_single(text)
    
    def _predict_single(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        scores = predictions.cpu().numpy()[0]
        labels = ["negative", "neutral", "positive"]
        
        result = {}
        for i, label in enumerate(labels):
            result[label] = float(scores[i])
        
        predicted_label = labels[scores.argmax()]
        confidence = float(scores.max())
        
        return {
            "sentiment": predicted_label,
            "confidence": confidence,
            "scores": result
        }
    
    def _predict_batch(self, texts):
        results = []
        for text in texts:
            results.append(self._predict_single(text))
        return results