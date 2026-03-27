from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import re

app = FastAPI(
    title="AI Financial Assistant",
    version="1.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")

try:
    cat_model = joblib.load(os.path.join(MODEL_PATH, "categorization_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "vectorizer.pkl"))
    cash_model = joblib.load(os.path.join(MODEL_PATH, "cashflow_model.pkl"))
    anomaly_stats = joblib.load(os.path.join(MODEL_PATH, "anomaly_stats.pkl"))
except Exception as e:
    print(f"❌ Error loading models: {e}")
    cat_model = None
    vectorizer = None
    cash_model = None
    anomaly_stats = None

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

class CategoryRequest(BaseModel):
    description: str

class CashflowRequest(BaseModel):
    month: int
    day: int
    weekday: int

class AnomalyRequest(BaseModel):
    amount: float

@app.get("/")
def home():
    return {"message": "API running 🚀"}

@app.post("/predict-category")
def predict_category(req: CategoryRequest):
    try:
        if not cat_model or not vectorizer:
            return {"error": "Model not loaded"}

        text = clean_text(req.description)
        vec = vectorizer.transform([text])
        prediction = cat_model.predict(vec)[0]

        return {"category": prediction}
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict-cashflow")
def predict_cashflow(req: CashflowRequest):
    try:
        if not cash_model:
            return {"error": "Model not loaded"}

        prediction = cash_model.predict([[req.month, req.day, req.weekday]])[0]
        return {"predicted_balance": float(prediction)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/detect-anomaly")
def detect_anomaly(req: AnomalyRequest):
    try:
        if not anomaly_stats:
            return {"error": "Model not loaded"}

        mean = anomaly_stats['mean']
        std = anomaly_stats['std']

        z = (req.amount - mean) / std
        return {"anomaly": 1 if abs(z) > 3 else 0}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)