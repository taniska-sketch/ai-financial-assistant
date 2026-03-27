from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os
import re

# Initialize FastAPI app
app = FastAPI(
    title="AI Financial Assistant",
    description="ML-powered API for Categorization, Cashflow Prediction & Anomaly Detection",
    version="1.0"
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models")

# Load models safely
try:
    cat_model = joblib.load(os.path.join(MODEL_PATH, "categorization_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "vectorizer.pkl"))
    cash_model = joblib.load(os.path.join(MODEL_PATH, "cashflow_model.pkl"))
    anomaly_stats = joblib.load(os.path.join(MODEL_PATH, "anomaly_stats.pkl"))

    mean = anomaly_stats['mean']
    std = anomaly_stats['std']

except Exception as e:
    print("Error loading models:", e)
    cat_model = None
    vectorizer = None
    cash_model = None
    mean = 0
    std = 1


# -------------------------
# Utility Function
# -------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


# -------------------------
# Request Schemas
# -------------------------

class CategoryRequest(BaseModel):
    description: str


class CashflowRequest(BaseModel):
    month: int
    day: int
    weekday: int


class AnomalyRequest(BaseModel):
    amount: float


# -------------------------
# Routes
# -------------------------

@app.get("/")
def home():
    return {"message": "API running 🚀"}


@app.post("/predict-category")
def predict_category(req: CategoryRequest):
    if cat_model is None:
        return {"error": "Model not loaded"}

    text = clean_text(req.description)
    vec = vectorizer.transform([text])
    result = cat_model.predict(vec)[0]

    return {"category": result}


@app.post("/predict-cashflow")
def predict_cashflow(req: CashflowRequest):
    if cash_model is None:
        return {"error": "Model not loaded"}

    prediction = cash_model.predict([[req.month, req.day, req.weekday]])[0]

    return {"predicted_balance": float(prediction)}


@app.post("/detect-anomaly")
def detect_anomaly(req: AnomalyRequest):
    z = (req.amount - mean) / std
    result = 1 if abs(z) > 3 else 0

    return {"anomaly": result}
# force update