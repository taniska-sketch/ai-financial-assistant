from fastapi import FastAPI
from pydantic import BaseModel

# IMPORTANT: absolute import for Render
from backend.utils import predict_category, predict_cashflow, detect_anomaly


# Force enable docs properly
app = FastAPI(
    title="AI Financial Assistant API",
    description="ML-powered financial assistant for categorization, prediction, and anomaly detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


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
def category(req: CategoryRequest):
    result = predict_category(req.description)
    return {"category": result}


@app.post("/predict-cashflow")
def cashflow(req: CashflowRequest):
    result = predict_cashflow(req.month, req.day, req.weekday)
    return {"predicted_balance": result}


@app.post("/detect-anomaly")
def anomaly(req: AnomalyRequest):
    result = detect_anomaly(req.amount)
    return {"anomaly": result}