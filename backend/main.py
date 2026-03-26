from fastapi import FastAPI
from pydantic import BaseModel
from backend.utils import predict_category, predict_cashflow, detect_anomaly

app = FastAPI()   # ⭐ THIS IS IMPORTANT


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
def category(req: CategoryRequest):
    return {"category": predict_category(req.description)}


@app.post("/predict-cashflow")
def cashflow(req: CashflowRequest):
    return {"predicted_balance": predict_cashflow(req.month, req.day, req.weekday)}


@app.post("/detect-anomaly")
def anomaly(req: AnomalyRequest):
    return {"anomaly": detect_anomaly(req.amount)}