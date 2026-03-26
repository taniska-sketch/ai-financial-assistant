import joblib
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def predict_category(text):
    cat_model = joblib.load(os.path.join(MODEL_PATH, "categorization_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "vectorizer.pkl"))

    text = clean_text(text)
    vec = vectorizer.transform([text])
    return cat_model.predict(vec)[0]


def predict_cashflow(month, day, weekday):
    cash_model = joblib.load(os.path.join(MODEL_PATH, "cashflow_model.pkl"))
    return float(cash_model.predict([[month, day, weekday]])[0])


def detect_anomaly(amount):
    anomaly_stats = joblib.load(os.path.join(MODEL_PATH, "anomaly_stats.pkl"))

    mean = anomaly_stats['mean']
    std = anomaly_stats['std']

    z = (amount - mean) / std
    return 1 if abs(z) > 3 else 0