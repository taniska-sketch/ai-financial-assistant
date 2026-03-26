import joblib
import re
import os

# Get current file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go to models folder
MODEL_PATH = os.path.join(BASE_DIR, "..", "models")

# Load models
cat_model = joblib.load(os.path.join(MODEL_PATH, "categorization_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_PATH, "vectorizer.pkl"))
cash_model = joblib.load(os.path.join(MODEL_PATH, "cashflow_model.pkl"))
anomaly_stats = joblib.load(os.path.join(MODEL_PATH, "anomaly_stats.pkl"))

mean = anomaly_stats['mean']
std = anomaly_stats['std']


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text


def predict_category(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    return cat_model.predict(vec)[0]


def predict_cashflow(month, day, weekday):
    return float(cash_model.predict([[month, day, weekday]])[0])


def detect_anomaly(amount):
    z = (amount - mean) / std
    return 1 if abs(z) > 3 else 0