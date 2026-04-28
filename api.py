from fastapi import FastAPI
import numpy as np
import pickle
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("../models/model.h5")
scaler = pickle.load(open("../models/scaler.pkl", "rb"))
pca = pickle.load(open("../models/pca.pkl", "rb"))

def decide_action(pred):
    if pred == 0:
        return "✅ Normal"
    elif pred == 1:
        return "⚠️ Warning: Monitor system"
    else:
        return "🚨 Critical: Restart service"

@app.get("/predict")
def predict(cpu: float, memory: float, errors: float):
    X = [[cpu, memory, errors]]

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    pred = np.argmax(model.predict(X_pca), axis=1)[0]

    return {
        "prediction": int(pred),
        "action": decide_action(pred)
    }