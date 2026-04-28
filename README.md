# Smart IT Incident Predictor & Auto-Responder

## Overview
This project predicts IT system incidents using PCA + ANN and automatically suggests actions.

## Features
- Real-time prediction
- PCA-based feature reduction
- ANN-based classification
- Auto-response system
- Streamlit dashboard

## Tech Stack
- Python
- FastAPI
- TensorFlow
- Streamlit

## How to Run

### 1. Train Model
cd models
python train_model.py

### 2. Start API
cd backend
uvicorn api:app --reload

### 3. Run UI
cd frontend
streamlit run app.py

### 4. Run Simulator
cd utils
python simulator.py