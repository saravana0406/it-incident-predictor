import streamlit as st
import requests

st.title("Smart IT Incident Predictor")

cpu = st.slider("CPU Usage", 0, 100, 50)
memory = st.slider("Memory Usage", 0, 100, 50)
errors = st.slider("Error Count", 0, 10, 1)

if st.button("Predict"):
    res = requests.get(
        "http://127.0.0.1:8000/predict",
        params={"cpu": cpu, "memory": memory, "errors": errors}
    )

    data = res.json()

    st.write("Prediction:", data["prediction"])
    st.write("Action:", data["action"])