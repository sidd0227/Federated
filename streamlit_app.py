# streamlit_app.py

import streamlit as st
import json
import os
import plotly.express as px

# Set page config
st.set_page_config(page_title="Federated Heart Risk Prediction", page_icon="🦁", layout="wide")

# Title
st.title("Federated Heart Attack Risk Predictor")

st.divider()

# Load saved accuracies
try:
    with open('results/accuracy.json', 'r') as f:
        accuracy_data = json.load(f)
        global_accuracy = accuracy_data['global_accuracy'] * 100
        client_accuracies = accuracy_data['client_accuracies']
except FileNotFoundError:
    st.error("🚨 Accuracy file not found. Please train the model first by running `main.py`.")
    st.stop()
 
# Main Content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Project Overview")
    st.write("""
    - This project implements a **Federated Learning** approach to predict heart attack risks.
    - Data is split across multiple clients, models are trained locally, and then averaged to create a global model.
    - The approach ensures **data privacy** while still achieving **high accuracy**.
    """)

with col2:
    st.image('images/hospital_banner.png', caption="Secure Healthcare Data", use_column_width=True)

st.divider()

# Display Global Accuracy
st.subheader("🌎 Global Model Accuracy")
st.metric(label="Global Testing Accuracy", value=f"{global_accuracy:.2f}%")
st.success("✅ Model successfully aggregated from multiple clients!")

st.divider()

# Display Client-wise Accuracy
st.subheader("📊 Client-wise Local Model Accuracy (real)")

fig = px.bar(
    x=list(client_accuracies.keys()),
    y=list(client_accuracies.values()),
    labels={'x': 'Clients', 'y': 'Accuracy (%)'},
    title="Real Client-wise Model Accuracy",
    text=list(client_accuracies.values()),
    color=list(client_accuracies.keys())
)
st.plotly_chart(fig)

# Footer
st.caption("Made by CSE-B TEAM")
