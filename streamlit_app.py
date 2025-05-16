# streamlit_app.py

import streamlit as st
import json
import os
import plotly.express as px
import pandas as pd

# Set page config
st.set_page_config(page_title="Federated Heart Risk Prediction", page_icon="🦁", layout="wide")

# Title
st.title("Federated Heart Attack Risk Predictor")
st.divider()

# Load saved accuracies
try:
    with open('results/accuracy.json', 'r') as f:
        accuracy_data = json.load(f)
        global_accuracy = (accuracy_data['global_accuracy']) * 100
        client_accuracies = accuracy_data['client_accuracies']
except FileNotFoundError:
    st.error("🚨 Accuracy file not found. Please train the model first by running `main.py`.")
    st.stop()

# Convert to DataFrame
df_clients = pd.DataFrame({
    'client': list(client_accuracies.keys()),
    'accuracy': list(client_accuracies.values())
})

# Project Overview
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Project Overview")
    st.write("""
    - Implements **Federated Learning** to predict heart attack risks.
    - Data is distributed across clients; each trains a local model.
    - Aggregated into a **global model** while preserving **data privacy**.
    """)

with col2:
    st.image('images/hospital_banner.png', caption="Secure Healthcare Data", use_column_width=True)

st.divider()

# Global Accuracy
st.subheader("🌍 Global Model Accuracy")
st.metric(label="Global Testing Accuracy", value=f"{global_accuracy:.2f}%")
st.success("✅ Model successfully aggregated from multiple clients!")

st.divider()

# Client-wise Accuracy - Bar Chart
st.subheader("📊 Real Client-wise Local Model Accuracy")

fig_bar = px.bar(
    x=df_clients['client'],
    y=df_clients['accuracy'],
    labels={'x': 'Clients', 'y': 'Accuracy (%)'},
    title="Real Client-wise Model Accuracy",
    text=df_clients['accuracy'],
    color=df_clients['client']
)
st.plotly_chart(fig_bar)

# Client-wise Line Plot
st.subheader("📈 Client Accuracy Trend")

fig_line = px.line(
    df_clients,
    x='client',
    y='accuracy',
    markers=True,
    title="Client-wise Accuracy Trend"
)
st.plotly_chart(fig_line)

# Histogram
st.subheader("📊 Histogram of Client Accuracies")

fig_hist = px.histogram(
    df_clients,
    x='accuracy',
    nbins=10,
    title="Distribution of Local Model Accuracies"
)
st.plotly_chart(fig_hist)

# Box Plot
st.subheader("📦 Accuracy Spread Across Clients")

fig_box = px.box(
    df_clients,
    y='accuracy',
    title="Box Plot of Client Accuracies"
)
st.plotly_chart(fig_box)

# Pie Chart of Accuracy Ranges
st.subheader("🎯 Accuracy Range Distribution")

def categorize_accuracy(acc):
    if acc < 45:
        return "< 45%"
    elif acc <= 50:
        return "45-50%"
    else:
        return "> 50%"

df_clients['range'] = df_clients['accuracy'].apply(categorize_accuracy)
range_counts = df_clients['range'].value_counts().reset_index()
range_counts.columns = ['range', 'count']

fig_pie = px.pie(
    range_counts,
    names='range',
    values='count',
    title="Client Accuracy Ranges"
)
st.plotly_chart(fig_pie)

# Top and Bottom Clients
st.subheader("🏆 Top & 🔻 Bottom Performing Clients")

sorted_clients = sorted(client_accuracies.items(), key=lambda x: x[1], reverse=True)

st.write("### 🏆 Top 3 Clients")
st.table(sorted_clients[:3])

st.write("### 🔻 Bottom 3 Clients")
st.table(sorted_clients[-3:])

# Footer
st.divider()
st.caption("Made by CSE-B TEAM ❤️ | Secure Federated AI in Healthcare")
