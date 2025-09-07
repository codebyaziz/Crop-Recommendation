import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_FILE = "crop_model_new.pkl"
DATA_FILE = "Crop_recommendation.csv"

st.set_page_config(page_title="🌱 AI Crop Recommendation", layout="centered")
st.title("🌱 AI-Based Crop Recommendation System for Farmers")

if not os.path.exists(MODEL_FILE):
    st.info("Training model as it does not exist...")
    df = pd.read_csv(DATA_FILE)
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"Model trained and saved! Accuracy: {accuracy*100:.2f}%")
else:
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv(DATA_FILE)
    st.success("Model loaded successfully ✅")

st.sidebar.header("Input Parameters")
st.sidebar.write("Enter soil and weather conditions to get crop recommendation.")

def user_input_features():
    N = st.sidebar.slider("Nitrogen (N)", 0, 140, 90, help="Amount of nitrogen in soil")
    P = st.sidebar.slider("Phosphorus (P)", 0, 140, 42, help="Amount of phosphorus in soil")
    K = st.sidebar.slider("Potassium (K)", 0, 205, 43, help="Amount of potassium in soil")
    temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 20.0, help="Temperature in Celsius")
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 80.0, help="Humidity in percentage")
    ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5, help="Soil pH value")
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 200.0, help="Rainfall in mm")
    
    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    return data

input_data = user_input_features()

if st.button("Predict Crop"):
    prediction = model.predict(input_data)[0]
    st.success(f"Recommended Crop: **{prediction}**")
    st.balloons()

with st.expander("📊 Dataset & Model Info"):
    st.subheader("Dataset Snapshot")
    st.dataframe(df.head())
    
    st.subheader("Dataset Description")
    st.write(df.describe())
    
    if os.path.exists(MODEL_FILE):
        st.subheader("Model Details")
        st.write(f"Type: RandomForestClassifier")
        st.write(f"Number of trees: 100")
        X = df.drop('label', axis=1)
        y = df['label']
        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)
        st.write(f"Training accuracy: {acc*100:.2f}%")

with st.expander("📈 Dataset Visualizations"):
    st.subheader("N, P, K Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[['N','P','K']], kde=True, ax=ax, palette="viridis")
    ax.set_xlabel("N, P, K Values")
    st.pyplot(fig)
    
    st.subheader("Temperature, Humidity, pH, Rainfall Distribution")
    fig2, ax2 = plt.subplots(2, 2, figsize=(10,6))
    sns.histplot(df['temperature'], kde=True, ax=ax2[0,0], color="tomato")
    ax2[0,0].set_title("Temperature")
    sns.histplot(df['humidity'], kde=True, ax=ax2[0,1], color="skyblue")
    ax2[0,1].set_title("Humidity")
    sns.histplot(df['ph'], kde=True, ax=ax2[1,0], color="lightgreen")
    ax2[1,0].set_title("Soil pH")
    sns.histplot(df['rainfall'], kde=True, ax=ax2[1,1], color="orange")
    ax2[1,1].set_title("Rainfall (mm)")
    plt.tight_layout()
    st.pyplot(fig2)
    
    st.subheader("Most Common Crops")
    crop_counts = df['label'].value_counts()
    fig3, ax3 = plt.subplots(figsize=(6,6))
    ax3.pie(crop_counts, labels=crop_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors)
    ax3.axis('equal')
    st.pyplot(fig3)


