import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns


model = pickle.load(open("crop_model.pkl", "rb"))
df = pd.read_csv("Crop_recommendation.csv") 

crop_images = {
    "rice": "images/rice.jpg",
    "maize": "images/maize.jpg",
    "chickpea": "images/chickpea.jpg",
    "kidneybeans": "images/kidneybeans.jpg",
    "pigeonpeas": "images/pigeonpeas.jpg",
    "mothbeans": "images/mothbeans.jpg",
    "mungbean": "images/mungbean.jpg",
    "blackgram": "images/blackgram.jpg",
    "lentil": "images/lentil.jpg",
    "pomegranate": "images/pomegranate.jpg",
    "banana": "images/banana.jpg",
    "mango": "images/mango.jpg",
    "grapes": "images/grapes.jpg",
    "watermelon": "images/watermelon.jpg",
    "muskmelon": "images/muskmelon.jpg",
    "apple": "images/apple.jpg",
    "orange": "images/orange.jpg",
    "papaya": "images/papaya.jpg",
    "cotton": "images/cotton.jpg",
    "jute": "images/jute.jpg",
    "coconut": "images/coconut.jpg",
    "coffee": "images/coffee.jpg"
}

crop_categories = {
    "All": list(crop_images.keys()),
    "Cereals": ["rice", "maize"],
    "Pulses": ["chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"],
    "Fruits": ["pomegranate", "banana", "mango", "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya"],
    "Cash Crops": ["cotton", "jute", "coconut", "coffee"]
}


st.sidebar.title("🌱 Crop Recommender")
page = st.sidebar.radio("Go to", ["Home", "Dataset Insights", "Crop Gallery", "About"])


if page == "Home":
    st.title("🌾 AI-Based Crop Recommendation System")
    st.markdown("### Helping farmers choose the best crop based on soil & weather data 🌦️")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N) [mg/kg]", 0, 200)
        P = st.number_input("Phosphorus (P) [mg/kg]", 0, 200)
        K = st.number_input("Potassium (K) [mg/kg]", 0, 200)
        budget = st.number_input("Budget (₹ per acre)", 1000, 50000, step=500)

    with col2:
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0)
        ph = st.number_input("pH Value", 0.0, 14.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0)

    with st.expander("ℹ️ About Parameters"):
        st.markdown("""
        - **Nitrogen (N):** Essential for leaf growth and chlorophyll production.  
        - **Phosphorus (P):** Encourages root development and flowering.  
        - **Potassium (K):** Improves plant strength and disease resistance.  
        - **Temperature (°C):** Affects metabolic rate and growth cycles.  
        - **Humidity (%):** Higher humidity reduces water loss in plants.  
        - **pH Value:** Determines soil acidity/alkalinity; affects nutrient availability.  
        - **Rainfall (mm):** Provides necessary water supply for crops.  
        - **Budget (₹/acre):** Estimated money farmer can spend for cultivation.  
        """)

    if st.button("🌿 Recommend Crop"):
        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        crop = prediction[0]
        st.success(f"✅ Recommended Crop: **{crop.capitalize()}**")

        if crop in crop_images and os.path.exists(crop_images[crop]):
            st.image(crop_images[crop], width=300, caption=f"Recommended: {crop.capitalize()}")
        else:
            st.warning("⚠️ No image available for this crop.")

        sustainability_score = round((100 - abs(ph - 7) * 5 + humidity / 2) % 100, 2)
        total_score = round((sustainability_score + (budget / 50000) * 100) / 2, 2)

        st.info(f"🌍 Sustainability Score: **{sustainability_score}/100**")
        st.info(f"📊 Total Expected Score: **{total_score}/100**")

        result_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall, budget, crop, sustainability_score, total_score]],
                                 columns=["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall", "Budget", "Recommended Crop", "Sustainability Score", "Total Score"])
        st.download_button("📥 Download Recommendation", 
                           result_df.to_csv(index=False), 
                           "recommendation.csv",
                           "text/csv")


elif page == "Dataset Insights":
    st.title("📊 Dataset Insights")
    st.write("This is the dataset used to train the model.")

    if st.checkbox("Show raw dataset"):
        st.dataframe(df.head(20))

    st.subheader("🌾 Crop Distribution")
    st.bar_chart(df['label'].value_counts())

    st.subheader("📈 Feature Distributions")
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    selected_feature = st.selectbox("Select a feature to view distribution:", features)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, bins=30, ax=ax, color="green")
    ax.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig)

    st.subheader("📦 Outlier Detection (Boxplot)")
    fig, ax = plt.subplots()
    sns.boxplot(data=df[features], ax=ax)
    ax.set_title("Boxplot of Features")
    st.pyplot(fig)

    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

    st.subheader("📑 Summary Statistics")
    st.write(df.describe())


elif page == "Crop Gallery":
    st.title("🌿 Crop Gallery")
    st.write("Browse images of all crops available in the dataset.")

    category = st.selectbox("📂 Select a crop category:", list(crop_categories.keys()))

    search = st.text_input("🔍 Search for a crop (e.g., rice, mango, cotton):").strip().lower()

    selected_crops = crop_categories[category]

    filtered_crops = {crop: crop_images[crop] for crop in selected_crops if search in crop.lower()} if search else {crop: crop_images[crop] for crop in selected_crops}

    if filtered_crops:
        cols = st.columns(3)  
        i = 0
        for crop, img_path in filtered_crops.items():
            if os.path.exists(img_path):
                with cols[i % 3]:
                    st.image(img_path, caption=crop.capitalize(), use_container_width=True)
            i += 1
    else:
        st.warning("⚠️ No crops found for your selection.")


else:
    st.title("ℹ️ About This Project")
    st.markdown("""
    This project is built for farmers to **recommend the most suitable crop** 
    based on soil nutrients (N, P, K), weather conditions (temperature, humidity, rainfall), 
    soil pH, and even budget.

    **Features:**
    - Crop Recommendation 🌾
    - Dataset Insights 📊
    - Crop Gallery 🌿
    - Sustainability & Score Metrics 🌍
    - Downloadable Reports 📥

    **Tech Stack:**
    - Python 🐍  
    - scikit-learn 🤖  
    - Streamlit 🌐  
    - Seaborn & Matplotlib 📊  

    Developed as part of a Hackathon to support sustainable agriculture 🌱.
    """)

