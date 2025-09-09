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


crop_economics = {
    "rice": (15000, 40000, 6),
    "maize": (12000, 35000, 7),
    "chickpea": (10000, 30000, 8),
    "kidneybeans": (13000, 33000, 7),
    "pigeonpeas": (11000, 29000, 8),
    "mothbeans": (9000, 25000, 9),
    "mungbean": (9500, 26000, 9),
    "blackgram": (10000, 27000, 8),
    "lentil": (10500, 28000, 8),
    "pomegranate": (40000, 100000, 7),
    "banana": (35000, 95000, 6),
    "mango": (50000, 120000, 7),
    "grapes": (45000, 110000, 7),
    "watermelon": (20000, 60000, 8),
    "muskmelon": (22000, 65000, 8),
    "apple": (60000, 140000, 6),
    "orange": (55000, 125000, 7),
    "papaya": (25000, 70000, 9),
    "cotton": (30000, 80000, 5),
    "jute": (28000, 75000, 6),
    "coconut": (35000, 95000, 8),
    "coffee": (50000, 130000, 6),
}


st.sidebar.title("🌱 Crop Recommender")
page = st.sidebar.radio("Go to", ["Home", "Dataset Insights", "Crop Gallery", "About"])


if page == "Home":
    st.title("🌾 AI-Based Crop Recommendation System")
    st.markdown("### Helping farmers choose the best crop based on soil & weather data 🌦️ + economics 💰")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N) [mg/kg]", 0, 200)
        P = st.number_input("Phosphorus (P) [mg/kg]", 0, 200)
        K = st.number_input("Potassium (K) [mg/kg]", 0, 200)
        budget = st.number_input("💵 Your Budget (₹/acre)", 5000, 200000, 30000)

    with col2:
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0)
        ph = st.number_input("pH Value", 0.0, 14.0)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0)

    if st.button("🌿 Recommend Crops"):
        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        main_crop = prediction[0]

        results = []
        for crop, (cost, profit, sustain) in crop_economics.items():
            if cost <= budget:
                roi = profit - cost
                total_score = (roi / 1000) * 0.6 + sustain * 10 * 0.4
                results.append([crop, cost, profit, roi, sustain, total_score])

        results_df = pd.DataFrame(results, columns=["Crop", "Cost (₹)", "Profit (₹)", "ROI (₹)", "Sustainability", "Total Score"])
        top3 = results_df.sort_values("Total Score", ascending=False).head(3)

        st.success(f"✅ Recommended Crop (ML Model): **{main_crop.capitalize()}**")
        if main_crop in crop_images and os.path.exists(crop_images[main_crop]):
            st.image(crop_images[main_crop], width=300, caption=f"Predicted: {main_crop.capitalize()}")

        st.subheader("📊 Top 3 Best Crops within Budget")
        def colorize(val, col):
            if col == "ROI (₹)":
                if val > 40000: return "background-color: lightgreen"
                elif val > 20000: return "background-color: khaki"
                else: return "background-color: salmon"
            if col == "Sustainability":
                if val >= 8: return "background-color: lightgreen"
                elif val >= 6: return "background-color: khaki"
                else: return "background-color: salmon"
            return ""
        
        st.dataframe(top3.style.applymap(lambda v: colorize(v, "ROI (₹)"), subset=["ROI (₹)"])
                              .applymap(lambda v: colorize(v, "Sustainability"), subset=["Sustainability"]))

        st.subheader("📈 Total Score Comparison (Top 3 Crops)")
        fig, ax = plt.subplots()
        sns.barplot(x="Crop", y="Total Score", data=top3, palette="viridis", ax=ax)
        ax.set_title("Top 3 Crop Scores")
        st.pyplot(fig)

        st.subheader("💵 Profit vs Cost")
        fig2, ax2 = plt.subplots()
        top3_melted = top3.melt(id_vars="Crop", value_vars=["Cost (₹)", "Profit (₹)"], var_name="Type", value_name="Amount")
        sns.barplot(x="Crop", y="Amount", hue="Type", data=top3_melted, ax=ax2)
        ax2.set_title("Profit vs Cost")
        st.pyplot(fig2)

        st.download_button("📥 Download Top 3 Results", top3.to_csv(index=False), "top3_crops.csv", "text/csv")



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

