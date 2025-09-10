import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components



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

crop_costs = {crop: (i + 1) * 1000 for i, crop in enumerate(crop_images.keys())}
crop_revenue = {crop: (i + 1) * 1800 for i, crop in enumerate(crop_images.keys())}
sustainability = {crop: (i % 5 + 1) * 20 for i, crop in enumerate(crop_images.keys())}  # 20–100 scale


st.sidebar.title("🌱 Crop Recommender")
page = st.sidebar.radio("Go to", ["Home", "Crop Recommendation", "Dataset Insights", "Crop Gallery", "About"])


if page == "Home":
    st.title("🌱 Smart Crop Recommendation System")
    st.write("Welcome to the Smart Crop Recommendation System. This tool helps farmers choose the best crops based on soil and weather conditions, budget, and sustainability.")


    st.markdown("### 🚀 Features")
    st.markdown("""
    - 📊 Intelligent crop recommendation based on soil nutrients and climate  
    - 💰 Budget planning with ROI & sustainability scoring  
    - 🖼️ Crop gallery with detailed info  
    - 📈 Visualizations for better decisions  
    - 📑 Exportable PDF reports  
    """)

    st.markdown("### 🎯 Goal")
    st.info("Our mission is to empower farmers with AI-driven insights for **profitable, sustainable, and data-informed agriculture.**")

    # ------------------------
    # 🤖 AI Chatbot Embed
    # ------------------------
    st.markdown("## 🤖 AI Assistant")
    st.write("Ask questions and get instant help!")

    
elif page == "Crop Recommendation":
    
    st.title("🌾 AI-Based Crop Recommendation System")
    st.markdown("### Helping farmers choose the best crop based on soil & weather data 🌦️")

    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N) [mg/kg]", 0, 200)
        st.caption("🔹 Boosts leaf growth & chlorophyll formation.")
        P = st.number_input("Phosphorus (P) [mg/kg]", 0, 200)
        st.caption("🔹 Supports root development & flowering.")
        K = st.number_input("Potassium (K) [mg/kg]", 0, 200)
        st.caption("🔹 Increases resistance & improves fruit quality.")

    with col2:
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
        st.caption("🌡️ Affects crop growth rate and yield.")
        humidity = st.number_input("Humidity (%)", 0.0, 100.0)
        st.caption("💧 Controls water evaporation & disease risk.")
        ph = st.number_input("Soil pH Value", 0.0, 14.0)
        st.caption("⚖️ Determines soil acidity/alkalinity.")
        rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0)
        st.caption("🌧️ Water available during crop cycle.")

    budget = st.number_input("💰 Enter your budget (in ₹)", 1000, 100000, step=1000)

    if st.button("🌿 Recommend Crop"):
        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        crop = prediction[0]

        st.success(f"✅ Best Recommended Crop: **{crop.capitalize()}**")

        scores = []
        for c in crop_images.keys():
            cost = crop_costs[c]
            revenue = crop_revenue[c]
            profit = revenue - cost
            roi = (profit / cost) * 100
            sustain = sustainability[c]
            total_score = roi * 0.6 + sustain * 0.4
            if cost <= budget:
                scores.append([c, cost, revenue, profit, roi, sustain, total_score])

        top3 = sorted(scores, key=lambda x: x[-1], reverse=True)[:3]

        if top3:
            st.subheader("📊 Top 3 Best Crops within Budget")

            styled_df = pd.DataFrame(top3, columns=["Crop", "Cost", "Revenue", "Profit", "ROI (%)", "Sustainability", "Total Score"])
            styled_df = styled_df.style.background_gradient(subset=["ROI (%)"], cmap="Greens") \
                                       .background_gradient(subset=["Sustainability"], cmap="Blues") \
                                       .background_gradient(subset=["Total Score"], cmap="YlOrRd")
            st.dataframe(styled_df, use_container_width=True)

            st.subheader("📈 Score Comparison")
            fig, ax = plt.subplots()
            crops = [x[0] for x in top3]
            scores_val = [x[-1] for x in top3]
            ax.bar(crops, scores_val, color=["#4CAF50", "#2196F3", "#FFC107"])
            ax.set_ylabel("Total Score")
            st.pyplot(fig)

            st.subheader("💰 Profit vs Cost Comparison")
            fig2, ax2 = plt.subplots()
            bar_width = 0.35
            index = range(len(top3))
            costs = [x[1] for x in top3]
            profits = [x[3] for x in top3]
            ax2.bar(index, costs, bar_width, label="Cost", color="red")
            ax2.bar([i + bar_width for i in index], profits, bar_width, label="Profit", color="green")
            ax2.set_xticks([i + bar_width / 2 for i in index])
            ax2.set_xticklabels(crops)
            ax2.set_ylabel("₹ (INR)")
            ax2.legend()
            st.pyplot(fig2)

        if crop in crop_images and os.path.exists(crop_images[crop]):
            st.image(crop_images[crop], width=300, caption=f"Recommended: {crop.capitalize()}")


elif page == "Dataset Insights":
    st.title("📊 Dataset Insights")
    st.write("This is the dataset used to train the model.")

    if st.checkbox("Show raw dataset"):
        st.dataframe(df.head(20))

    st.subheader("🌾 Crop Distribution")
    st.bar_chart(df['label'].value_counts())

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    st.subheader("📈 Feature Distributions")
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
    search = st.text_input("🔍 Search for a crop:").strip().lower()

    selected_crops = crop_categories[category]
    filtered_crops = {c: crop_images[c] for c in selected_crops if search in c.lower()} if search else {c: crop_images[c] for c in selected_crops}

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
    and soil pH.

    ### 🔍 Features
    - AI-powered crop prediction
    - Cost, revenue, profit, and ROI analysis
    - Sustainability scoring
    - Top 3 crop comparison with charts
    - Interactive dataset insights
    - Crop gallery with images & categories

    **Tech Stack:**
    - Python 🐍  
    - scikit-learn 🤖  
    - Streamlit 🌐  
    - Seaborn & Matplotlib 📊  

    Developed as part of a Hackathon to support sustainable agriculture 🌱.
    """)


if page in ["Home"]:
    chatbot_widget = """
    <style>
        /* Floating chat button */
        .chatbot-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 28px;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            z-index: 10000;
        }

        /* Chatbot container */
        .chatbot-container {
            position: fixed;
            bottom: 90px;
            right: 20px;
            width: 350px;
            height: 500px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            background: white;
            display: none;
            flex-direction: column;
            z-index: 9999;
        }

        /* Close button */
        .chatbot-close {
            background: #ff4d4d;
            color: white;
            border: none;
            border-radius: 50%;
            width: 25px;
            height: 25px;
            font-size: 16px;
            cursor: pointer;
            position: absolute;
            top: 8px;
            right: 8px;
            z-index: 10001;
        }

        /* Chatbot iframe */
        .chatbot-iframe {
            width: 100%;
            height: 100%;
            border: none;
            border-radius: 12px;
        }
    </style>

    <!-- Floating Chat Button -->
    <button class="chatbot-btn" onclick="toggleChatbot()">💬</button>

    <!-- Chatbot Window -->
    <div class="chatbot-container" id="chatbot-box">
        <button class="chatbot-close" onclick="closeChatbot()">✖</button>
        <iframe
            src="https://cdn.jotfor.ms/agent/0199320e138f7b0286892804f65a537060c7"
            class="chatbot-iframe">
        </iframe>
    </div>

    <script>
        function toggleChatbot() {
            var box = document.getElementById("chatbot-box");
            if (box.style.display === "none" || box.style.display === "") {
                box.style.display = "flex";
            } else {
                box.style.display = "none";
            }
        }
        function closeChatbot() {
            document.getElementById("chatbot-box").style.display = "none";
        }
    </script>
    """
    components.html(chatbot_widget, height=650, width=400)


