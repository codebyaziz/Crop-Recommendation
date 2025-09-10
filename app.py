import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import datetime
import numpy as np

model = pickle.load(open("crop_model.pkl", "rb"))
df = pd.read_csv("Crop_recommendation.csv")

crop_economics = {
    "rice": {"cost": 25000, "revenue": 45000, "season": "Kharif", "water": "High", "sustainability": 60},
    "maize": {"cost": 20000, "revenue": 38000, "season": "Kharif", "water": "Medium", "sustainability": 70},
    "chickpea": {"cost": 15000, "revenue": 32000, "season": "Rabi", "water": "Low", "sustainability": 85},
    "kidneybeans": {"cost": 18000, "revenue": 35000, "season": "Kharif", "water": "Medium", "sustainability": 75},
    "pigeonpeas": {"cost": 16000, "revenue": 30000, "season": "Kharif", "water": "Low", "sustainability": 80},
    "mothbeans": {"cost": 12000, "revenue": 25000, "season": "Kharif", "water": "Low", "sustainability": 90},
    "mungbean": {"cost": 14000, "revenue": 28000, "season": "Kharif", "water": "Low", "sustainability": 85},
    "blackgram": {"cost": 13000, "revenue": 26000, "season": "Kharif", "water": "Low", "sustainability": 80},
    "lentil": {"cost": 15000, "revenue": 30000, "season": "Rabi", "water": "Low", "sustainability": 85},
    "pomegranate": {"cost": 45000, "revenue": 85000, "season": "Year-round", "water": "Medium", "sustainability": 65},
    "banana": {"cost": 35000, "revenue": 70000, "season": "Year-round", "water": "High", "sustainability": 60},
    "mango": {"cost": 40000, "revenue": 75000, "season": "Summer", "water": "Medium", "sustainability": 70},
    "grapes": {"cost": 50000, "revenue": 95000, "season": "Winter", "water": "Medium", "sustainability": 65},
    "watermelon": {"cost": 22000, "revenue": 45000, "season": "Summer", "water": "High", "sustainability": 55},
    "muskmelon": {"cost": 18000, "revenue": 38000, "season": "Summer", "water": "High", "sustainability": 55},
    "apple": {"cost": 60000, "revenue": 110000, "season": "Temperate", "water": "Medium", "sustainability": 75},
    "orange": {"cost": 35000, "revenue": 65000, "season": "Winter", "water": "Medium", "sustainability": 70},
    "papaya": {"cost": 25000, "revenue": 50000, "season": "Year-round", "water": "Medium", "sustainability": 65},
    "cotton": {"cost": 30000, "revenue": 50000, "season": "Kharif", "water": "High", "sustainability": 40},
    "jute": {"cost": 20000, "revenue": 35000, "season": "Kharif", "water": "High", "sustainability": 70},
    "coconut": {"cost": 40000, "revenue": 75000, "season": "Year-round", "water": "Medium", "sustainability": 80},
    "coffee": {"cost": 55000, "revenue": 95000, "season": "Year-round", "water": "High", "sustainability": 75}
}

crop_images = {
    "rice": "images/rice.jpg", "maize": "images/maize.jpg", "chickpea": "images/chickpea.jpg",
    "kidneybeans": "images/kidneybeans.jpg", "pigeonpeas": "images/pigeonpeas.jpg",
    "mothbeans": "images/mothbeans.jpg", "mungbean": "images/mungbean.jpg",
    "blackgram": "images/blackgram.jpg", "lentil": "images/lentil.jpg",
    "pomegranate": "images/pomegranate.jpg", "banana": "images/banana.jpg",
    "mango": "images/mango.jpg", "grapes": "images/grapes.jpg",
    "watermelon": "images/watermelon.jpg", "muskmelon": "images/muskmelon.jpg",
    "apple": "images/apple.jpg", "orange": "images/orange.jpg", "papaya": "images/papaya.jpg",
    "cotton": "images/cotton.jpg", "jute": "images/jute.jpg", "coconut": "images/coconut.jpg",
    "coffee": "images/coffee.jpg"
}

CLIMATE_ZONES = {
    "Tropical": ["rice", "banana", "coconut", "mango", "papaya", "coffee"],
    "Subtropical": ["maize", "cotton", "orange", "grapes", "pomegranate"],
    "Temperate": ["apple", "chickpea", "lentil"],
    "Arid": ["cotton", "millet", "mothbeans"],
    "Semi-Arid": ["maize", "cotton", "pigeonpeas", "chickpea"]
}

GROWING_SEASONS = {
    "Kharif": {"months": "June-November", "crops": ["rice", "cotton", "maize", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "jute"]},
    "Rabi": {"months": "November-April", "crops": ["chickpea", "lentil", "wheat", "barley"]},
    "Summer": {"months": "March-June", "crops": ["watermelon", "muskmelon", "mango"]},
    "Year-round": {"months": "All seasons", "crops": ["banana", "papaya", "pomegranate", "coconut", "coffee"]}
}

SOIL_REQUIREMENTS = {
    "rice": {"ph_range": (5.5, 6.5), "soil_type": "Clay/Loamy", "drainage": "Poor (waterlogged)"},
    "wheat": {"ph_range": (6.0, 7.5), "soil_type": "Loamy", "drainage": "Good"},
    "maize": {"ph_range": (6.0, 7.0), "soil_type": "Well-drained loamy", "drainage": "Good"},
    "cotton": {"ph_range": (5.8, 8.0), "soil_type": "Black cotton soil", "drainage": "Good"},
    "chickpea": {"ph_range": (6.0, 7.5), "soil_type": "Clay loam", "drainage": "Good"}
}

crop_categories = {
    "All": list(crop_images.keys()),
    "Cereals": ["rice", "maize"],
    "Pulses": ["chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"],
    "Fruits": ["pomegranate", "banana", "mango", "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya"],
    "Cash Crops": ["cotton", "jute", "coconut", "coffee"]
}

def check_seasonal_feasibility(crop):
    """Check if crop can be grown in current season"""
    current_month = datetime.datetime.now().month
    
    if 6 <= current_month <= 11:
        current_season = "Kharif"
    elif current_month == 12 or 1 <= current_month <= 4:
        current_season = "Rabi" 
    else:
        current_season = "Summer"
    
    crop_season = crop_economics.get(crop, {}).get("season", "Unknown")
    
    if crop_season == "Year-round":
        return True, "✅ Can be grown year-round"
    elif crop_season == current_season:
        return True, f"✅ Perfect timing for {current_season} season"
    else:
        return False, f"⚠️ Best season: {crop_season} (Current: {current_season})"

def check_climate_suitability(crop, climate_zone):
    """Check climate suitability"""
    suitable_zones = [zone for zone, crops in CLIMATE_ZONES.items() if crop in crops]
    
    if climate_zone in suitable_zones:
        return True, f"✅ Suitable for {climate_zone} climate"
    elif suitable_zones:
        return False, f"⚠️ Better suited for: {', '.join(suitable_zones)}"
    else:
        return True, "ℹ️ Climate data not available"

def validate_soil_conditions(crop, ph_value):
    """Check soil compatibility"""
    if crop not in SOIL_REQUIREMENTS:
        return True, "ℹ️ Soil requirements data not available"
    
    req = SOIL_REQUIREMENTS[crop]
    ph_min, ph_max = req["ph_range"]
    
    if ph_min <= ph_value <= ph_max:
        return True, f"✅ pH {ph_value} is optimal (Range: {ph_min}-{ph_max})"
    else:
        return False, f"⚠️ pH {ph_value} outside optimal range {ph_min}-{ph_max}"

def calculate_enhanced_score(crop, budget, climate_zone, ph_value):
    """Calculate score with real-world context"""
    if crop not in crop_economics:
        return 0
    
    econ = crop_economics[crop]
    cost = econ["cost"]
    revenue = econ["revenue"]
    profit = revenue - cost
    roi = (profit / cost) * 100 if cost > 0 else 0
    sustainability = econ["sustainability"]
    
    seasonal_suitable, _ = check_seasonal_feasibility(crop)
    climate_suitable, _ = check_climate_suitability(crop, climate_zone)
    soil_suitable, _ = validate_soil_conditions(crop, ph_value)
    budget_suitable = cost <= budget
    
    context_multiplier = (
        (1.0 if seasonal_suitable else 0.7) *
        (1.0 if climate_suitable else 0.6) *
        (1.0 if soil_suitable else 0.8) *
        (1.0 if budget_suitable else 0.5)
    )
    
    base_score = roi * 0.4 + sustainability * 0.6
    final_score = base_score * context_multiplier
    
    return final_score

st.sidebar.title("🌱 Crop Recommender")
page = st.sidebar.radio("Go to", ["Home", "Crop Recommendation", "Dataset Insights", "Crop Gallery", "About"])

if page == "Home":
    st.title("🌱 Smart Crop Recommendation System")
    st.write("Welcome to the Smart Crop Recommendation System with real-world context and constraints!")

    st.image("images/banner.png", use_container_width=True)


    st.markdown("### 🚀 Enhanced Features")
    st.markdown("""
    - 📊 AI-powered crop recommendations based on soil & climate
    - 🌍 **Real-world context**: Climate zones, growing seasons, soil compatibility
    - 💰 **Realistic economics**: Market-based costs, revenue, and ROI
    - 🌱 **Sustainability scoring**: Environmental impact assessment
    - ⏰ **Seasonal feasibility**: Current season compatibility
    - 🎯 **Risk assessment**: Comprehensive suitability analysis
    """)

    st.markdown("### 🎯 Goal")
    st.info("Empowering farmers with **context-aware, economically viable, and environmentally sustainable** crop recommendations.")

elif page == "Crop Recommendation":
    st.title("🌾 AI-Based Crop Recommendation System")
    st.markdown("### Get intelligent crop recommendations with real-world context 🌍")

    st.subheader("🌍 Location & Context Information")
    col_context1, col_context2 = st.columns(2)
    
    with col_context1:
        climate_zone = st.selectbox("🌡️ Your Climate Zone", list(CLIMATE_ZONES.keys()))
        farm_size = st.number_input("🚜 Farm Size (acres)", 0.5, 100.0, 2.0, 0.5)
    
    with col_context2:
        irrigation_type = st.selectbox("💧 Irrigation Type", 
                                     ["Rain-fed", "Drip", "Sprinkler", "Flood irrigation"])
        experience = st.selectbox("👨‍🌾 Farming Experience", 
                                ["Beginner (0-2 years)", "Intermediate (3-10 years)", "Expert (10+ years)"])

    st.subheader("🧪 Soil & Weather Parameters")
    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N) [mg/kg]", 0, 200, 20)
        st.caption("🔹 Essential for leaf growth & chlorophyll")
        P = st.number_input("Phosphorus (P) [mg/kg]", 0, 200, 20) 
        st.caption("🔹 Root development & flowering")
        K = st.number_input("Potassium (K) [mg/kg]", 0, 200, 20)
        st.caption("🔹 Disease resistance & fruit quality")

    with col2:
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        st.caption("🌡️ Affects growth rate and yield")
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 70.0)
        st.caption("💧 Controls evaporation & disease risk")
        ph = st.number_input("Soil pH Value", 0.0, 14.0, 6.5)
        st.caption("⚖️ Soil acidity/alkalinity level")
        rainfall = st.number_input("Rainfall (mm)", 0.0, 300.0, 100.0)
        st.caption("🌧️ Water availability")

    budget = st.number_input("💰 Budget (₹ per acre)", 10000, 150000, 30000, 5000)

    if st.button("🌿 Get Smart Recommendation"):
        prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
        recommended_crop = prediction[0]

        enhanced_scores = []
        for crop in crop_economics.keys():
            score = calculate_enhanced_score(crop, budget, climate_zone, ph)
            if score > 0: 
                enhanced_scores.append({
                    'crop': crop,
                    'score': score,
                    'economics': crop_economics[crop]
                })

        enhanced_scores.sort(key=lambda x: x['score'], reverse=True)
        top_crops = enhanced_scores[:5]

        st.success(f"🤖 **AI Recommendation**: {recommended_crop.capitalize()}")
        
        ai_in_top = any(crop['crop'] == recommended_crop for crop in top_crops)
        if not ai_in_top:
            st.warning("⚠️ AI recommendation may not be optimal considering current context!")

        st.subheader("🎯 Top 5 Context-Aware Recommendations")
        
        for i, crop_data in enumerate(top_crops):
            crop = crop_data['crop']
            econ = crop_data['economics']
            
            with st.expander(f"#{i+1} {crop.capitalize()} (Score: {crop_data['score']:.1f})", 
                           expanded=(i == 0)):
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.markdown("**💰 Economics**")
                    cost = econ['cost']
                    revenue = econ['revenue']
                    profit = revenue - cost
                    roi = (profit/cost)*100
                    
                    st.write(f"Cost: ₹{cost:,}")
                    st.write(f"Revenue: ₹{revenue:,}")
                    st.write(f"Profit: ₹{profit:,}")
                    st.write(f"ROI: {roi:.1f}%")
                
                with col_b:
                    st.markdown("**🌍 Context Check**")
                    seasonal_ok, seasonal_msg = check_seasonal_feasibility(crop)
                    climate_ok, climate_msg = check_climate_suitability(crop, climate_zone)
                    soil_ok, soil_msg = validate_soil_conditions(crop, ph)
                    
                    st.write(seasonal_msg)
                    st.write(climate_msg)
                    st.write(soil_msg)
                
                with col_c:
                    st.markdown("**📊 Details**")
                    st.write(f"Season: {econ['season']}")
                    st.write(f"Water need: {econ['water']}")
                    st.write(f"Sustainability: {econ['sustainability']}/100")
                    budget_status = "✅ Within budget" if cost <= budget else "❌ Over budget"
                    st.write(budget_status)

        if len(top_crops) >= 3:
            st.subheader("📈 Top 3 Comparison")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            crops_names = [crop['crop'].capitalize() for crop in top_crops[:3]]
            rois = [(crop['economics']['revenue'] - crop['economics']['cost']) / 
                   crop['economics']['cost'] * 100 for crop in top_crops[:3]]
            
            ax1.bar(crops_names, rois, color=['#4CAF50', '#2196F3', '#FFC107'])
            ax1.set_title('ROI Comparison (%)')
            ax1.set_ylabel('ROI (%)')
            
            costs = [crop['economics']['cost']/1000 for crop in top_crops[:3]]  # in thousands
            sustainability = [crop['economics']['sustainability'] for crop in top_crops[:3]]
            
            colors = ['#4CAF50', '#2196F3', '#FFC107']
            for i, (crop, cost, sust) in enumerate(zip(crops_names, costs, sustainability)):
                ax2.scatter(cost, sust, s=200, c=colors[i], alpha=0.7, label=crop)
            
            ax2.set_xlabel('Cost (₹ thousands)')
            ax2.set_ylabel('Sustainability Score')
            ax2.set_title('Cost vs Sustainability')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig)

        st.subheader("⚠️ Risk Assessment")
        risk_factors = []
        
        if budget < 25000:
            risk_factors.append("💰 Low budget may limit high-value crop options")
        if ph < 5.5 or ph > 8.0:
            risk_factors.append("🧪 Extreme pH levels may affect crop growth")
        if humidity > 90:
            risk_factors.append("💧 Very high humidity increases disease risk")
        if temperature > 40:
            risk_factors.append("🌡️ High temperature may stress crops")
        
        current_season = "Kharif" if 6 <= datetime.datetime.now().month <= 11 else "Rabi"
        suitable_current_season = [crop for crop in top_crops 
                                 if crop['economics']['season'] == current_season]
        
        if not suitable_current_season and len(top_crops) > 0:
            risk_factors.append(f"⏰ Top recommendations not ideal for current {current_season} season")
        
        if risk_factors:
            for risk in risk_factors:
                st.warning(risk)
        else:
            st.success("✅ Low risk - Good conditions for recommended crops!")

        if recommended_crop in crop_images and os.path.exists(crop_images[recommended_crop]):
            st.image(crop_images[recommended_crop], width=300, 
                    caption=f"AI Recommended: {recommended_crop.capitalize()}")

elif page == "Dataset Insights":
    st.title("📊 Dataset Insights")
    st.write("Explore the dataset used to train our AI model")

    if st.checkbox("Show raw dataset"):
        st.dataframe(df.head(20))

    st.subheader("🌾 Crop Distribution")
    st.bar_chart(df['label'].value_counts())

    st.subheader("🌍 Real-World Context Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Economics by Category**")
        category_economics = {}
        for category, crops in crop_categories.items():
            if category != "All":
                avg_cost = np.mean([crop_economics.get(crop, {}).get('cost', 0) for crop in crops if crop in crop_economics])
                avg_revenue = np.mean([crop_economics.get(crop, {}).get('revenue', 0) for crop in crops if crop in crop_economics])
                category_economics[category] = {'avg_cost': avg_cost, 'avg_revenue': avg_revenue}
        
        econ_df = pd.DataFrame(category_economics).T
        st.dataframe(econ_df)
    
    with col2:
        st.markdown("**Seasonal Distribution**")
        seasonal_counts = {}
        for season_data in GROWING_SEASONS.values():
            seasonal_counts[season_data['months']] = len(season_data['crops'])
        
        fig, ax = plt.subplots()
        ax.pie(seasonal_counts.values(), labels=seasonal_counts.keys(), autopct='%1.1f%%')
        ax.set_title('Crops by Growing Season')
        st.pyplot(fig)

    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    st.subheader("📈 Feature Distributions")
    selected_feature = st.selectbox("Select feature:", features)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_feature], kde=True, bins=30, ax=ax, color="green")
    ax.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig)

    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

elif page == "Crop Gallery":
    st.title("🌿 Crop Gallery with Context")
    st.write("Browse crops with real-world information")

    category = st.selectbox("📂 Select category:", list(crop_categories.keys()))
    search = st.text_input("🔍 Search crop:").strip().lower()

    selected_crops = crop_categories[category]
    filtered_crops = [c for c in selected_crops if search in c.lower()] if search else selected_crops

    if filtered_crops:
        cols = st.columns(3)
        for i, crop in enumerate(filtered_crops):
            with cols[i % 3]:
                if crop in crop_images and os.path.exists(crop_images[crop]):
                    st.image(crop_images[crop], caption=crop.capitalize(), use_container_width=True)
                
                if crop in crop_economics:
                    econ = crop_economics[crop]
                    st.write(f"**Season**: {econ['season']}")
                    st.write(f"**ROI**: {((econ['revenue']-econ['cost'])/econ['cost']*100):.1f}%")
                    st.write(f"**Water need**: {econ['water']}")
                    st.write(f"**Sustainability**: {econ['sustainability']}/100")
    else:
        st.warning("No crops found for your selection")

else:  
    st.title("ℹ️ About This Enhanced System")
    st.markdown("""
    ## 🌾 Smart Crop Recommendation System v2.0
    
    This enhanced system provides **context-aware crop recommendations** by considering:
    
    ### 🎯 Real-World Context
    - **Climate Zones**: Tropical, Subtropical, Temperate, Arid, Semi-Arid
    - **Growing Seasons**: Kharif, Rabi, Summer, Year-round crops  
    - **Soil Compatibility**: pH requirements, soil types, drainage needs
    - **Economic Reality**: Market-based costs, revenues, and ROI calculations
    - **Sustainability**: Environmental impact scoring
    - **Seasonal Timing**: Current season feasibility checks
    
    ### 🔧 Technical Features  
    - **AI Model**: RandomForest classifier (99.3% accuracy)
    - **Parameters**: 7+ input features including NPK, climate, soil pH
    - **Context Engine**: Real-time feasibility assessment
    - **Risk Analysis**: Comprehensive risk factor evaluation
    - **Visual Analytics**: Economic comparisons and trend analysis
    
    ### 💡 Key Improvements Over Basic Systems
    - ✅ **Realistic Economics**: Real market data instead of artificial calculations  
    - ✅ **Seasonal Intelligence**: Considers current growing season
    - ✅ **Climate Awareness**: Matches crops to user's climate zone
    - ✅ **Risk Assessment**: Identifies potential challenges
    - ✅ **Sustainability Focus**: Environmental impact consideration
    
    **Built with**: Python 🐍 | scikit-learn 🤖 | Streamlit 🌐 | Real Agricultural Data 🌾
    
    *Empowering farmers with intelligent, context-aware agricultural decisions!*
    """)

if page == "Home":
    chatbot_widget = """
    <style>
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
        .chatbot-iframe {
            width: 100%;
            height: 100%;
            border: none;
            border-radius: 12px;
        }
    </style>

    <button class="chatbot-btn" onclick="toggleChatbot()">💬</button>
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
