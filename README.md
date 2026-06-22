🌾 Smart Crop Recommendation System
An AI-powered crop recommendation system that goes beyond a single model prediction — combining a neural network with explainability, economic context, fertilizer guidance, crop rotation planning, and a soil-type image classifier, all wrapped in a Flask web app.

✨ Features
🔮 Crop Recommendation
Neural network classifier trained on soil nutrients (N, P, K), temperature, humidity, pH, and rainfall
Confidence scoring — shows the model's actual prediction confidence, not just a label
Top-3 predictions — see the model's next-best guesses, not just the top one
District-based weather lookup (live data via Open-Meteo API) or manual input
Context-aware scoring layered on top of the AI prediction: climate zone, growing season, soil pH compatibility, and budget
🧠 Explainability — "Why this crop?"
For every prediction, each input feature is compared against that crop's typical profile in the training data and labeled strong match / good match / borderline / unusual — so the recommendation isn't a black box.
🧪 Fertilizer Suggestions
Flags meaningful gaps between your soil's N-P-K levels and the recommended crop's ideal range (e.g. "Nitrogen is low — consider adding ~70 kg/ha").
🔄 Crop Rotation Planning
Suggests a complementary crop for next season based on agronomic family — e.g. a legume after a nitrogen-hungry cereal to help restore soil fertility — ranked by pH compatibility with your current soil.
📊 Model Comparison
A dedicated page trains a Random Forest and a Gradient Boosting classifier live, on the same train/test split used to evaluate the neural network, and reports:
Accuracy, macro-precision, macro-recall, macro-F1 (macro-averaged so strong performance on common crops can't mask weak performance on rare ones)
Per-model confusion matrices
📸 Soil Type Detection (Image Classification)
Upload a photo of soil and a CNN — trained from scratch on a public, labeled dataset of soil photographs — identifies the soil type (Alluvial, Arid, Black, Laterite, Mountain, Red, or Yellow), with:
General agronomic guidance on what tends to grow well in that soil type
Transparent per-class accuracy reporting — including an honest flag that Alluvial soil detection is currently weaker due to limited training data for that class
💰 Economics & Risk
Real cost, revenue, ROI, and sustainability scoring per crop
Seasonal feasibility checks (Kharif / Rabi / Summer)
Risk factor analysis (budget, pH, humidity, heat)
📈 Dataset Insights
Crop distribution, feature histograms, correlation heatmaps, and seasonal breakdowns — all rendered server-side.
🖼️ Crop Gallery
Browse all crops by category (cereals, pulses, fruits, cash crops) with economics and sustainability data.
---
🛠️ Tech Stack
Layer	Technology
Backend	Flask
ML — Crop Recommendation	TensorFlow/Keras neural network, exported to TensorFlow Lite for lightweight deployment
ML — Soil Detection	Custom CNN trained from scratch, exported to TensorFlow Lite
ML — Comparison Models	scikit-learn (Random Forest, HistGradientBoosting)
Inference Runtime	`ai-edge-litert` (lightweight TFLite interpreter, ~20MB vs ~600MB for full TensorFlow)
Data	pandas, NumPy
Visualization	Matplotlib, Seaborn (server-side rendering)
Image Handling	Pillow
Weather Data	Open-Meteo API
Deployment	Gunicorn, Render-ready (`render.yaml`, `Procfile`)
📂 Project Structure
```
crop_recommendation_flask/
├── app.py                      
├── requirements.txt
├── Procfile                    
├── render.yaml                 
├── runtime.txt                 
├── nn_model.tflite              
├── soil_model.tflite            
├── soil_class_names.txt
├── scaler.pkl
├── label_encoder.pkl
├── Crop_recommendation.csv      
├── templates/                   
└── static/
    └── images/                  
```
---
🚀 Getting Started
Prerequisites
Python 3.10+
pip
Installation
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd crop_recommendation_flask

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```
Run locally
```bash
python app.py
```
Open http://localhost:5000. First startup trains the Random Forest / Gradient Boosting comparison models — takes a few seconds.
---
🧭 Pages
Route	Description
`/`	Home
`/recommend`	Crop recommendation — confidence, explainability, fertilizer tips, rotation advice
`/soil-detector`	Soil type detection from an uploaded photo
`/insights`	Dataset insights and visualizations
`/model-comparison`	Neural Network vs Random Forest vs Gradient Boosting
`/gallery`	Crop gallery with economics
`/about`	About the project
