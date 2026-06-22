import os
import io
import base64
import pickle
import datetime

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed for a server
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from ai_edge_litert.interpreter import Interpreter
from flask import Flask, render_template, request, jsonify

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model + preprocessing objects once at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Using a TFLite interpreter instead of full TensorFlow/Keras at runtime.
# Same trained weights, same predictions (verified to match within 1e-7),
# but the runtime dependency is ~20MB instead of 600MB+, which matters a lot
# on free-tier hosting where RAM and build size are capped.
_interpreter = Interpreter(model_path=os.path.join(BASE_DIR, "nn_model.tflite"))
_interpreter.allocate_tensors()
_input_details = _interpreter.get_input_details()
_output_details = _interpreter.get_output_details()


def nn_predict(X):
    """Run the TFLite model on a batch of feature rows, return softmax probs."""
    X = np.asarray(X, dtype=np.float32)
    results = np.zeros((X.shape[0], _output_details[0]["shape"][-1]), dtype=np.float32)
    for i in range(X.shape[0]):
        _interpreter.set_tensor(_input_details[0]["index"], X[i : i + 1])
        _interpreter.invoke()
        results[i] = _interpreter.get_tensor(_output_details[0]["index"])[0]
    return results


with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

# --- Soil-type image classifier (separate model, separate concern) -------
# Detects soil type from a photo: Alluvial, Arid, Black, Laterite, Mountain,
# Red, Yellow. Trained from scratch (no internet access to pretrained
# ImageNet weights in this environment) on a public 1,188-image dataset.
# Validation accuracy is ~70% overall; Alluvial soil specifically is weaker
# (~25% recall) due to having only 51 training images for that class - this
# is surfaced honestly in the UI rather than hidden.
_soil_interpreter = Interpreter(model_path=os.path.join(BASE_DIR, "soil_model.tflite"))
_soil_interpreter.allocate_tensors()
_soil_input_details = _soil_interpreter.get_input_details()
_soil_output_details = _soil_interpreter.get_output_details()
_SOIL_IMG_SIZE = (160, 160)

with open(os.path.join(BASE_DIR, "soil_class_names.txt")) as f:
    SOIL_CLASS_NAMES = f.read().strip().split("\n")

# Soil classes with notably weaker validation recall, to caveat in the UI.
SOIL_LOW_CONFIDENCE_CLASSES = {"Alluvial_Soil"}


def soil_predict(pil_image):
    """Run the TFLite soil classifier on a PIL image, return (label, confidence, all_probs)."""
    img = pil_image.convert("RGB").resize(_SOIL_IMG_SIZE)
    arr = np.array(img, dtype=np.float32)[np.newaxis, ...]
    _soil_interpreter.set_tensor(_soil_input_details[0]["index"], arr)
    _soil_interpreter.invoke()
    probs = _soil_interpreter.get_tensor(_soil_output_details[0]["index"])[0]
    top_idx = int(np.argmax(probs))
    label = SOIL_CLASS_NAMES[top_idx]
    ranked = sorted(
        [{"label": SOIL_CLASS_NAMES[i], "confidence": round(float(probs[i]) * 100, 1)} for i in range(len(probs))],
        key=lambda x: x["confidence"],
        reverse=True,
    )
    return label, round(float(probs[top_idx]) * 100, 1), ranked


# General agronomic guidance by soil type (not derived from the NPK dataset -
# this is separate, broader knowledge about what each soil type suits).
SOIL_CROP_GUIDANCE = {
    "Alluvial_Soil": {
        "description": "Fertile, found in river plains and floodplains. Rich in potash, phosphoric acid, and lime; among the most productive soils for intensive cultivation.",
        "suited_crops": ["Rice", "Wheat", "Sugarcane", "Maize", "Pulses", "Oilseeds"],
    },
    "Arid_Soil": {
        "description": "Sandy, low in moisture and organic matter, found in dry/desert regions. Drains quickly and needs drought-tolerant crops.",
        "suited_crops": ["Bajra (Pearl Millet)", "Barley", "Moth Beans", "Guar", "Cotton (with irrigation)"],
    },
    "Black_Soil": {
        "description": "Also called 'black cotton soil' — clay-rich, retains moisture well, develops deep cracks when dry. Excellent for cotton and oilseeds.",
        "suited_crops": ["Cotton", "Soybean", "Sugarcane", "Wheat", "Citrus fruits"],
    },
    "Laterite_Soil": {
        "description": "Found in high-rainfall tropical regions, acidic and low in fertility but workable with proper amendments.",
        "suited_crops": ["Cashew", "Tea", "Coffee", "Coconut", "Tapioca"],
    },
    "Mountain_Soil": {
        "description": "Found in hilly/mountainous terrain, variable depth and fertility, generally cooler growing conditions.",
        "suited_crops": ["Tea", "Coffee", "Spices", "Apple", "Temperate fruits"],
    },
    "Red_Soil": {
        "description": "Iron-oxide rich (hence the color), generally low in nitrogen and humus but responds well to fertilization.",
        "suited_crops": ["Groundnut", "Millets", "Pulses", "Potato", "Tobacco"],
    },
    "Yellow_Soil": {
        "description": "A hydrated form of red soil, similar properties but typically found in slightly wetter conditions.",
        "suited_crops": ["Pulses", "Oilseeds", "Millets", "Maize"],
    },
}

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

df = pd.read_csv(os.path.join(BASE_DIR, "Crop_recommendation.csv"))

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Per-crop ideal profile (mean + std) computed once from the training data.
# Used for explainability (how well do the user's inputs match this crop's
# typical conditions) and for fertilizer gap suggestions.
_crop_stats = df.groupby("label")[FEATURE_COLS].agg(["mean", "std"])

def get_crop_profile(crop):
    """Return {feature: {mean, std}} for a given crop label."""
    if crop not in _crop_stats.index:
        return None
    row = _crop_stats.loc[crop]
    return {
        feat: {"mean": float(row[(feat, "mean")]), "std": float(row[(feat, "std")]) or 1.0}
        for feat in FEATURE_COLS
    }

FEATURE_LABELS = {
    "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
    "temperature": "Temperature", "humidity": "Humidity",
    "ph": "Soil pH", "rainfall": "Rainfall",
}

# Simple agronomic rotation logic: after harvesting a crop from one family,
# suggest a crop from a complementary family for the next season (e.g.
# legumes after heavy nitrogen-feeding cereals to help replenish the soil).
CROP_FAMILY = {
    "rice": "cereal", "maize": "cereal",
    "chickpea": "legume", "kidneybeans": "legume", "pigeonpeas": "legume",
    "mothbeans": "legume", "mungbean": "legume", "blackgram": "legume", "lentil": "legume",
    "pomegranate": "fruit", "banana": "fruit", "mango": "fruit", "grapes": "fruit",
    "watermelon": "fruit", "muskmelon": "fruit", "apple": "fruit", "orange": "fruit", "papaya": "fruit",
    "cotton": "cash", "jute": "cash", "coconut": "cash", "coffee": "cash",
}
LEGUME_CROPS = [c for c, fam in CROP_FAMILY.items() if fam == "legume"]
ROTATION_ADVICE = {
    "cereal": {
        "next_family": "legume",
        "reason": "Cereals are heavy nitrogen feeders. Rotating into a legume next season helps fix nitrogen back into the soil naturally.",
    },
    "cash": {
        "next_family": "legume",
        "reason": "Cash crops like cotton and jute deplete soil nutrients quickly. A legume break crop restores fertility before the next high-demand crop.",
    },
    "fruit": {
        "next_family": None,
        "reason": "Most fruit crops are perennial or long-cycle, so rotation typically isn't applicable within the same season.",
    },
    "legume": {
        "next_family": "cereal",
        "reason": "After a nitrogen-fixing legume, the soil is well-suited for a nitrogen-hungry cereal.",
    },
}


def suggest_rotation(crop, ph_value):
    family = CROP_FAMILY.get(crop)
    if not family:
        return None
    advice = ROTATION_ADVICE.get(family)
    if not advice or not advice["next_family"]:
        return {"reason": advice["reason"] if advice else None, "suggestions": []}

    candidates = [c for c, fam in CROP_FAMILY.items() if fam == advice["next_family"]]
    # Prefer candidates whose typical pH range is close to the current soil pH
    scored = []
    for c in candidates:
        profile = get_crop_profile(c)
        if profile:
            scored.append((abs(profile["ph"]["mean"] - ph_value), c))
    scored.sort()
    suggestions = [c for _, c in scored[:3]] if scored else candidates[:3]
    return {"reason": advice["reason"], "suggestions": suggestions}


def fertilizer_suggestion(crop, N, P, K):
    """Compare the user's N-P-K to this crop's ideal mean and suggest gaps."""
    profile = get_crop_profile(crop)
    if not profile:
        return []
    tips = []
    for feat, given in [("N", N), ("P", P), ("K", K)]:
        ideal = profile[feat]["mean"]
        diff = ideal - given
        # Only flag meaningful gaps (>10% of the ideal value)
        if ideal > 0 and abs(diff) / ideal > 0.10:
            if diff > 0:
                tips.append(f"{FEATURE_LABELS[feat]} is low — consider adding about {diff:.0f} kg/ha to reach the typical level for {crop} (~{ideal:.0f}).")
            else:
                tips.append(f"{FEATURE_LABELS[feat]} is higher than typical for {crop} (~{ideal:.0f}) — no need to add more; excess can leach or harm the crop.")
    return tips


def explain_prediction(crop, feature_values):
    """
    Lightweight explainability: for each feature, compute how many standard
    deviations the user's value is from this crop's typical (training-data)
    mean. Small deviation = feature supports the prediction; large deviation
    = feature is atypical for this crop despite the model picking it.
    """
    profile = get_crop_profile(crop)
    if not profile:
        return []
    explanations = []
    for feat, value in zip(FEATURE_COLS, feature_values):
        mean = profile[feat]["mean"]
        std = profile[feat]["std"]
        z = abs(value - mean) / std if std else 0
        if z < 0.5:
            fit = "strong match"
        elif z < 1.2:
            fit = "good match"
        elif z < 2.0:
            fit = "borderline"
        else:
            fit = "unusual for this crop"
        explanations.append({
            "feature": FEATURE_LABELS[feat],
            "value": round(value, 1),
            "typical": round(mean, 1),
            "fit": fit,
            "z": round(z, 2),
        })
    # Sort so the most supportive (best-matching) features show first
    explanations.sort(key=lambda e: e["z"])
    return explanations


# ---------------------------------------------------------------------------
# Model comparison: train RandomForest and HistGradientBoosting (scikit-learn's
# fast histogram-based boosting - same core idea as XGBoost, no extra heavy
# dependency) on the same split used to evaluate the existing neural network,
# so accuracy/F1/confusion matrices are directly comparable. This runs once
# at startup (training takes <2 seconds on this dataset size) rather than
# being baked-in/static.
# ---------------------------------------------------------------------------
def build_model_comparison():
    X = df.drop("label", axis=1).values
    y = df["label"].values
    y_enc = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    hgb = HistGradientBoostingClassifier(random_state=42)
    hgb.fit(X_train, y_train)
    hgb_pred = hgb.predict(X_test)

    X_test_scaled = scaler.transform(X_test)
    nn_probs = nn_predict(X_test_scaled)
    nn_pred = np.argmax(nn_probs, axis=1)

    class_names = label_encoder.classes_

    def metrics_for(y_true, y_pred):
        return {
            "accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
            "precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
            "recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
            "f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        }

    results = {
        "Neural Network": {"metrics": metrics_for(y_test, nn_pred), "cm": confusion_matrix(y_test, nn_pred)},
        "Random Forest": {"metrics": metrics_for(y_test, rf_pred), "cm": confusion_matrix(y_test, rf_pred)},
        "Gradient Boosting": {"metrics": metrics_for(y_test, hgb_pred), "cm": confusion_matrix(y_test, hgb_pred)},
    }
    return results, class_names


MODEL_COMPARISON, COMPARISON_CLASS_NAMES = build_model_comparison()

# ---------------------------------------------------------------------------
# Static reference data (same as the original Streamlit app)
# ---------------------------------------------------------------------------
district_data = {
    "Ranchi": {"lat": 23.3441, "lon": 85.3096, "N": 50, "P": 30, "K": 40, "ph": 6.2, "organic_carbon": 1.2},
    "Hazaribagh": {"lat": 23.9950, "lon": 85.3600, "N": 48, "P": 28, "K": 38, "ph": 6.1, "organic_carbon": 1.1},
    "Ramgarh": {"lat": 23.6303, "lon": 85.5158, "N": 52, "P": 32, "K": 42, "ph": 6.3, "organic_carbon": 1.3},
    "Koderma": {"lat": 24.4684, "lon": 85.5947, "N": 46, "P": 26, "K": 36, "ph": 6.0, "organic_carbon": 1.0},
    "Chatra": {"lat": 24.2071, "lon": 84.8719, "N": 44, "P": 25, "K": 35, "ph": 5.9, "organic_carbon": 0.9},
    "Garhwa": {"lat": 24.1635, "lon": 83.8074, "N": 42, "P": 24, "K": 33, "ph": 5.8, "organic_carbon": 0.8},
    "Palamu": {"lat": 24.0396, "lon": 83.9709, "N": 43, "P": 25, "K": 34, "ph": 5.9, "organic_carbon": 0.9},
    "Latehar": {"lat": 23.7444, "lon": 84.5041, "N": 45, "P": 27, "K": 37, "ph": 6.0, "organic_carbon": 1.0},
    "Jamshedpur": {"lat": 22.8056, "lon": 86.2029, "N": 55, "P": 35, "K": 45, "ph": 6.5, "organic_carbon": 1.4},
    "Seraikela-Kharsawan": {"lat": 22.7022, "lon": 85.9606, "N": 53, "P": 33, "K": 43, "ph": 6.4, "organic_carbon": 1.3},
    "Chaibasa": {"lat": 22.5561, "lon": 85.8072, "N": 51, "P": 31, "K": 41, "ph": 6.3, "organic_carbon": 1.2},
    "Dhanbad": {"lat": 23.7956, "lon": 86.4304, "N": 54, "P": 34, "K": 44, "ph": 6.4, "organic_carbon": 1.3},
    "Bokaro": {"lat": 23.6693, "lon": 86.1511, "N": 52, "P": 32, "K": 42, "ph": 6.3, "organic_carbon": 1.2},
    "Giridih": {"lat": 24.1914, "lon": 86.3082, "N": 49, "P": 29, "K": 39, "ph": 6.2, "organic_carbon": 1.1},
    "Deoghar": {"lat": 24.4846, "lon": 86.6947, "N": 47, "P": 28, "K": 38, "ph": 6.1, "organic_carbon": 1.0},
    "Dumka": {"lat": 24.2676, "lon": 87.2497, "N": 45, "P": 26, "K": 36, "ph": 6.0, "organic_carbon": 1.0},
    "Jamtara": {"lat": 23.9628, "lon": 86.8026, "N": 48, "P": 28, "K": 38, "ph": 6.1, "organic_carbon": 1.1},
    "Pakur": {"lat": 24.6339, "lon": 87.8497, "N": 43, "P": 25, "K": 34, "ph": 5.9, "organic_carbon": 0.9},
    "Godda": {"lat": 24.8267, "lon": 87.2142, "N": 44, "P": 26, "K": 35, "ph": 5.9, "organic_carbon": 0.9},
    "Sahibganj": {"lat": 25.2381, "lon": 87.6476, "N": 46, "P": 27, "K": 36, "ph": 6.0, "organic_carbon": 1.0},
    "Simdega": {"lat": 22.6170, "lon": 84.5170, "N": 49, "P": 29, "K": 39, "ph": 6.2, "organic_carbon": 1.1},
    "Gumla": {"lat": 23.0434, "lon": 84.5423, "N": 47, "P": 28, "K": 37, "ph": 6.1, "organic_carbon": 1.0},
    "Khunti": {"lat": 23.0737, "lon": 85.2789, "N": 48, "P": 29, "K": 38, "ph": 6.1, "organic_carbon": 1.1},
    "Lohardaga": {"lat": 23.4336, "lon": 84.6827, "N": 46, "P": 27, "K": 37, "ph": 6.0, "organic_carbon": 1.0},
}

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
    "coffee": {"cost": 55000, "revenue": 95000, "season": "Year-round", "water": "High", "sustainability": 75},
}

crop_images = {c: f"images/{c}.jpg" for c in crop_economics.keys()}

CLIMATE_ZONES = {
    "Tropical": ["rice", "banana", "coconut", "mango", "papaya", "coffee"],
    "Subtropical": ["maize", "cotton", "orange", "grapes", "pomegranate"],
    "Temperate": ["apple", "chickpea", "lentil"],
    "Arid": ["cotton", "millet", "mothbeans"],
    "Semi-Arid": ["maize", "cotton", "pigeonpeas", "chickpea"],
}

GROWING_SEASONS = {
    "Kharif": {"months": "June-November", "crops": ["rice", "cotton", "maize", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "jute"]},
    "Rabi": {"months": "November-April", "crops": ["chickpea", "lentil", "wheat", "barley"]},
    "Summer": {"months": "March-June", "crops": ["watermelon", "muskmelon", "mango"]},
    "Year-round": {"months": "All seasons", "crops": ["banana", "papaya", "pomegranate", "coconut", "coffee"]},
}

SOIL_REQUIREMENTS = {
    "rice": {"ph_range": (5.5, 6.5), "soil_type": "Clay/Loamy", "drainage": "Poor (waterlogged)"},
    "wheat": {"ph_range": (6.0, 7.5), "soil_type": "Loamy", "drainage": "Good"},
    "maize": {"ph_range": (6.0, 7.0), "soil_type": "Well-drained loamy", "drainage": "Good"},
    "cotton": {"ph_range": (5.8, 8.0), "soil_type": "Black cotton soil", "drainage": "Good"},
    "chickpea": {"ph_range": (6.0, 7.5), "soil_type": "Clay loam", "drainage": "Good"},
}

crop_categories = {
    "All": list(crop_images.keys()),
    "Cereals": ["rice", "maize"],
    "Pulses": ["chickpea", "kidneybeans", "pigeonpeas", "mothbeans", "mungbean", "blackgram", "lentil"],
    "Fruits": ["pomegranate", "banana", "mango", "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya"],
    "Cash Crops": ["cotton", "jute", "coconut", "coffee"],
}

# Simple in-memory weather cache (mirrors @st.cache_data(ttl=3600))
_weather_cache = {}


def get_weather(lat, lon):
    key = (lat, lon)
    cached = _weather_cache.get(key)
    if cached and (datetime.datetime.now() - cached["time"]).total_seconds() < 3600:
        return cached["data"]
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation"
            f"&forecast_days=1"
        )
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        temp = data["hourly"]["temperature_2m"][0]
        humidity = data["hourly"]["relative_humidity_2m"][0]
        rainfall = data["hourly"]["precipitation"][0]
        result = (temp, humidity, rainfall)
        _weather_cache[key] = {"time": datetime.datetime.now(), "data": result}
        return result
    except Exception:
        return None, None, None


def check_seasonal_feasibility(crop):
    current_month = datetime.datetime.now().month
    if 6 <= current_month <= 11:
        current_season = "Kharif"
    elif current_month == 12 or 1 <= current_month <= 4:
        current_season = "Rabi"
    else:
        current_season = "Summer"

    crop_season = crop_economics.get(crop, {}).get("season", "Unknown")

    if crop_season == "Year-round":
        return True, "Can be grown year-round"
    elif crop_season == current_season:
        return True, f"Perfect timing for {current_season} season"
    else:
        return False, f"Best season: {crop_season} (Current: {current_season})"


def check_climate_suitability(crop, climate_zone):
    suitable_zones = [zone for zone, crops in CLIMATE_ZONES.items() if crop in crops]
    if climate_zone in suitable_zones:
        return True, f"Suitable for {climate_zone} climate"
    elif suitable_zones:
        return False, f"Better suited for: {', '.join(suitable_zones)}"
    else:
        return True, "Climate data not available"


def validate_soil_conditions(crop, ph_value):
    if crop not in SOIL_REQUIREMENTS:
        return True, "Soil requirements data not available"
    req = SOIL_REQUIREMENTS[crop]
    ph_min, ph_max = req["ph_range"]
    if ph_min <= ph_value <= ph_max:
        return True, f"pH {ph_value} is optimal (Range: {ph_min}-{ph_max})"
    else:
        return False, f"pH {ph_value} outside optimal range {ph_min}-{ph_max}"


def calculate_enhanced_score(crop, budget, climate_zone, ph_value):
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
        (1.0 if seasonal_suitable else 0.7)
        * (1.0 if climate_suitable else 0.6)
        * (1.0 if soil_suitable else 0.8)
        * (1.0 if budget_suitable else 0.5)
    )

    base_score = roi * 0.4 + sustainability * 0.6
    return base_score * context_multiplier


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    districts = list(district_data.keys())
    climate_zones = list(CLIMATE_ZONES.keys())
    result = None

    if request.method == "POST":
        district = request.form.get("district", "--Manual Input--")
        climate_zone = request.form.get("climate_zone", "Tropical")
        budget = float(request.form.get("budget", 30000))

        if district != "--Manual Input--" and district in district_data:
            d = district_data[district]
            temp, humidity, rainfall = get_weather(d["lat"], d["lon"])
            if temp is None:
                temp, humidity, rainfall = 25.0, 60.0, 100.0  # fallback if weather API fails
            N, P, K, ph = d["N"], d["P"], d["K"], d["ph"]
        else:
            district = "Manual Input"
            N = float(request.form.get("N", 50))
            P = float(request.form.get("P", 50))
            K = float(request.form.get("K", 50))
            ph = float(request.form.get("ph", 6.5))
            temp = float(request.form.get("temp", 25.0))
            humidity = float(request.form.get("humidity", 50.0))
            rainfall = float(request.form.get("rainfall", 100.0))

        features = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)
        prediction = nn_predict(features_scaled)[0]  # softmax probabilities
        predicted_class = int(np.argmax(prediction))
        recommended_crop = label_encoder.inverse_transform([predicted_class])[0]
        confidence = float(prediction[predicted_class]) * 100

        # Top-3 AI predictions with confidence, not just the single top label
        top3_idx = np.argsort(prediction)[::-1][:3]
        top3_predictions = [
            {
                "crop": label_encoder.inverse_transform([i])[0],
                "confidence": round(float(prediction[i]) * 100, 1),
            }
            for i in top3_idx
        ]

        explanation = explain_prediction(recommended_crop, [N, P, K, temp, humidity, ph, rainfall])
        fertilizer_tips = fertilizer_suggestion(recommended_crop, N, P, K)
        rotation = suggest_rotation(recommended_crop, ph)

        enhanced_scores = []
        for crop in crop_economics.keys():
            score = calculate_enhanced_score(crop, budget, climate_zone, ph)
            if score > 0:
                enhanced_scores.append({"crop": crop, "score": score, "economics": crop_economics[crop]})
        enhanced_scores.sort(key=lambda x: x["score"], reverse=True)
        top_crops = enhanced_scores[:5]

        for c in top_crops:
            crop = c["crop"]
            econ = c["economics"]
            seasonal_ok, seasonal_msg = check_seasonal_feasibility(crop)
            climate_ok, climate_msg = check_climate_suitability(crop, climate_zone)
            soil_ok, soil_msg = validate_soil_conditions(crop, ph)
            roi = (econ["revenue"] - econ["cost"]) / econ["cost"] * 100
            c.update({
                "roi": round(roi, 1),
                "profit": econ["revenue"] - econ["cost"],
                "seasonal_msg": seasonal_msg,
                "climate_msg": climate_msg,
                "soil_msg": soil_msg,
                "budget_status": "Within budget" if econ["cost"] <= budget else "Over budget",
            })

        ai_in_top = any(c["crop"] == recommended_crop for c in top_crops)

        risk_factors = []
        if budget < 25000:
            risk_factors.append("Low budget may limit high-value crop options")
        if ph < 5.5 or ph > 8.0:
            risk_factors.append("Extreme pH levels may affect crop growth")
        if humidity > 90:
            risk_factors.append("Very high humidity increases disease risk")
        if temp and temp > 40:
            risk_factors.append("High temperature may stress crops")

        chart_b64 = None
        if len(top_crops) >= 3:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            crops_names = [c["crop"].capitalize() for c in top_crops[:3]]
            rois = [c["roi"] for c in top_crops[:3]]
            ax1.bar(crops_names, rois, color=["#4CAF50", "#2196F3", "#FFC107"])
            ax1.set_title("ROI Comparison (%)")
            ax1.set_ylabel("ROI (%)")

            costs = [c["economics"]["cost"] / 1000 for c in top_crops[:3]]
            sustainability = [c["economics"]["sustainability"] for c in top_crops[:3]]
            colors = ["#4CAF50", "#2196F3", "#FFC107"]
            for i, (name, cost, sust) in enumerate(zip(crops_names, costs, sustainability)):
                ax2.scatter(cost, sust, s=200, c=colors[i], alpha=0.7, label=name)
            ax2.set_xlabel("Cost (Rs thousands)")
            ax2.set_ylabel("Sustainability Score")
            ax2.set_title("Cost vs Sustainability")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            chart_b64 = fig_to_base64(fig)

        result = {
            "district": district,
            "recommended_crop": recommended_crop,
            "confidence": round(confidence, 1),
            "top3_predictions": top3_predictions,
            "explanation": explanation,
            "fertilizer_tips": fertilizer_tips,
            "rotation": rotation,
            "ai_in_top": ai_in_top,
            "weather": {"temp": temp, "humidity": humidity, "rainfall": rainfall},
            "soil": {"N": N, "P": P, "K": K, "ph": ph},
            "top_crops": top_crops,
            "risk_factors": risk_factors,
            "chart": chart_b64,
        }

    return render_template(
        "recommend.html",
        districts=districts,
        climate_zones=climate_zones,
        result=result,
    )


@app.route("/insights")
def insights():
    feature = request.args.get("feature", "N")
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    if feature not in features:
        feature = "N"

    # crop distribution bar chart
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    df["label"].value_counts().plot(kind="bar", ax=ax1, color="seagreen")
    ax1.set_title("Crop Distribution")
    crop_dist_b64 = fig_to_base64(fig1)

    # feature distribution
    fig2, ax2 = plt.subplots()
    sns.histplot(df[feature], kde=True, bins=30, ax=ax2, color="green")
    ax2.set_title(f"Distribution of {feature}")
    feature_dist_b64 = fig_to_base64(fig2)

    # correlation heatmap
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", ax=ax3)
    heatmap_b64 = fig_to_base64(fig3)

    # seasonal pie chart
    fig4, ax4 = plt.subplots()
    seasonal_counts = {v["months"]: len(v["crops"]) for v in GROWING_SEASONS.values()}
    ax4.pie(seasonal_counts.values(), labels=seasonal_counts.keys(), autopct="%1.1f%%")
    ax4.set_title("Crops by Growing Season")
    seasonal_b64 = fig_to_base64(fig4)

    raw_table = df.head(20).to_html(classes="data-table", index=False)

    return render_template(
        "insights.html",
        features=features,
        selected_feature=feature,
        crop_dist_chart=crop_dist_b64,
        feature_dist_chart=feature_dist_b64,
        heatmap_chart=heatmap_b64,
        seasonal_chart=seasonal_b64,
        raw_table=raw_table,
    )


@app.route("/gallery")
def gallery():
    category = request.args.get("category", "All")
    search = request.args.get("search", "").strip().lower()

    if category not in crop_categories:
        category = "All"

    selected_crops = crop_categories[category]
    filtered_crops = [c for c in selected_crops if search in c.lower()] if search else selected_crops

    crops_info = []
    for crop in filtered_crops:
        econ = crop_economics.get(crop, {})
        roi = None
        if econ:
            roi = round((econ["revenue"] - econ["cost"]) / econ["cost"] * 100, 1)
        crops_info.append({
            "name": crop,
            "image": crop_images.get(crop),
            "season": econ.get("season"),
            "roi": roi,
            "water": econ.get("water"),
            "sustainability": econ.get("sustainability"),
        })

    return render_template(
        "gallery.html",
        categories=list(crop_categories.keys()),
        selected_category=category,
        search=search,
        crops=crops_info,
    )


@app.route("/model-comparison")
def model_comparison():
    selected_model = request.args.get("model", "Neural Network")
    if selected_model not in MODEL_COMPARISON:
        selected_model = "Neural Network"

    model_names = list(MODEL_COMPARISON.keys())

    # Bar chart comparing accuracy / precision / recall / F1 across models
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    metric_names = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(metric_names))
    width = 0.25
    colors = ["#4f7a4a", "#d99a3d", "#c1652f"]
    for i, name in enumerate(model_names):
        values = [MODEL_COMPARISON[name]["metrics"][m] for m in metric_names]
        ax1.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)])
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(["Accuracy", "Precision", "Recall", "F1"])
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Model Comparison on the Same Test Split")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    comparison_chart = fig_to_base64(fig1)

    # Confusion matrix for the selected model
    cm = MODEL_COMPARISON[selected_model]["cm"]
    fig2, ax2 = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="YlGnBu", ax=ax2,
        xticklabels=COMPARISON_CLASS_NAMES, yticklabels=COMPARISON_CLASS_NAMES,
        cbar=False,
    )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title(f"Confusion Matrix — {selected_model}")
    plt.setp(ax2.get_xticklabels(), rotation=90)
    plt.setp(ax2.get_yticklabels(), rotation=0)
    confusion_chart = fig_to_base64(fig2)

    return render_template(
        "model_comparison.html",
        model_names=model_names,
        selected_model=selected_model,
        results=MODEL_COMPARISON,
        comparison_chart=comparison_chart,
        confusion_chart=confusion_chart,
    )


@app.route("/soil-detector", methods=["GET", "POST"])
def soil_detector():
    result = None
    error = None

    if request.method == "POST":
        file = request.files.get("soil_image")
        if not file or file.filename == "":
            error = "Please choose an image to upload."
        else:
            try:
                pil_image = Image.open(file.stream)
                label, confidence, ranked = soil_predict(pil_image)
                guidance = SOIL_CROP_GUIDANCE.get(label, {})

                # Re-encode the uploaded image as base64 so it can be shown
                # back to the user alongside the result without saving to disk.
                buf = io.BytesIO()
                pil_image.convert("RGB").save(buf, format="JPEG")
                buf.seek(0)
                image_b64 = base64.b64encode(buf.read()).decode("utf-8")

                result = {
                    "label": label,
                    "label_display": label.replace("_", " "),
                    "confidence": confidence,
                    "ranked": ranked,
                    "guidance": guidance,
                    "image_b64": image_b64,
                    "low_confidence_class": label in SOIL_LOW_CONFIDENCE_CLASSES,
                }
            except Exception:
                error = "Couldn't read that file as an image. Please upload a JPG or PNG photo."

    return render_template("soil_detector.html", result=result, error=error)


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
