# Crop Recommendation - Flask App

## A note on deployment
I can't create accounts or push to hosting platforms on your behalf — no
network access to Render/Railway/PythonAnywhere and no way to hold a live
server myself. What's below makes deployment take you about 10 minutes:
the app is now lightweight enough to actually fit on a free tier, and
config files for Render are included.

## What changed to make this deployable
The earlier version used full TensorFlow (~600MB+ installed) and XGBoost
(which drags in a ~300MB unused CUDA dependency on Linux even for CPU-only
use). Free hosting tiers (Render free = 512MB RAM) would likely fail to
build or run out of memory with that footprint. Both models are now:

- **Crop recommendation model**: converted to **TensorFlow Lite**, served
  via `ai-edge-litert` (~20MB) instead of full TensorFlow. Verified to give
  identical predictions (max difference 1e-7) to the original Keras model.
- **Model comparison page**: now uses scikit-learn's `HistGradientBoostingClassifier`
  instead of XGBoost — same core idea (histogram-based gradient boosting),
  zero extra heavy dependencies, since scikit-learn is already required.

Total install footprint dropped from ~1GB+ to well under 200MB.

## Deploy to Render (recommended — free tier, reputed, straightforward)

1. Push this folder to a new GitHub repository (Render deploys from GitHub):
   ```bash
   cd crop_recommendation_flask
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to **render.com** → sign up / log in (GitHub login is easiest).
3. Click **New +** → **Web Service** → connect your GitHub repo.
4. Render will detect `render.yaml` automatically and pre-fill the settings
   (build command, start command, Python version). If it doesn't, set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
5. Choose the **Free** plan and click **Create Web Service**.
6. Wait for the build to finish (3-5 minutes) — Render gives you a live URL
   like `https://crop-recommendation-flask.onrender.com`.

Free tier note: Render's free web services spin down after 15 minutes of
inactivity and take ~30-60 seconds to wake up on the next request. This is
normal and expected on the free tier, not a bug.

## Deploy to Railway (alternative)
1. Push to GitHub as above.
2. Go to **railway.app** → New Project → Deploy from GitHub repo.
3. Railway auto-detects the `Procfile` and deploys. Add a generated domain
   under Settings → Networking once it's live.

## Deploy to PythonAnywhere (alternative, no GitHub required)
1. Sign up at **pythonanywhere.com** (free tier).
2. Upload the project as a zip via the Files tab, then unzip via a Bash console.
3. Go to the Web tab → Add a new web app → Flask → point it at `app.py`.
4. In a Bash console: `pip install --user -r requirements.txt`.
5. Reload the web app from the Web tab.

## Folder structure
```
crop_recommendation_flask/
├── app.py
├── requirements.txt
├── Procfile              <- for Render/Railway
├── render.yaml            <- Render one-click config
├── runtime.txt            <- pins Python version
├── nn_model.tflite         <- crop recommendation model (lightweight)
├── soil_model.tflite       <- soil photo classifier (lightweight)
├── soil_class_names.txt
├── scaler.pkl
├── label_encoder.pkl
├── Crop_recommendation.csv
├── templates/
└── static/
    └── images/             <- put your crop images here (rice.jpg, maize.jpg, etc.)
```

## Features in this version

1. **Crop recommendation** with confidence scoring, top-3 predictions,
   "why this crop?" explainability, fertilizer gap suggestions, and
   crop-rotation planning.
2. **Model comparison page** (`/model-comparison`) — Neural Network vs
   Random Forest vs Gradient Boosting, trained live on the same split,
   with macro-averaged precision/recall/F1 and confusion matrices.
3. **Soil photo detector** (`/soil-detector`) — upload a photo of soil and
   a CNN (trained from scratch on a public 1,188-image, 7-class dataset)
   identifies the soil type, with general crop guidance for that soil.
   Validation accuracy is ~70% overall; performance is honestly reported
   per class in the UI — Alluvial soil specifically is weaker (~25% recall)
   because only 51 training photos were available for it. This is flagged
   to the user rather than hidden.
4. Dataset insights, crop gallery, and the original economics/risk/seasonal
   scoring system.

### What I deliberately did NOT add: yield estimation
`Crop_recommendation.csv` has no yield/quantity column, so there's no
honest way to train a yield regressor from it. I didn't fabricate numbers
that would look like model output but aren't, since this could influence a
real planting decision.

## Run it locally first (recommended before deploying)

1. ```bash
   python3 -m venv venv
   source venv/bin/activate        # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Add your crop images into `static/images/` (filenames matching the crop
   name in lowercase, e.g. `static/images/rice.jpg`).
3. ```bash
   python app.py
   ```
4. Open **http://localhost:5000**

First startup trains the Random Forest / Gradient Boosting comparison
models — a few seconds, then the server is ready.

## Pages
- `/` — Home
- `/recommend` — AI crop recommendation + confidence, explainability,
  fertilizer tips, rotation advice
- `/soil-detector` — soil type detection from an uploaded photo
- `/insights` — Dataset insights (charts, correlation heatmap)
- `/model-comparison` — NN vs Random Forest vs Gradient Boosting
- `/gallery` — Crop gallery with economics
- `/about` — About the project

## Design notes
- Palette: forest green `#1f3d2b`, soil brown `#5b3a29`, wheat cream `#f4ecd8`,
  harvest gold `#d99a3d`, terracotta `#c1652f`.
- Type: Fraunces (serif, headings) + Inter (body) + JetBrains Mono (metric numbers).
- Charts render server-side as PNG and embed directly in the page.

## Notes
- District weather lookup uses the free Open-Meteo API and needs internet
  access; falls back to defaults if unreachable.
- The soil-image dataset used for training came from a public GitHub mirror
  (Phantom-fs/Soil-Classification-Dataset, originally sourced from Kaggle's
  "Soil Classification Image Data" by Omkar Gurav and related collections).
  If you redistribute this app publicly, credit that dataset source.
