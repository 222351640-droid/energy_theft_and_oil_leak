# ⚡ AI-Powered Energy Theft & Leak Detection System  

## 📌 Project Overview  
This repository contains an AI solution for detecting electricity theft and oil-pipeline leaks.
The system uses time-series feature engineering, unsupervised ML (IsolationForest / Autoencoders) and an LSTM option for sequences. A Streamlit dashboard and simple FastAPI inference endpoint are included to demonstrate the solution.

**Course:** Business Analysis 3.2 (AIBUY3A)
**Deliverables:** notebooks, trained models, Streamlit dashboard, poster, presentation, final report. 

**Theme Alignment:** AI for Industrial Applications (Energy & Oil sector).  

---

## 👥 Team Members  
- **Elton** – Project Lead & Dashboard Developer  
- **Shannon** – Energy Data Wrangler  
- **Amelia** – Pipeline Data Wrangler  
- **Rirhandzu** – Feature Engineer (Energy)  
- **Nathi** – Feature Engineer (Pipeline)  
- **Rhulani** – ML Model Builder (Energy Theft & Pipeline Leak)  
- **Unity** – Documentation & Report Writer  

---

📂 Repo layout (important files)
```bash
energy_theft_and_oil_leak/
├── data/
│   ├── energyTdata.csv
│   ├── pipelineLdata.csv
│   ├── energy_data_transformed.csv
│   ├── pipeline_data_processed.csv
│   ├── energy_features.csv
│   ├── pipeline_features.csv
│   ├── energy_model.joblib
│   ├── pipeline_model.joblib
│   ├── energy_scaler.joblib
│   └── pipeline_scaler.joblib
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_pipeline_data_loading_and_eda.ipynb
│   ├── 03_feature_engineering_energy.ipynb
│   ├── 03_feature_engineering_pipeline.ipynb
│   ├── 04_model_energy_theft.ipynb
│   └── 04_model_pipeline_leak.ipynb
├── src/
│   └── dashboard/
│       └── app.py           # Streamlit dashboard
├── poster.pdf
├── presentation.pdf
├── report.docx
├── requirements.txt
└── README.md
```

---
⚙️ Setup (run locally)

Recommended: use a virtual environment.
## ⚙️ Setup Instructions  

### 1. Clone the Repository  
```bash
git clone https://github.com/<your-repo-name>.git
cd <your-repo-name>
```



---

## 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```
---

## 3. Install Dependencies
```
pip install -r requirements.txt
```
---
🚀 How to run the project components
A. Streamlit dashboard

From project root:

cd src/dashboard
streamlit run app.py


Open the printed URL (usually http://localhost:8501) in your browser.

Important: The dashboard expects data and models to exist. Place model & feature CSV files in project-root/data/ (or update the DATA_DIR constant in app.py to match your layout).

B. FastAPI model demo (optional)

From project root:

uvicorn src.deployment.app:app --reload --port 8000


Visit: http://127.0.0.1:8000/docs to see API docs and test inference endpoints.

C. Notebooks (EDA → Features → Modeling)

Open the notebooks in this order:

notebooks/01_data_loading_and_eda.ipynb — energy ED

notebooks/02_pipeline_data_loading_and_eda.ipynb — pipeline EDA

notebooks/03_feature_engineering_energy.ipynb — feature engineering

notebooks/03_feature_engineering_pipeline.ipynb — feature engineering

notebooks/04_model_energy_theft.ipynb — models for energy

notebooks/04_model_pipeline_leak.ipynb — models for pipeline

Open with Jupyter:

jupyter notebook
# then open the notebooks via the browser UI

🧩 How the dashboard expects files (paths)

Default code assumes data/ is at project root (next to src/). If your dashboard uses a different path, edit PROJECT_ROOT or DATA_DIR in src/dashboard/app.py.

Expected filenames (place in energy_theft_and_oil_leak/data/):

energy_features.csv

pipeline_features.csv

energy_model.joblib

pipeline_model.joblib

energy_scaler.joblib

pipeline_scaler.joblib

🧪 Quick test (if models not present)

If you don't have saved *.joblib models yet, run the model notebooks to train and save them:

Ensure 04_model_energy_theft.ipynb saves energy_model.joblib and energy_scaler.joblib to data/.

Ensure 04_model_pipeline_leak.ipynb saves pipeline_model.joblib and pipeline_scaler.joblib to data/.

Example save code (used in the notebooks):

import joblib
joblib.dump(iso_forest, 'data/energy_model.joblib')
joblib.dump(scaler, 'data/energy_scaler.joblib')

🧭 Contribution workflow (required)

All team members must follow this workflow to keep the commit history clear.

Branching
# create branch (example)
git checkout -b features-energy-rirhandzu

Commit & push
git add .
git commit -m "Add feature engineering notebook for energy"
git push origin features-energy-rirhandzu

Open a Pull Request

Go to the repo on GitHub → Compare & pull request.

Add description: what changed, files added.

✅ What each member should commit (short checklist)

Elton: src/dashboard/app.py, dashboard screenshots, merge PRs.

Shannon: notebooks/01_data_loading_and_eda.ipynb, data/energy_data_transformed.csv.

Amelia: notebooks/02_pipeline_data_loading_and_eda.ipynb, data/pipeline_data_processed.csv.

Rirhandzu: notebooks/03_feature_engineering_energy.ipynb, data/energy_features.csv.

Nathi: notebooks/03_feature_engineering_pipeline.ipynb, data/pipeline_features.csv, poster.pdf.

Rhulani: notebooks/04_model_energy_theft.ipynb, notebooks/04_model_pipeline_leak.ipynb, data/*.joblib.

Unity: report.docx, presentation.pdf, final README polishing.

🗓 Timeline (compact)

Day 1: Data & EDA (Shannon & Amelia) — DONE

Day 2: Feature engineering (Rirhandzu & Nathi) — IN PROGRESS / DUE

Day 3: Modeling + Dashboard prototype (Vinny & Elton) — DUE (target 70% ready)

Day 4: Poster draft (Shannon & Amelia)

Day 5: Slide deck draft (Team)

Day 6–10: Final polishing, rehearsals, submission

Day 5–10: Report polishing, dashboard improvements, practice run

Day 10: Final Presentation

🔧 Troubleshooting (quick)

Streamlit can’t find data/models → check DATA_DIR in src/dashboard/app.py and ensure files are present.

Joblib load error → ensure models were saved with compatible scikit-learn versions.

Large notebook outputs/plots → reduce sample() sizes in dashboard (e.g., sample(2000)).
