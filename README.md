# ⚡ AI-Powered Energy Theft & Leak Detection System  

## 📌 Project Overview  
This project is part of **Business Analysis 3.2 (AIBUY3A)**.  
We developed an **AI solution** to detect **electricity theft** and **oil pipeline leaks** using machine learning and time-series analysis.  

**Key Features:**  
- Detect **energy theft** from smart meter data.  
- Detect **pipeline leaks** from flow and pressure data.  
- Machine learning models: **Isolation Forest, Autoencoder, LSTM**.  
- **Streamlit Dashboard** for real-time anomaly monitoring.  
- **FastAPI Backend** for model inference.  

**Theme Alignment:** AI for Industrial Applications (Energy & Oil sector).  

---

## 👥 Team Members  
- **Elton** – Project Lead & Dashboard Developer  
- **Shannon** – Energy Data Wrangler  
- **Amelia** – Pipeline Data Wrangler  
- **Rirhandzu** – Feature Engineer (Energy)  
- **Nathi** – Feature Engineer (Pipeline)  
- **Vinny** – ML Model Builder (Energy Theft & Pipeline Leak)  
- **Unity** – Documentation & Report Writer  

---

📂 Final Repo Structure
/data
   energyTdata.csv
   pipelineLdata.csv
   energy_data_transformed.csv
   pipeline_data_processed.csv
   energy_features.csv
   pipeline_features.csv
/notebooks
   01_data_loading_and_eda.ipynb
   02_pipeline_data_loading_and_eda.ipynb
   03_feature_engineering_energy.ipynb
   03_feature_engineering_pipeline.ipynb
   04_model_energy_theft.ipynb
   04_model_pipeline_leak.ipynb
/src
   preprocessing/
   models/
   deployment/
   dashboard/app.py
poster.pdf
presentation.pdf
report.docx
README.md


---

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

📊 Evaluation Metrics

Precision, Recall, F1-Score (model performance).

Confusion Matrix.

Detection Latency (speed of detection).

📑 Deliverables

Codebase (notebooks, scripts, models).

Poster (visual summary).

Presentation (20 min + Q&A).

Final Report (problem, objectives, methods, results, risks).

Grammarly Certificate (proof of writing quality).

🛡️ Academic Integrity

This project is original work by our team.
We confirm that all sources have been referenced and plagiarism has been avoided.

📅 Timeline

Day 1: Energy + Pipeline EDA (Shannon & Amelia)

Day 2: Feature Engineering (Rirhandzu & Nathi)

Day 3: ML Models + Dashboard (Vinny & Elton) → 70% complete for lecturer review

Day 4: Poster Draft (Shannon & Amelia)

Day 5: Slides Draft (Rirhandzu, Nathi, Vinny, Unity, Elton final polish)

Day 6–10: Report polishing, dashboard improvements, practice run

Day 10: Final Presentation
