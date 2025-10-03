
# ⚡ AI-Powered Energy Theft & Leak Detection System

## 📌 Project Overview
This project is part of **Business Analysis 3.2 (AIBUY3A)**.
We developed an **AI solution** to detect **electricity theft** and **oil pipeline leaks** using machine learning and time-series analysis.

**Key Features:**
- Energy theft detection from smart meter data.
- Pipeline leak detection from flow/pressure data.
- Machine Learning models: Isolation Forest, Autoencoder, LSTM.
- Streamlit dashboard for real-time anomaly monitoring.
- FastAPI backend for model inference.

**Theme Alignment:** AI for Industrial Applications (Energy & Oil sector).

---

## 👥 Team Members
- Member 1 – Project Lead & Dashboard Developer
- Member 2 – Energy Data Wrangler
- Member 3 – Pipeline Data Wrangler
- Member 4 – Feature Engineer (Energy)
- Member 5 – Feature Engineer (Pipeline)
- Member 6 – ML Model Builder (Energy Theft)
- Member 7 – ML Model Builder (Pipeline Leak)

---

## 📂 Repository Structure



---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-repo-name>.git
cd <your-repo-namhttps://github.com/222351640-droid/energy_theft_and_oil_leak.git

3. Install Dependencies
pip install -r requirements.txt

Usage
Run Notebooks (Data Prep → Features → Models)
jupyter notebook notebooks/01_data_loading_and_eda.ipynb

Run FastAPI (Model API)
uvicorn src.deployment.app:app --reload


Visit: http://127.0.0.1:8000/docs

Run Dashboard
streamlit run src/dashboard/app.py

Contribution Workflow (VERY IMPORTANT)

To ensure proper collaboration and grading, everyone must use branches.
Follow these steps:

1. Create a New Branch
git checkout -b your-branch-name


👉 Example branch names:

energy-data-prep-member2

pipeline-data-prep-member3

features-energy-member4

model-pipeline-member7

dashboard-member1

2. Add & Commit Your Work
git add .
git commit -m "Added feature engineering for pipeline dataset"

3. Push Your Branch
git push origin your-branch-name

4. Open a Pull Request (PR)

Go to GitHub → Repo → "Compare & Pull Request".

Add a short description of your changes.

Assign the Project Lead as reviewer.

Wait for review/approval before merge.

📊 Evaluation Metrics

Precision, Recall, F1-Score

Confusion Matrix

Detection Latency (time-to-detect)

📑 Deliverables

Codebase (notebooks, scripts, models)

Poster (project summary)

Presentation (20 minutes + Q&A)

Final Report (problem, objectives, methods, results)

Grammarly Certificate
