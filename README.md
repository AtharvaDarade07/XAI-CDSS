# Explainable AI for Personalized Clinical Decision Support System

> An explainable, patient-centered Clinical Decision Support System (CDSS) that combines machine-learning-based diabetes risk prediction with SHAP/LIME explanations, explanation consistency analysis, secure APIs, and a clinician-oriented web dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![React](https://img.shields.io/badge/React-Frontend-61DAFB)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange)
![Explainable AI](https://img.shields.io/badge/XAI-SHAP%20%7C%20LIME-purple)
![Database](https://img.shields.io/badge/Database-SQLite%20%7C%20PostgreSQL--ready-4479A1)
![Tests](https://img.shields.io/badge/Testing-Backend%20%2B%20Frontend-success)

---

## 📌 Overview

Healthcare machine learning systems can provide useful predictions, but a prediction alone may not be sufficient for clinical decision support.

This project develops an **Explainable AI-based Clinical Decision Support System** for diabetes risk prediction using the **Pima Indians Diabetes Dataset**.

The system combines:

- Machine learning prediction
- Patient-specific explanations
- SHAP and LIME explainability
- Explanation consistency analysis
- Secure authentication and role-based access control
- Patient and clinical-record management
- Prediction and explanation persistence
- Clinician-oriented dashboard
- Reproducible evaluation and testing
- Human-in-the-loop decision support

The system is designed as a **research prototype** and is **not intended to provide autonomous medical diagnosis or treatment recommendations**.

---

# 🎯 Research Problem

Many healthcare AI systems focus primarily on prediction accuracy. However, clinicians may also need to understand:

- Why a prediction was produced
- Which patient features influenced the prediction
- Whether different explanation techniques provide consistent information
- How the explanation relates to the individual patient
- How predictions and explanations can be securely stored and traced

This project addresses these concerns by integrating **prediction, explainability, consistency analysis, secure backend services, and clinician-oriented presentation into a single CDSS workflow**.

---

# 🔬 Research Gap

Existing healthcare AI research has demonstrated effective disease-risk prediction and increasingly uses explainability techniques such as SHAP and LIME.

However, prediction, patient-specific explanation, explanation consistency, secure system integration, and clinician-oriented presentation are often addressed as separate components rather than as a unified clinical decision-support workflow.

### Proposed Contribution

This project integrates:

```text
Patient Data
     ↓
Data Preprocessing
     ↓
ML Prediction
     ↓
Risk Probability / Class
     ↓
SHAP + LIME
     ↓
Explanation Consistency
     ↓
Patient-Specific Explanation
     ↓
Clinician Dashboard
     ↓
Human / Clinician Final Decision


🧠 Key Features
1. Machine Learning Prediction
Multiple machine-learning algorithms were evaluated:
- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest
- XGBoost
Model selection is performed using 5-fold stratified cross-validation on the training partition, with ROC-AUC as the primary selection metric.
Selected Research Model
Logistic Regression — Selected v2
2. Explainable AI
The system provides local, patient-specific explanations using SHAP and LIME.
SHAP
SHAP explains how individual features contribute to the model output.
The system provides:
- Feature name
- Patient value
- SHAP contribution
- Direction of contribution
- Feature ranking
- Model baseline/output information
LIME
LIME provides a second local explanation for the same prediction.
The system records:
- Feature
- LIME weight
- Direction
- Feature ranking
- Explanation configuration
3. Explanation Consistency
SHAP and LIME explanations can differ.
Therefore, the project evaluates their agreement using:
- Top-3 feature overlap
- Top-5 feature overlap
- Spearman rank correlation
- Direction/sign agreement
- Shared important features
This helps evaluate whether the two explanation methods identify similar influential factors for a given prediction.
4. Personalized Explanation
The system presents explanations using the actual patient's feature values.
Example:
Patient Feature        Influence
-----------------------------------------
Age                    ↑ Positive
Glucose                ↑ Positive
Blood Pressure         ↓ Negative
BMI                    ↑ Positive
Pregnancies            ↓ Negative

The explanation describes model behavior and should not be interpreted as medical causation.
5. Clinician-Oriented Workflow
The system follows a human-in-the-loop workflow:
Clinician Login
      ↓
Select Patient
      ↓
Review Patient Record
      ↓
Run Prediction
      ↓
View Prediction
      ↓
View SHAP Explanation
      ↓
View LIME Explanation
      ↓
Check Explanation Consistency
      ↓
Clinician Reviews Evidence
      ↓
Final Clinical Decision

The system does not autonomously diagnose or prescribe treatment.
🏗️ System Architecture
┌───────────────────────────────┐
│       React Frontend          │
│                               │
│ Login / Dashboard             │
│ Patients / Records            │
│ Prediction / SHAP / LIME      │
│ Consistency Analysis          │
└───────────────┬───────────────┘
                │ REST API
                ↓
┌───────────────────────────────┐
│          FastAPI              │
│                               │
│ Authentication & RBAC         │
│ Patient APIs                  │
│ Prediction APIs               │
│ Explanation APIs              │
└───────────────┬───────────────┘
                │
        ┌───────┴────────┐
        ↓                ↓
┌──────────────┐  ┌──────────────┐
│ ML Service   │  │ XAI Service  │
│              │  │              │
│ Prediction   │  │ SHAP         │
│ Pipeline     │  │ LIME         │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └────────┬────────┘
                ↓
       ┌──────────────────┐
       │ SQLAlchemy ORM   │
       │                  │
       │ Patients         │
       │ Records          │
       │ Predictions      │
       │ Explanations     │
       │ Model Versions   │
       │ Audit Logs       │
       └──────────────────┘

Detailed architecture documentation:
📄 [`docs/final_architecture_and_workflow.md`](docs/final_architecture_and_workflow.md)
📂 Project Structure
xai-cdss/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── models/
│   │   ├── schemas/
│   │   ├── services/
│   │   └── main.py
│   │
│   └── alembic/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── context/
│   │   ├── pages/
│   │   ├── services/
│   │   └── types/
│   └── package.json
│
├── data/
│   ├── raw/
│   └── processed/
│
├── ml/
│   ├── pipeline.py
│   ├── train.py
│   ├── shap_explainer.py
│   ├── lime_explainer.py
│   ├── personalized_explanation.py
│   ├── registry/
│   └── results/
│
├── evaluation/
│   ├── figures/
│   ├── reports/
│   └── results/
│
├── docs/
│   └── final_architecture_and_workflow.md
│
├── scripts/
│   └── create_dev_admin.py
│
├── tests/
│
├── requirements.txt
├── docker-compose.yml
└── README.md

The development SQLite database is generated locally and should not be treated as a required source-controlled project artifact.

📊 Dataset
The project uses the:
Pima Indians Diabetes Dataset
Property	Value
Rows	768
Features	8
Target	1
Classes	0 / 1
Dataset Type	Binary Classification


Features
Pregnancies
Glucose
BloodPressure
SkinThickness
Insulin
BMI
DiabetesPedigreeFunction
Age

Target
Outcome

Where:
0 → Negative class
1 → Positive class

Data Preprocessing
Some medical measurements use 0 as an invalid/missing-coded value.
The pipeline treats zero as missing for:
Glucose
BloodPressure
SkinThickness
Insulin
BMI

Pregnancies = 0 remains a valid value.
Preprocessing is performed inside the ML pipeline to reduce the risk of data leakage.
🤖 Model Selection
Four models were evaluated using 5-fold stratified cross-validation on the training partition.
Model	Mean ROC-AUC	Accuracy	F1
Logistic Regression	0.8434	0.7882	0.6545
SVC	0.8333	0.7801	0.6485
Random Forest	0.8209	0.7687	0.6458
XGBoost	0.8023	0.7508	0.6223


Selected Model
Logistic Regression
Model artifact:
ml/registry/logistic_regression_selected_v2.joblib

Selection Criterion
Mean ROC-AUC on training-only 5-fold stratified cross-validation.

📈 Independent Test Evaluation
The selected model was evaluated on the held-out test partition.
Metric	Result
Accuracy	0.7078
Precision	0.6000
Recall	0.5000
F1	0.5455
ROC-AUC	0.8130
Brier Score	0.1754


Confusion Matrix
                 Predicted
               0          1
Actual  0      82         18
        1      27         27

ROC-AUC Confidence Interval
95% Bootstrap CI:
0.7396 – 0.8729

These results are dataset-specific and should not be interpreted as evidence of clinical effectiveness.
🔍 XAI Evaluation
For the selected model, SHAP and LIME were evaluated on a patient-specific example.
SHAP
The selected Logistic Regression model uses:
LinearExplainer

SHAP explanations are calculated for all eight input features.
LIME Configuration
Random seed: 42
Number of features: 8
Number of samples: 5000

Example Consistency Results
Metric	Result
Top-3 overlap	2 / 3
Top-5 overlap	4 / 5
Spearman correlation	0.6429
Direction agreement	7 / 8


These measurements represent case-level explanation consistency, not global XAI stability.

🔐 Security
The backend includes:
- JWT authentication
- Password hashing using bcrypt
- Role-based access control
- Admin and clinician roles
- Protected API endpoints
- Active-user validation
- Audit-log data model
- Environment-based configuration
Example Roles
Admin
Clinician

Security Note
This is a research prototype.
Production healthcare deployment would require additional controls including:
- HTTPS/TLS
- Strong secret management
- Token/session management
- Rate limiting
- Secure cookie strategy
- Infrastructure security
- Privacy/compliance controls
- Security auditing
- Clinical governance
🗄️ Database
The system uses SQLAlchemy with an SQLite development database and a PostgreSQL-compatible design.
Main Entities
Roles
Users
Patients
Patient Records
Predictions
Explanations
Model Versions
Audit Logs

Relationships
Roles
  │
  └── Users
        │
        └── Audit Logs

Patients
  │
  └── Patient Records
        │
        └── Predictions
              │
              ├── Explanations
              │
              └── Model Versions

Database migrations are managed using Alembic.
🖥️ Frontend
The frontend is built using:
- React
- TypeScript
- Vite
- React Router
- Axios
- Recharts
- Responsive CSS
Main Screens
Login
Dashboard
Patients
Patient Details
Clinical Record
Prediction
SHAP Explanation
LIME Explanation
Consistency Analysis

⚙️ Backend
The backend is built using:
- Python 3.11
- FastAPI
- SQLAlchemy
- Alembic
- Pydantic
- JWT
- Passlib / bcrypt
Important API Areas
/auth

/patients

/patients/{patient_id}/records

/patients/{patient_id}/records/{record_id}/predict

/predictions/{prediction_id}/shap

/predictions/{prediction_id}/lime

/predictions/{prediction_id}/consistency

🎬 Demo Workflow
The complete prototype workflow is:
Login
  ↓
Create / Select Patient
  ↓
Add Clinical Record
  ↓
Run Prediction
  ↓
View Prediction Result
  ↓
View SHAP Explanation
  ↓
View LIME Explanation
  ↓
View Explanation Consistency
  ↓
Review Patient-Specific Evidence
  ↓
Clinician Final Decision

📸 Screenshots
Add project screenshots here after uploading them to the repository:
## Dashboard

![Dashboard](docs/screenshots/dashboard.png)

## Prediction

![Prediction](docs/screenshots/prediction.png)

## SHAP Explanation

![SHAP Explanation](docs/screenshots/shap.png)

## LIME Explanation

![LIME Explanation](docs/screenshots/lime.png)

## Consistency Analysis

![Consistency Analysis](docs/screenshots/consistency.png)

🚀 Running the Project
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd xai-cdss

2. Create Python Environment
python -m venv .venv

Windows
.venv\Scripts\Activate.ps1

3. Install Backend Dependencies
pip install -r requirements.txt

4. Configure Environment Variables
Create a .env file.
Example:
CDSS_DATABASE_URL=sqlite:///./backend/cdss.db
CDSS_JWT_SECRET=change-this-in-development
CDSS_JWT_ALGORITHM=HS256
CDSS_ACCESS_TOKEN_EXPIRE_MINUTES=30

Never commit real secrets to GitHub.

5. Run Database Migrations
alembic upgrade head

6. Create Development Admin
python scripts/create_dev_admin.py

Use the credentials configured by the script for local development.
Never use development credentials in production.

7. Start FastAPI
From the project root:
uvicorn backend.app.main:app --host 127.0.0.1 --port 8002

API:
http://127.0.0.1:8002

Swagger documentation:
http://127.0.0.1:8002/docs

8. Start React Frontend
cd frontend
npm install
npm run dev

The frontend normally runs at:
http://127.0.0.1:5173

Configure the frontend API URL using:
VITE_API_BASE_URL=http://127.0.0.1:8002

🧪 Testing
Backend
pytest

The project includes automated backend tests covering areas such as:
- Authentication
- RBAC
- Patient APIs
- Clinical records
- Prediction APIs
- SHAP
- LIME
- Consistency analysis
- Integration behavior
Frontend
npm test

Frontend tests cover key UI and workflow components.
Production Build
npm run build

🔬 Evaluation
The project evaluates multiple dimensions rather than prediction accuracy alone.
Machine Learning
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- Brier score
- Calibration information
Explainability
- SHAP explanations
- LIME explanations
- Feature ranking
- Feature contribution direction
Explanation Consistency
- Top-3 overlap
- Top-5 overlap
- Spearman correlation
- Direction agreement
Software Quality
- Backend tests
- Frontend tests
- API integration testing
- Authentication testing
- Reproducibility checks
- Build verification
📚 Documentation
Detailed architecture and workflow documentation:
📄 [`Final Architecture & Workflow`](docs/final_architecture_and_workflow.md)
Additional project documentation:
docs/
evaluation/reports/
evaluation/results/
ml/results/

⚠️ Important Limitations
This project is an academic research prototype.
It should not be used as a standalone medical diagnostic system.
Current limitations include:
- Pima Indians Diabetes Dataset only
- Limited dataset size and population
- No external clinical validation
- No prospective clinical validation
- No formal clinician usability study
- Case-level XAI consistency rather than global stability
- Prototype-level security
- No direct hospital/EHR integration
- No autonomous treatment recommendation
- Dataset-specific model performance
- Further clinical governance would be required for deployment
🛡️ Responsible AI
The system follows a human-in-the-loop approach.
AI Predicts
     ↓
AI Explains
     ↓
AI Supports
     ↓
Clinician Reviews
     ↓
Human Makes Final Decision

The system does not:
- Diagnose autonomously
- Prescribe treatment
- Replace clinicians
- Claim causal relationships from SHAP/LIME
Model explanations describe how the model arrived at its output, not why a medical condition biologically occurred.
🧑‍⚕️ Intended Use
The prototype demonstrates how an AI-assisted clinical workflow could provide:
1. Patient information
2. ML-based prediction
3. Patient-specific explanation
4. Explanation comparison
5. Evidence for clinician review
The intended user is a clinician or authorized healthcare professional in a controlled research/prototype environment.
🧩 Research Contribution
The primary contribution of this project is not simply a diabetes classifier.
It is an integrated prototype combining:
Machine Learning
       +
Patient-Specific XAI
       +
SHAP
       +
LIME
       +
Explanation Consistency
       +
Secure Backend
       +
Database Persistence
       +
Clinician Dashboard
       +
Human-in-the-Loop

This creates a unified Explainable Clinical Decision Support workflow.
🔮 Future Work
Potential future research includes:
- External validation on independent datasets
- Prospective clinical evaluation
- Larger and more diverse patient populations
- Formal clinician usability studies
- Global XAI stability analysis
- Improved calibration
- Production-grade security
- EHR integration
- Model monitoring and drift detection
- Clinical governance and regulatory assessment
- Additional validated disease-specific modules
👨‍💻 Author
Atharva Darade
B.Tech Computer Engineering
Vishwakarma Institute of Technology, Pune
⚠️ Disclaimer
This software is developed for academic and research purposes only.
It is not a medical device and should not be used to diagnose, treat, cure, or prevent any disease.
All predictions and explanations generated by the system are experimental machine-learning outputs and require review by a qualified healthcare professional.
⭐ Project Status
Research Prototype — Active Development
Core Components
- [x] Dataset audit
- [x] ML preprocessing pipeline
- [x] Model comparison
- [x] Model selection
- [x] Independent evaluation
- [x] SHAP explanations
- [x] LIME explanations
- [x] Explanation consistency analysis
- [x] Personalized explanations
- [x] FastAPI backend
- [x] Authentication & RBAC
- [x] Database & migrations
- [x] React frontend
- [x] Prediction workflow
- [x] Clinician-oriented dashboard
- [x] Backend tests
- [x] Frontend tests
- [x] Architecture documentation
📌 Research Prototype Notice
The selected research model is Logistic Regression v2 according to the documented model-selection protocol.
Before presenting the live application as the final research system, ensure that the active runtime model is synchronized with the selected model artifact:
ml/registry/logistic_regression_selected_v2.joblib

The live runtime should report the corresponding Logistic Regression model version rather than the historical:
xgboost-baseline-v1

This ensures consistency between the research evaluation, documentation, and live demonstration.

### One thing I deliberately added at the end

I kept the **runtime synchronization note** because it is currently important for your project. Your research README says Logistic Regression v2 is selected, while your last live demo was still running XGBoost v1. That should be fixed before you call the GitHub repository your final version.

Also, the README itself currently has the strong research framing and responsible-AI positioning already present in your uploaded version. :chatgpt-content-reference{index="0"}

**After you paste this into `README.md`, the next thing I'd do is add 4–5 actual screenshots from your working dashboard.** That will make the GitHub repository look substantially more professional.