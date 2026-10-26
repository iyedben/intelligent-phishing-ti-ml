# Intelligent Phishing Detection and Threat Intelligence Correlation Using Machine Learning and Explainable AI

## 1. Project Overview

### Project Title

**Intelligent Phishing Detection and Threat Intelligence Correlation Using Machine Learning and Explainable AI**

### Short Name

**IPDTI-ML**

### Project Category

- Cybersecurity
- Machine Learning
- Threat Intelligence
- Explainable Artificial Intelligence
- Web Application Development

### Project Level

Intermediate university cybersecurity and machine-learning project.

### Project Description

This project aims to develop a hybrid phishing-analysis platform that combines supervised machine learning, Explainable AI (XAI), and open-source Threat Intelligence (TI) to detect and contextualize suspicious URLs.

The machine-learning component predicts whether a URL exhibits characteristics associated with phishing. The Explainable AI component explains the factors contributing to the model's decision. The Threat Intelligence component correlates the analyzed URL against external security sources such as PhishTank, URLhaus, and OpenPhish.

A risk-correlation engine combines these sources of evidence into a transparent risk score and classification. The results are exposed through a RESTful FastAPI backend and a separate HTML/CSS/JavaScript frontend.

---

# 2. Problem Statement

Phishing attacks frequently use fraudulent URLs and websites designed to impersonate legitimate services and deceive users.

Traditional blacklist-based approaches have an important limitation: newly created phishing URLs may not yet be present in known threat-intelligence databases.

Machine-learning approaches can potentially identify previously unseen phishing URLs by analyzing characteristics of URLs, domains, and, where appropriate, webpage-derived features. However, machine-learning predictions alone do not necessarily provide sufficient contextual evidence or explain why a particular URL was classified as malicious.

Threat-intelligence feeds provide valuable information about known malicious indicators but cannot guarantee detection of previously unknown phishing URLs.

Therefore, this project proposes a hybrid approach combining:

**Machine Learning + Explainable AI + Threat Intelligence**

The objective is to provide not only a prediction, but also an understandable explanation and independent cybersecurity evidence.

---

# 3. Research Question

### Main Research Question

> How can machine-learning-based phishing detection be improved and contextualized through Explainable AI and correlation with open-source Threat Intelligence?

### Sub-Questions

1. How accurately can machine-learning algorithms classify phishing URLs?
2. Which URL and webpage characteristics contribute most to phishing predictions?
3. How can Explainable AI improve the interpretability of phishing classifications?
4. How can Threat Intelligence complement machine-learning predictions?
5. How does a hybrid ML + TI approach compare with ML-only detection?

---

# 4. Objectives

## 4.1 Main Objective

Develop a web-based phishing-analysis platform that combines machine learning, Explainable AI, and open-source Threat Intelligence to detect and assess suspicious URLs.

## 4.2 Specific Objectives

### O1 — Dataset Preparation

Acquire, inspect, clean, and prepare a publicly available phishing dataset for supervised machine learning.

### O2 — Feature Engineering

Identify and extract relevant URL, domain, security, and webpage characteristics.

### O3 — Machine Learning

Train and compare multiple supervised classification algorithms.

### O4 — Model Evaluation

Evaluate models using accuracy, precision, recall, F1-score, specificity, ROC-AUC, and confusion matrices.

### O5 — Explainable AI

Use SHAP-based techniques to provide global and local explanations for model predictions.

### O6 — Threat Intelligence

Integrate free/open-source phishing and malicious-URL intelligence sources.

### O7 — Correlation

Correlate machine-learning predictions with Threat Intelligence evidence.

### O8 — Risk Assessment

Develop a transparent risk-scoring engine combining multiple sources of evidence.

### O9 — Application Development

Develop a web application allowing a security analyst to submit a URL and obtain an analysis.

### O10 — Testing

Evaluate the system using functional, ML, integration, and security-oriented tests.

---

# 5. Scope

## 5.1 Included in the Core Scope

- Phishing URL classification
- URL feature extraction
- Supervised machine learning
- Model comparison
- Explainable AI using SHAP
- Threat Intelligence correlation
- PhishTank integration
- URLhaus integration
- OpenPhish integration where technically available
- Risk scoring
- REST API
- Web dashboard
- SQLite-based optional analysis history
- Unit and integration testing
- Security considerations
- Technical documentation

## 5.2 Out of Scope for the Initial Version

The following features are considered future extensions:

- Automatic browsing of arbitrary submitted URLs
- Full webpage sandboxing
- Browser extension
- Automatic URL blocking
- Automated firewall changes
- SIEM integration
- Automated incident response
- Deep-learning-based detection
- Large-scale distributed deployment

These may be considered after the core system is completed.

---

# 6. Proposed Architecture

The application follows a client-server architecture.

```text
                         USER / SECURITY ANALYST
                                  |
                                  v
                    +-------------------------+
                    |       FRONTEND          |
                    |                         |
                    | HTML / CSS / JavaScript |
                    +------------+------------+
                                 |
                           HTTP / JSON
                                 |
                                 v
                    +-------------------------+
                    |        FASTAPI          |
                    |        BACKEND          |
                    +------------+------------+
                                 |
             +-------------------+-------------------+
             |                   |                   |
             v                   v                   v
     +---------------+   +---------------+   +---------------+
     |    Feature    |   |      ML       |   |   Threat      |
     |   Extraction  |   |     Model     |   | Intelligence  |
     +-------+-------+   +-------+-------+   +-------+-------+
             |                   |                   |
             |                   v                   |
             |           +---------------+           |
             |           |      SHAP     |           |
             |           |      XAI      |           |
             |           +-------+-------+           |
             |                   |                   |
             +-------------------+-------------------+
                                 |
                                 v
                    +-------------------------+
                    |    RISK CORRELATION     |
                    |         ENGINE          |
                    +------------+------------+
                                 |
                                 v
                    +-------------------------+
                    |     FINAL ASSESSMENT    |
                    |                         |
                    | Risk Score              |
                    | Risk Level              |
                    | ML Probability          |
                    | XAI Explanation         |
                    | TI Evidence             |
                    +------------+------------+
                                 |
                                 v
                              FRONTEND
```

---

# 7. Application Architecture

## Frontend

The frontend will use:

- HTML5
- CSS3
- JavaScript

Responsibilities:

- URL submission
- Display analysis status
- Display ML prediction
- Display probability
- Display risk score
- Display Threat Intelligence results
- Display XAI explanations
- Display final risk classification

## Backend

The backend will use:

- Python
- FastAPI
- Uvicorn
- Pydantic

Responsibilities:

- API request validation
- URL analysis orchestration
- Feature extraction
- ML inference
- XAI generation
- Threat Intelligence queries
- Risk correlation
- Response generation

Flask is not part of the selected architecture.

---

# 8. Dataset

## Primary Dataset

**PhiUSIIL — Phishing URL Dataset**

Source:

UCI Machine Learning Repository

The dataset contains approximately:

- 235,795 instances
- 54 features
- 134,850 legitimate samples
- 100,945 phishing samples

The dataset contains both URL-related and webpage-derived characteristics.

## Dataset Usage

The dataset will be used for:

1. Exploratory data analysis
2. Feature engineering
3. Model training
4. Model validation
5. Model testing
6. Feature-importance analysis

---

# 9. Dataset Limitations

PhiUSIIL contains both URL-level and webpage-derived features.

Therefore, two experimental configurations will be considered.

### Configuration A — Enriched Detection

Uses the selected URL and webpage-derived features available in the dataset.

### Configuration B — URL-Only Detection

Uses only characteristics that can be derived from the URL without visiting the destination website.

This distinction allows the project to investigate early phishing detection versus enriched detection.

The URL-only configuration is particularly important because a real-world security system should avoid blindly opening an arbitrary submitted URL.

---

# 10. Feature Engineering

Features will be organized into several categories.

## URL Lexical Features

Examples:

- URL length
- Number of digits
- Number of letters
- Number of dots
- Number of hyphens
- Number of slashes
- Number of special characters
- Number of parameters
- Number of fragments

## Domain Features

Examples:

- Domain length
- Number of subdomains
- TLD characteristics
- Domain entropy
- IP-address-based hostname detection

## Security and Obfuscation Features

Examples:

- HTTPS presence
- IP address usage
- Encoded characters
- Suspicious characters
- URL entropy
- Obfuscation indicators

## Webpage Features

Where supported by the selected dataset configuration:

- Title characteristics
- Favicon presence
- Redirect behavior
- External links
- Forms
- Password fields
- External form submission
- URL/title similarity

The final feature list will be based on the actual dataset schema rather than artificially creating unsupported dataset fields.

---

# 11. Data Preprocessing

The preprocessing pipeline will include:

1. Dataset loading
2. Schema inspection
3. Target identification
4. Duplicate detection
5. Missing-value analysis
6. Invalid-record handling
7. Data-type normalization
8. Feature selection
9. Class-distribution analysis
10. Train/validation/test splitting

Potential duplicate and highly similar URL records will be investigated to reduce the risk of data leakage.

---

# 12. Data Splitting

The recommended initial split is:

- 70% training
- 15% validation
- 15% testing

Stratified splitting will be used to preserve class proportions.

Cross-validation will be used during model development and hyperparameter tuning.

The final test set will remain isolated until final evaluation.

---

# 13. Machine Learning Algorithms

The project will compare several supervised classification algorithms.

## 13.1 Logistic Regression

Purpose:

- Baseline
- Interpretable linear model
- Performance reference

## 13.2 Decision Tree

Purpose:

- Interpretable rule-based model
- Feature-decision visualization

## 13.3 Random Forest

Purpose:

- Strong tabular-data baseline
- Ensemble learning
- Feature importance analysis

## 13.4 XGBoost

Purpose:

- Advanced gradient-boosting model
- High-performance tabular classification
- Comparison against Random Forest

## Optional

Support Vector Machine may be evaluated if time and computational resources permit.

---

# 14. Machine Learning Evaluation

The models will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- ROC-AUC
- Confusion matrix

Special attention will be given to false negatives because a phishing URL classified as legitimate can represent a significant security risk.

---

# 15. Explainable AI

The project will use **SHAP — SHapley Additive exPlanations**.

Two levels of explanation will be implemented.

## Global Explainability

The system will identify the features that generally contribute most to model decisions.

Example:

```text
URL Length
URL Entropy
Domain Length
Number of Subdomains
Obfuscation Indicators
```

## Local Explainability

For a specific URL, the system will show the main factors contributing to the individual prediction.

Example:

```text
Prediction: PHISHING
Probability: 94.7%

Main contributing factors:
- High URL entropy
- Suspicious URL length
- Multiple subdomains
- Obfuscation indicators
```

---

# 16. Threat Intelligence

The system will integrate open-source/free Threat Intelligence sources where their current access mechanisms and usage terms permit.

## PhishTank

Used for known phishing URL correlation.

## URLhaus

Used for malicious URL and malware-distribution intelligence.

## OpenPhish

Used as an additional phishing intelligence source where access is available under its applicable terms.

The application will normalize results from the different sources into a common internal format.

Example:

```json
{
  "source": "phishtank",
  "match": true,
  "type": "phishing"
}
```

---

# 17. Threat Intelligence Correlation

The Threat Intelligence layer will not replace machine learning.

Instead, it will provide additional evidence.

```text
Machine Learning
       +
Explainable AI
       +
Threat Intelligence
       |
       v
Risk Correlation
```

The system will distinguish between:

- ML-only evidence
- Known TI evidence
- Combined evidence
- No external TI evidence

This allows the project to investigate whether TI improves contextualization of ML predictions.

---

# 18. Risk Scoring

A transparent risk score from 0 to 100 will be implemented.

An initial proposed design is:

```text
Machine Learning evidence       0–60
Threat Intelligence evidence   0–30
URL heuristic evidence         0–10
                                ----
Total                           0–100
```

Initial risk categories:

```text
0–24      LOW
25–49     MEDIUM
50–74     HIGH
75–100    CRITICAL
```

These thresholds are project-defined and will be documented as such. They are not presented as universal cybersecurity standards.

The scoring mechanism may be refined after experimental evaluation.

---

# 19. API Design

The FastAPI backend will expose REST endpoints.

## Health Check

```http
GET /api/health
```

## URL Analysis

```http
POST /api/analyze
```

Example request:

```json
{
  "url": "https://example.com/login"
}
```

Example response:

```json
{
  "prediction": "phishing",
  "ml_probability": 0.947,
  "risk_score": 92,
  "risk_level": "critical",
  "threat_intelligence": {},
  "explanation": []
}
```

## Model Information

```http
GET /api/model-info
```

Optional:

```http
GET /api/history
```

for analysis history stored in SQLite.

---

# 20. Security Considerations

The application will initially analyze URL characteristics without automatically browsing arbitrary user-submitted URLs.

The system should avoid directly performing unrestricted requests such as:

```python
requests.get(user_submitted_url)
```

because arbitrary URL fetching may introduce security risks including:

- SSRF
- malicious redirects
- resource exhaustion
- unintended downloads
- interaction with internal services

If webpage fetching is introduced later, it should be implemented as an isolated and sandboxed subsystem.

Additional protections should include:

- URL validation
- request size limits
- API error handling
- timeout controls
- rate limiting where appropriate
- secure configuration
- secrets stored through environment variables
- safe logging

---

# 21. Software Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.11/3.12 |
| Frontend | HTML5, CSS3, JavaScript |
| Backend | FastAPI |
| API server | Uvicorn |
| Data processing | Pandas, NumPy |
| ML | Scikit-learn |
| Advanced ML | XGBoost |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn |
| Threat Intelligence | PhishTank, URLhaus, OpenPhish |
| Database | SQLite |
| Model serialization | Joblib |
| Testing | Pytest |
| Version control | Git |
| Repository | GitHub |
| Optional containerization | Docker |

---

# 22. Development Environment

Recommended environment:

```text
Windows 11
Python 3.11 or 3.12
VS Code
Git
Python virtual environment
```

GPU acceleration is not required for the core project.

AWS cloud resources are not required for the core implementation.

---

# 23. Installation

Create the project:

```bash
mkdir phishing-ti-ml
cd phishing-ti-ml
```

Create the virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the core dependencies:

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn shap joblib
```

Install backend dependencies:

```bash
pip install fastapi uvicorn pydantic python-dotenv requests
```

Install testing tools:

```bash
pip install pytest httpx
```

Save dependencies:

```bash
pip freeze > requirements.txt
```

---

# 24. Repository Structure

```text
phishing-ti-ml/
│
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   └── feature_schema.json
│
├── backend/
│   ├── main.py
│   │
│   ├── api/
│   │   ├── routes.py
│   │   └── schemas.py
│   │
│   ├── services/
│   │   ├── analyzer.py
│   │   ├── feature_service.py
│   │   ├── ml_service.py
│   │   ├── xai_service.py
│   │   ├── ti_service.py
│   │   └── risk_service.py
│   │
│   ├── threat_intelligence/
│   │   ├── phishtank.py
│   │   ├── urlhaus.py
│   │   └── openphish.py
│   │
│   └── core/
│       ├── config.py
│       └── logging.py
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── ml/
│   ├── notebooks/
│   │   ├── 01_eda.ipynb
│   │   ├── 02_features.ipynb
│   │   ├── 03_training.ipynb
│   │   ├── 04_evaluation.ipynb
│   │   └── 05_xai.ipynb
│   │
│   └── scripts/
│       ├── train.py
│       └── evaluate.py
│
├── tests/
│   ├── test_features.py
│   ├── test_ml.py
│   ├── test_ti.py
│   ├── test_risk.py
│   └── test_api.py
│
├── docs/
│   └── PROJECT_SPECIFICATION.md
│
└── reports/
    └── figures/
```

---

# 25. Development Methodology

The implementation will follow these phases:

```text
1. Project definition
        ↓
2. Environment setup
        ↓
3. Dataset acquisition
        ↓
4. Exploratory data analysis
        ↓
5. Feature engineering
        ↓
6. Model training
        ↓
7. Model evaluation
        ↓
8. Explainable AI
        ↓
9. Threat Intelligence integration
        ↓
10. Risk correlation
        ↓
11. FastAPI development
        ↓
12. Frontend development
        ↓
13. Integration
        ↓
14. Testing
        ↓
15. Documentation
```

---

# 26. Experimental Plan

## Experiment 1 — Baseline

Train Logistic Regression and establish baseline performance.

## Experiment 2 — Model Comparison

Compare:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

## Experiment 3 — Feature Selection

Compare full selected features with a reduced feature set.

## Experiment 4 — URL-Only Detection

Evaluate detection using only features obtainable from the URL itself.

## Experiment 5 — Explainability

Analyze global and local SHAP explanations.

## Experiment 6 — Threat Intelligence

Evaluate known malicious URLs against available TI sources.

## Experiment 7 — Hybrid Detection

Compare:

```text
ML only
vs.
ML + Threat Intelligence
```

This is the primary experiment demonstrating the value of the hybrid approach.

---

# 27. Eight-Week Implementation Plan

## Week 1 — Research and Environment

Tasks:

- Finalize project scope
- Create GitHub repository
- Install Python and development tools
- Create virtual environment
- Install dependencies
- Create repository structure
- Study dataset documentation
- Study TI source documentation
- Define research questions

Deliverables:

- Project specification
- GitHub repository
- Development environment
- Initial README

---

## Week 2 — Dataset and EDA

Tasks:

- Acquire PhiUSIIL
- Inspect dataset schema
- Analyze labels
- Analyze class distribution
- Detect duplicates
- Analyze missing values
- Identify feature categories
- Identify URL-only features

Deliverables:

- Dataset analysis notebook
- EDA figures
- Dataset documentation

---

## Week 3 — Feature Engineering

Tasks:

- Implement URL normalization
- Implement URL parsing
- Implement URL feature extraction
- Build reusable feature-extraction functions
- Create preprocessing pipeline
- Add feature tests

Deliverables:

- Feature extraction module
- Preprocessing pipeline
- Unit tests

---

## Week 4 — Machine Learning

Tasks:

- Train baseline model
- Train Decision Tree
- Train Random Forest
- Train XGBoost
- Perform cross-validation
- Evaluate models
- Generate confusion matrices

Deliverables:

- Trained models
- Evaluation results
- Model comparison

---

## Week 5 — Optimization and XAI

Tasks:

- Tune selected models
- Perform feature selection
- Generate SHAP global explanations
- Generate SHAP local explanations
- Select final model

Deliverables:

- Final ML model
- SHAP analysis
- Feature-importance figures

---

## Week 6 — Threat Intelligence

Tasks:

- Implement PhishTank integration
- Implement URLhaus integration
- Implement OpenPhish integration where available
- Normalize TI results
- Handle API/feed failures
- Implement TI unit tests

Deliverables:

- Threat Intelligence modules
- TI integration tests

---

## Week 7 — Risk Engine and Application

Tasks:

- Implement risk scoring
- Build FastAPI backend
- Implement `/api/analyze`
- Implement `/api/health`
- Build frontend
- Connect frontend to FastAPI
- Display ML/XAI/TI results

Deliverables:

- Working web application
- REST API
- Risk engine

---

## Week 8 — Testing and Finalization

Tasks:

- Unit testing
- Integration testing
- API testing
- ML evaluation
- Security testing
- Error handling
- Final experiments
- Documentation
- Screenshots
- Presentation preparation

Deliverables:

- Final GitHub repository
- Final report
- Presentation
- Working demonstration

---

# 28. Project Deliverables

The final project will contain:

## Software

- Complete source code
- ML pipeline
- Trained model
- XAI implementation
- Threat Intelligence modules
- Risk engine
- FastAPI backend
- Frontend
- Tests

## Documentation

- README
- Project specification
- Architecture documentation
- Dataset documentation
- Methodology
- API documentation
- Threat Intelligence documentation

## Academic Deliverables

- Technical report
- Experimental results
- Figures and tables
- Presentation
- Demonstration

---

# 29. Success Criteria

The project will be considered successful if:

### Machine Learning

- Multiple models are trained and compared.
- Appropriate evaluation metrics are reported.
- The final model demonstrates reliable classification performance.
- Data leakage is investigated and minimized.

### Explainable AI

- Global explanations are generated.
- Individual predictions can be explained.
- Important contributing features are displayed.

### Threat Intelligence

- Multiple intelligence sources can be queried or processed.
- Known indicators can be correlated.
- TI failures do not crash the analysis pipeline.

### Application

- Users can submit a URL.
- The API validates the input.
- The system returns a structured analysis.
- The frontend displays the results clearly.

### Security

- The application does not automatically browse arbitrary URLs.
- Secrets are not hard-coded.
- Errors are handled safely.
- User input is validated.

---

# 30. Future Work

Potential future extensions include:

1. Sandboxed webpage analysis
2. Browser extension
3. Wazuh/SIEM integration
4. Automated alert generation
5. URL blocking
6. Real-time monitoring
7. Deep-learning models
8. Domain reputation analysis
9. DNS-based features
10. Isolated webpage rendering
11. Continuous model retraining
12. Production deployment

---

# 31. Expected Final Result

The final system should allow an analyst to enter a URL and receive:

```text
Prediction
    ↓
Phishing / Legitimate

ML Probability
    ↓
e.g. 94.7%

Explainability
    ↓
Why the model made the decision

Threat Intelligence
    ↓
Known phishing/malicious evidence

Risk Score
    ↓
e.g. 92/100

Risk Level
    ↓
CRITICAL
```

The project therefore moves beyond simple phishing classification and demonstrates an integrated cybersecurity workflow combining:

**Machine Learning + Explainable AI + Threat Intelligence + Risk Assessment + Web Application Development.**
