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

The backend keeps the model output, explainability results, provider evidence, and correlation assessment separate. The current API does not calculate a separate numeric risk score. Results are exposed through a FastAPI backend and a separate HTML/CSS/JavaScript frontend.

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

The frontend uses:

- HTML5
- CSS3
- JavaScript

Responsibilities:

- URL submission
- Display analysis status
- Display ML prediction
- Display probability
- Display the backend correlation assessment
- Display Threat Intelligence results
- Display XAI explanations
- Display separate model and provider evidence

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

# 18. Proposed Risk Scoring (Not Implemented)

A transparent risk score from 0 to 100 was proposed for future work. It is not implemented in the current service or API.

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

Any future scoring mechanism requires separate evaluation and documentation. Current correlation uses the deterministic qualitative rules documented in `docs/CORRELATION.md`.

---

# 19. Current API

The FastAPI backend exposes the following endpoints:

## Health Check

```http
GET /health
```

## URL Analysis

```http
POST /api/v1/analyze
```

Example request:

```json
{
  "url": "https://example.com/login"
}
```

The response contains `url`, an `ml` object (`prediction`, `phishing_probability`, `threshold`, and `positive_class`), an `explanation` object for class 1, separate normalized `threat_intelligence` results, and the backend `correlation` assessment. See `/docs` for the exact Pydantic schema; no sample probability or risk score is asserted here.

`/docs` exposes the generated OpenAPI documentation. The API does not currently provide model-info or history endpoints, a separate risk score, authentication, or a database.

---

# 20. Security Considerations

The application analyzes URL strings without browsing arbitrary user-submitted destinations.

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

Current protections include:

- URL validation (HTTP/HTTPS only; 4,096-character maximum)
- sanitized API errors
- finite provider request timeouts
- environment-based credentials and ignored `.env`

Rate limiting and authentication are not implemented. External traffic is limited to the fixed, configured Threat Intelligence provider/feed endpoints; the submitted destination is never fetched or resolved.
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

Required environment:

```text
Windows 11
Python 3.12
VS Code
Git
Python virtual environment
```

GPU acceleration is not required for the core project.

AWS cloud resources are not required for the core implementation.

---

# 23. Installation

Use Python 3.12 for this project. In Windows PowerShell, create and activate the virtual environment, then install runtime and development dependencies:

```powershell
py -3.12 -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
python scripts/check_environment.py
```

---

# 24. Current Repository Structure

The implemented repository is organized as follows (locally acquired data and the ignored candidate model are not included in Git):

```text
intelligent-phishing-ti-ml/
├── docs/
│   ├── CORRELATION.md
│   ├── DATASET.md
│   ├── EDA_FINDINGS.md
│   ├── FEATURE_ENGINEERING.md
│   ├── ML_RESULTS.md
│   ├── PHASE9_VALIDATION.md
│   ├── THREAT_INTELLIGENCE.md
│   └── XAI_RESULTS.md
├── frontend/
│   ├── css/styles.css
│   ├── index.html
│   └── js/app.js
├── ml/
│   ├── experiment_config.json
│   └── notebooks/01_eda.ipynb, 02_ml_experiments.ipynb, 03_xai_shap.ipynb
├── scripts/
│   ├── check_environment.py
│   ├── inspect_dataset.py
│   ├── test_threat_intel.py
│   └── validate_features.py
├── src/
│   ├── api/
│   ├── features/
│   ├── ml/
│   ├── services/
│   ├── threat_intel/
│   └── xai/
├── tests/
├── data/raw/                         # locally acquired, Git-ignored
├── models/                           # local candidate artifact, Git-ignored
├── requirements.txt
├── requirements-dev.txt
└── .env.example
```

The current Random Forest candidate is expected at `models/url_only_random_forest_seed2026_candidate.joblib`. Model artifacts are ignored by Git, so a fresh checkout needs the validated artifact supplied separately before model-backed analysis can run.

---

# 25. Development Methodology

The completed work is organized into these phases:

1. Environment and official dataset acquisition
2. Dataset inspection and exploratory analysis
3. Deterministic offline URL-only feature extraction
4. Controlled group-aware ML experiments
5. SHAP explanations for the saved research candidate
6. Threat Intelligence providers and evidence correlation
7. FastAPI foundation and URL analysis service
8. Local HTML/CSS/JavaScript frontend
9. Application hardening and end-to-end validation

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

# 31. Current Analysis Result

The current API and dashboard return the backend's existing analysis sections:

```text
ML prediction and phishing probability
    ↓
SHAP contributions for the phishing class
    ↓
Separate provider results and availability
    ↓
Backend correlation status and assessment
```

The current API does not calculate a separate risk score or risk level. The project demonstrates:

**Machine Learning + Explainable AI + Threat Intelligence + Evidence Correlation + Web Application Development.**

---

# 32. Phase 7 Threat Intelligence CLI

The modular provider and evidence-correlation layer is documented in [Threat Intelligence](docs/THREAT_INTELLIGENCE.md) and [Correlation Rules](docs/CORRELATION.md). It distinguishes the model's phishing prediction from PhishTank phishing records, URLhaus malware-distribution intelligence, and optional OpenPhish Community Feed listings. No submitted destination is opened.

The provider layer loads repository-root `.env` values with `python-dotenv` during provider initialization; existing process environment values take precedence. `PHISHTANK_APP_KEY` is optional; `URLHAUS_AUTH_KEY` is required for URLhaus. `.env.example` has empty placeholders only.

```powershell
$env:PHISHTANK_APP_KEY = "your-optional-phishtank-key"
$env:URLHAUS_AUTH_KEY = "your-urlhaus-auth-key"
python scripts/test_threat_intel.py "https://example.com/"
```

The optional OpenPhish Community Feed can be skipped with `--without-openphish`. The manual CLI makes provider lookup requests, so the submitted URL is disclosed to the enabled providers. Provider failures and absent credentials are returned as partial/unavailable evidence.

---

# 33. Phase 8A–8C FastAPI API and frontend

The FastAPI application factory is `src.api.app:create_app`. Start the local development server from the repository root:

```powershell
python -m uvicorn src.api.app:create_app --factory --reload
```

The process health endpoint is `GET http://127.0.0.1:8000/health` and returns:

```json
{"status":"ok"}
```

`/health` is intentionally unversioned for stable process monitoring. Phase 8B adds `POST http://127.0.0.1:8000/api/v1/analyze`:

```json
{"url":"https://example.com"}
```

The response separates `ml` (phishing probability, prediction, threshold, positive class), `explanation` (the strongest five existing SHAP contributions), `threat_intelligence` (one normalized result per provider), and `correlation` (the existing evidence rules). TI provider failures remain in the response as unavailable/error evidence when the local model succeeds. A no-match result means only that the queried source returned no record; it is not proof that the URL is safe.

Interactive API documentation is available at `http://127.0.0.1:8000/docs`; the OpenAPI schema describes the request, response, and validation errors.

The API uses the existing URL-only extractor, saved Random Forest candidate, SHAP explainer, TI providers, and correlation engine through an application service. The model/service is loaded lazily on the first `/api/v1/analyze` request and reused for that app process. The API never opens, downloads, renders, or executes the submitted destination. The only external requests are to the configured TI provider endpoints; submitting a URL discloses it to those providers.

## Frontend

Phase 8C adds a lightweight dashboard built with local HTML5, CSS3, and vanilla JavaScript. The browser sends the entered URL to this FastAPI application's `POST /api/v1/analyze` endpoint and renders the returned ML prediction, SHAP contributions, provider results, and correlation assessment. Analysis logic stays in the backend. The page checks `GET /health` once when it loads and reports whether the API is online.

Start the complete local application from the repository root:

```powershell
python -m uvicorn src.api.app:create_app --factory
```

Open `http://127.0.0.1:8000/` for the dashboard. The API documentation remains at `http://127.0.0.1:8000/docs`, and the health endpoint remains at `http://127.0.0.1:8000/health`.

The dashboard distinguishes model evidence from each provider result. A provider no-match means the URL was not found in that provider's current data and is not proof of safety; unavailable providers remain visibly unavailable. URLhaus malware-distribution results are displayed as malware intelligence, not automatically relabeled as phishing. The frontend does not open the submitted URL, load its content, or send it to third-party services directly. The backend's configured threat-intelligence integrations receive the URL for their lookups. This is an academic research interface, not a production deployment or guarantee of protection.
