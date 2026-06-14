# 📊 LendIQ ML — Digital Lending Portfolio Optimizer

> **ML-Powered Credit Risk Intelligence Platform** for Indian Digital Lending NBFCs  
> Built with Flask · scikit-learn · OpenAI GPT-4o · Vanilla JS

---

## 🚀 Features

| Module | Technology | Description |
|--------|-----------|-------------|
| Default Prediction | Random Forest + Gradient Boosting | Ensemble model predicting loan default probability |
| Early Warning | Isolation Forest + Rule Engine | Anomaly detection scoring 0–100 |
| Customer Segmentation | K-Means (k=4) | Unsupervised borrower clustering |
| Risk Grading | Logistic Regression | AAA → CCC credit quality classification |
| AI Recommendations | OpenAI GPT-4o | Contextual portfolio insights from real data |
| Auth | Session-based | Sign In / Sign Up with validation |

---

## 🛠️ Tech Stack

- **Backend**: Python 3.13, Flask 3.x, Flask-CORS
- **ML**: scikit-learn (RandomForest, GradientBoosting, IsolationForest, KMeans, LogisticRegression)
- **Data**: Pandas 3.x, NumPy 2.x
- **AI**: OpenAI GPT-4o (optional — falls back to ML-only if quota exceeded)
- **Frontend**: HTML5, Vanilla CSS, Vanilla JS, Chart.js (canvas)
- **Theme**: Violet/Indigo premium dark theme

---

## 📁 Project Structure

```
digital-lending-portfolio/
├── app.py                  # Flask backend — all API routes
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── ml/
│   ├── __init__.py
│   ├── data_generator.py   # Synthetic data + CSV normalization
│   ├── models.py           # All ML model definitions & training
│   └── analyzer.py         # Portfolio analytics engine
├── templates/
│   └── index.html          # Single-page app shell (Sign In/Up + Dashboard)
└── static/
    ├── style.css           # Full design system (violet/indigo theme)
    ├── app.js              # Frontend logic, auth flow, API calls
    └── charts.js           # Canvas-based chart renderers
```

---

## ⚡ Quick Start

### 1. Clone / Navigate to project
```bash
cd /Users/sharma/portfolioiit/digital-lending-portfolio
```

### 2. Create virtual environment
```bash
python3 -m venv venv
```

### 3. Install dependencies
```bash
./venv/bin/pip install -r requirements.txt
```

### 4. Run the server
```bash
./venv/bin/python3 app.py
```

### 5. Open in browser
```
http://localhost:5050
```

---

## 📂 CSV Upload Format

Upload any CSV with these columns (exact names or common aliases auto-mapped):

| Field | Aliases | Example |
|-------|---------|---------|
| `loan_amount` | `amount`, `principal`, `disbursed_amount` | `150000` |
| `cibil_score` | `credit_score`, `bureau_score` | `720` |
| `risk_grade` | `grade`, `rating`, `credit_quality` | `A`, `BBB`, `CCC` |
| `region` | `geography`, `location` | `Metro`, `Tier-2`, `Rural` |
| `employment` | `employment_type`, `job_type` | `Salaried`, `Self-Employed` |
| `product` | `product_type`, `loan_type` | `Personal Loan`, `BNPL` |
| `interest_rate` | `pricing_rate`, `roi` | `14.5` |
| `is_default` | `default`, `npa`, `bad_flag` | `0` or `1` |
| `dpd` | `delinquency_status`, `days_past_due` | `Current`, `DPD30+`, `0` |
| `tenure` | `tenure_months`, `term` | `12` |
| `channel` | `acquisition_channel`, `source` | `Digital-Direct` |
| `income` | `income_proxy`, `salary` | `55000` |
| `dti` | `dti_ratio`, `foir` | `0.35` |

> **Download sample template** from the app at `/api/sample-csv`

---

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /` | GET | Serve the SPA |
| `/api/generate` | POST | Generate N synthetic loans & train ML |
| `/api/upload` | POST | Upload CSV, normalize & train ML |
| `/api/metrics` | GET | Portfolio KPI metrics |
| `/api/risk` | GET | Risk distribution + feature importance |
| `/api/early-warning` | GET | Isolation Forest anomaly signals |
| `/api/segments?by=` | GET | Segment breakdown by any dimension |
| `/api/cohorts` | GET | Origination cohort analysis |
| `/api/clusters` | GET | K-Means customer cluster profiles |
| `/api/pricing` | GET | Pricing & ROI by product/grade |
| `/api/recommendations` | GET | ML + GPT-4o credit policy recommendations |
| `/api/predict` | POST | Predict single loan risk |
| `/api/status` | GET | Server & model status |
| `/api/reset` | POST | Clear all data (used by Back button) |
| `/api/sample-csv` | GET | Download sample CSV template |

---

## 🤖 ML Models

### Random Forest (Default Prediction)
- Features: credit score, loan size, DTI, LTV, pricing rate, employment, geography, behavioral signals
- Output: default probability 0–1
- Ensemble: blended 60% RF + 40% Gradient Boosting

### Isolation Forest (Early Warning)
- Trained on active loans only
- Composite score = cashflow (30pts) + balance volatility (25pts) + spending shock (20pts) + DTI (25pts) + anomaly (+10pts)
- High risk: score > 60 | Medium: 40–60 | Low: < 40

### K-Means (Customer Segmentation)
- k = min(4, n_samples) — adapts to small datasets
- Clusters: Prime Borrowers · Growth Segment · Watch List · High Risk

### Logistic Regression (Risk Grade)
- Multi-class: AAA / AA / A / BBB / BB / B / CCC
- Requires ≥ 2 unique grade classes

---

## 🔐 Authentication

The application implements a local client-side authentication system with persistent session memory:

- **Sign Up**: Register with Name, Email, Password, and Organisation. Credentials are encrypted and saved securely inside the browser's `localStorage` database.
- **Sign In**: Validates the entered Email and Password against registered users in the database. Returns clean feedback (e.g. invalid emails, wrong passwords, or unregistered accounts).
- **Session Management**: Successful login stores the user context in `sessionStorage` to maintain session persistence across pages.
- **Back-to-Upload Integration**: Clicking the **← Back** button from the main dashboard resets the active dataset and drops the user back to the upload screen rather than logging them out.

> **Production Note**: For enterprise deployments, integrate this with a standard backend authentication service (e.g., JWT, OAuth2, or Flask-Login).

---

## 🧠 OpenAI Integration

Set your API key in `app.py`:
```python
OPENAI_API_KEY = "sk-proj-..."
```

- Model: `gpt-4o`
- Generates 2 contextual AI recommendations per portfolio
- Falls back gracefully to ML-only recommendations if quota exceeded

---

## 📊 Dashboard Screens

1. **Dashboard** — KPI cards + cohort value + product mix + geography charts
2. **ML Risk Model** — Risk grade distribution + feature importance + scatter plot
3. **Early Warning** — Isolation Forest signals + watch-list table
4. **Segments** — AUM & default rate by geography / employment / product / channel
5. **Cohort Analysis** — Vintage performance matrix
6. **ML Clusters** — K-Means borrower profiles
7. **Pricing & ROI** — Rate optimization by grade and product
8. **ML Recommendations** — AI + ML credit policy memo

---

## 🎨 Theme

**Violet/Indigo Premium Dark Theme**
- Primary: `#7C3AED` (Violet)
- Accent: `#06B6D4` (Cyan)
- Background: Near-black with purple undertones `#06040F`
- Fonts: Inter + JetBrains Mono
- Effects: Radial gradient backgrounds, purple glow shadows, glassmorphism cards

---

## 📝 License

MIT License — Free to use for educational and portfolio purposes.

---

*Built for IIT Portfolio Project · LendIQ ML · 2026*
