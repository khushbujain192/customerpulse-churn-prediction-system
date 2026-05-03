# CustomerPulse: Customer Churn Prediction System

A complete project built with **Python + Flask + SQLite + Scikit-learn**.

This system predicts whether a customer is likely to churn, stores predictions per logged-in user, provides analytics dashboards, supports model retraining with CSV upload, and exposes API endpoints for integration.

## Features
- Secure authentication:
  - User registration
  - User login/logout
  - Password hashing
  - Protected routes
- Churn prediction form with probability output
- User-specific prediction history
- CSV export for prediction history
- Analytics dashboard (Chart.js)
- Model metrics page:
  - Accuracy, ROC-AUC
  - Classification report
  - Confusion matrix
  - Feature coefficients
- Model retraining:
  - Default dataset training
  - Custom CSV upload training
  - Training run history logs
- REST API support

## Tech Stack
- Backend: Flask
- Database: SQLite
- ML: Logistic Regression (Scikit-learn)
- Data: Pandas, NumPy
- Visualization: Chart.js
- Testing: Pytest

## Project Structure
```text
curms/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   ├── customer_churn.csv
│   └── uploads/
├── instance/
│   └── churn_app.db
├── model/
│   ├── churn_pipeline.joblib
│   └── training_metrics.json
├── static/
│   └── style.css
├── templates/
│   ├── base.html
│   ├── login.html
│   ├── register.html
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   ├── dashboard.html
│   ├── metrics.html
│   └── train.html
└── tests/
    └── test_app.py
```

## Required Dataset Columns
Your CSV should include:
- `gender`
- `senior_citizen`
- `partner`
- `dependents`
- `tenure`
- `online_security`
- `online_backup`
- `tech_support`
- `streaming_tv`
- `streaming_movies`
- `payment_method`
- `contract_type`
- `paperless_billing`
- `monthly_charges`
- `total_charges`
- `churn` (`Yes` / `No`)

## Setup
1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. (Optional) Train model manually
```bash
python train_model.py
```

4. Run application
```bash
python app.py
```

5. Open in browser
- `http://127.0.0.1:5000`

## First Use Flow
1. Register a new account
2. Login
3. Go to Predict page and submit customer data
4. Review result, history, dashboard, and metrics

## API Endpoints
- `GET /api/health`
- `POST /api/predict`

### Example Request: `POST /api/predict`
```json
{
  "gender": "Male",
  "senior_citizen": 1,
  "partner": "No",
  "dependents": "No",
  "tenure": 5,
  "online_security": "No",
  "online_backup": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "payment_method": "Electronic check",
  "contract_type": "Month-to-month",
  "paperless_billing": "Yes",
  "monthly_charges": 99.5,
  "total_charges": 497.5
}
```

## Security Note
Set a strong secret key before production:
```bash
export FLASK_SECRET_KEY="your-strong-random-secret"
```

## Run Tests
```bash
pytest -q
```
## Future Enhancements
- Role-based admin panel
- Passsword reset via email OTP/link
- Docker Deployment
- Model versioning and experiment tracking

