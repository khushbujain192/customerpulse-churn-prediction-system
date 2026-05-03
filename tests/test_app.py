from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import app


def _sample_payload() -> dict:
    return {
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
        "total_charges": 497.5,
    }


def _register_and_login(client) -> None:
    unique = uuid.uuid4().hex[:10]
    email = f"user_{unique}@example.com"

    response = client.post(
        "/register",
        data={
            "full_name": "Test User",
            "email": email,
            "password": "testpass123",
            "confirm_password": "testpass123",
        },
        follow_redirects=True,
    )

    assert response.status_code == 200


def test_auth_protection_and_routes() -> None:
    app.testing = True
    client = app.test_client()

    unauth = client.get("/", follow_redirects=False)
    assert unauth.status_code == 302
    assert "/login" in unauth.headers.get("Location", "")

    _register_and_login(client)

    assert client.get("/").status_code == 200
    assert client.get("/history").status_code == 200
    assert client.get("/dashboard").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/train").status_code == 200


def test_api_health() -> None:
    app.testing = True
    client = app.test_client()

    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()

    assert body is not None
    assert body["status"] == "ok"
    assert "model_exists" in body
    assert "database_exists" in body


def test_api_predict() -> None:
    app.testing = True
    client = app.test_client()

    response = client.post("/api/predict", json=_sample_payload())
    assert response.status_code == 200

    body = response.get_json()
    assert body is not None
    assert body["prediction"] in {"Yes", "No"}
    assert 0 <= body["churn_probability"] <= 1
    assert "input" in body


def test_export_history_csv_authenticated() -> None:
    app.testing = True
    client = app.test_client()

    _register_and_login(client)

    client.post("/predict", data=_sample_payload(), follow_redirects=True)
    response = client.get("/history/export")

    assert response.status_code == 200
    assert "text/csv" in response.headers.get("Content-Type", "")
    assert "id,created_at" in response.get_data(as_text=True)
