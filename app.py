from __future__ import annotations

import csv
import io
import json
import os
import sqlite3
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import joblib
import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
INSTANCE_DIR = BASE_DIR / "instance"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_PATH = INSTANCE_DIR / "churn_app.db"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "training_metrics.json"
DEFAULT_DATASET_PATH = DATA_DIR / "customer_churn.csv"
ALLOWED_UPLOAD_EXTENSIONS = {"csv"}

FEATURE_COLUMNS = [
    "gender",
    "senior_citizen",
    "partner",
    "dependents",
    "tenure",
    "online_security",
    "online_backup",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "payment_method",
    "contract_type",
    "paperless_billing",
    "monthly_charges",
    "total_charges",
]

CHOICES = {
    "gender": ["Female", "Male"],
    "yes_no": ["Yes", "No"],
    "payment_method": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "contract_type": ["Month-to-month", "One year", "Two year"],
}

FORM_DEFAULTS = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "No",
    "dependents": "No",
    "tenure": 12,
    "online_security": "No",
    "online_backup": "No",
    "tech_support": "No",
    "streaming_tv": "No",
    "streaming_movies": "No",
    "payment_method": "Electronic check",
    "contract_type": "Month-to-month",
    "paperless_billing": "Yes",
    "monthly_charges": 70.0,
    "total_charges": 840.0,
}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-this-secret-key")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=12)

_MODEL_CACHE: Any = None
_MODEL_MTIME: float | None = None


def _ensure_folders() -> None:
    for folder in (MODEL_DIR, INSTANCE_DIR, DATA_DIR, UPLOAD_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def _initialize_database() -> None:
    _ensure_folders()
    conn = sqlite3.connect(DATABASE_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            gender TEXT NOT NULL,
            senior_citizen INTEGER NOT NULL,
            partner TEXT NOT NULL,
            dependents TEXT NOT NULL,
            tenure INTEGER NOT NULL,
            online_security TEXT NOT NULL,
            online_backup TEXT NOT NULL,
            tech_support TEXT NOT NULL,
            streaming_tv TEXT NOT NULL,
            streaming_movies TEXT NOT NULL,
            payment_method TEXT NOT NULL,
            contract_type TEXT NOT NULL,
            paperless_billing TEXT NOT NULL,
            monthly_charges REAL NOT NULL,
            total_charges REAL NOT NULL,
            churn_prediction TEXT NOT NULL,
            churn_probability REAL NOT NULL
        )
        """
    )

    prediction_columns = [row[1] for row in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    if "user_id" not in prediction_columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN user_id INTEGER")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            dataset_path TEXT NOT NULL,
            accuracy REAL NOT NULL,
            roc_auc REAL NOT NULL,
            train_rows INTEGER NOT NULL,
            test_rows INTEGER NOT NULL,
            note TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def _get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def _close_db(_: Any) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _fetch_user_by_id(user_id: int) -> sqlite3.Row | None:
    return _get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def _fetch_user_by_email(email: str) -> sqlite3.Row | None:
    return _get_db().execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()


def _is_valid_email(email: str) -> bool:
    email = email.strip()
    if "@" not in email or "." not in email:
        return False
    if email.count("@") != 1:
        return False
    return len(email) >= 6


def _current_user_id() -> int | None:
    if g.current_user is None:
        return None
    return int(g.current_user["id"])


def _set_logged_in_user(user_id: int) -> None:
    session.clear()
    session["user_id"] = int(user_id)
    session.permanent = True


def _next_url(default_endpoint: str = "index") -> str:
    next_path = request.args.get("next") or request.form.get("next")
    if next_path and next_path.startswith("/") and not next_path.startswith("//"):
        return next_path
    return url_for(default_endpoint)


def login_required(view: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(view)
    def wrapped_view(*args: Any, **kwargs: Any) -> Any:
        if g.current_user is None:
            flash("Please login to continue.")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view


@app.before_request
def _load_current_user() -> None:
    user_id = session.get("user_id")
    g.current_user = None

    if user_id is None:
        return

    user = _fetch_user_by_id(int(user_id))
    if user is None:
        session.clear()
        return

    g.current_user = user


@app.context_processor
def _inject_globals() -> Dict[str, Any]:
    return {
        "current_user": g.get("current_user"),
        "current_year": datetime.now().year,
    }


def _record_training_run(metrics: Dict[str, Any], note: str | None = None) -> None:
    db = _get_db()
    db.execute(
        """
        INSERT INTO training_runs (
            created_at,
            dataset_path,
            accuracy,
            roc_auc,
            train_rows,
            test_rows,
            note
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            str(metrics.get("dataset_path", DEFAULT_DATASET_PATH)),
            float(metrics.get("accuracy", 0)),
            float(metrics.get("roc_auc", 0)),
            int(metrics.get("train_rows", 0)),
            int(metrics.get("test_rows", 0)),
            note,
        ),
    )
    db.commit()


def _ensure_model_exists() -> None:
    if MODEL_PATH.exists():
        return

    from train_model import train_and_save_model

    train_and_save_model()


def _load_metrics() -> Dict[str, Any]:
    if not METRICS_PATH.exists():
        return {}

    with METRICS_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _load_model(force_refresh: bool = False) -> Any:
    global _MODEL_CACHE
    global _MODEL_MTIME

    _ensure_model_exists()

    model_mtime = MODEL_PATH.stat().st_mtime
    if force_refresh or _MODEL_CACHE is None or _MODEL_MTIME != model_mtime:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        _MODEL_MTIME = model_mtime

    return _MODEL_CACHE


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_yes_no(value: Any, default: str = "No") -> str:
    as_text = str(value).strip().title()
    return as_text if as_text in CHOICES["yes_no"] else default


def _parse_payload(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "gender": str(raw_data.get("gender", FORM_DEFAULTS["gender"])).strip().title(),
        "senior_citizen": _safe_int(raw_data.get("senior_citizen"), FORM_DEFAULTS["senior_citizen"]),
        "partner": _normalize_yes_no(raw_data.get("partner"), FORM_DEFAULTS["partner"]),
        "dependents": _normalize_yes_no(raw_data.get("dependents"), FORM_DEFAULTS["dependents"]),
        "tenure": _safe_int(raw_data.get("tenure"), FORM_DEFAULTS["tenure"]),
        "online_security": _normalize_yes_no(
            raw_data.get("online_security"),
            FORM_DEFAULTS["online_security"],
        ),
        "online_backup": _normalize_yes_no(raw_data.get("online_backup"), FORM_DEFAULTS["online_backup"]),
        "tech_support": _normalize_yes_no(raw_data.get("tech_support"), FORM_DEFAULTS["tech_support"]),
        "streaming_tv": _normalize_yes_no(raw_data.get("streaming_tv"), FORM_DEFAULTS["streaming_tv"]),
        "streaming_movies": _normalize_yes_no(
            raw_data.get("streaming_movies"),
            FORM_DEFAULTS["streaming_movies"],
        ),
        "payment_method": str(raw_data.get("payment_method", FORM_DEFAULTS["payment_method"])).strip(),
        "contract_type": str(raw_data.get("contract_type", FORM_DEFAULTS["contract_type"])).strip(),
        "paperless_billing": _normalize_yes_no(
            raw_data.get("paperless_billing"),
            FORM_DEFAULTS["paperless_billing"],
        ),
        "monthly_charges": _safe_float(raw_data.get("monthly_charges"), FORM_DEFAULTS["monthly_charges"]),
        "total_charges": _safe_float(raw_data.get("total_charges"), FORM_DEFAULTS["total_charges"]),
    }

    if payload["gender"] not in CHOICES["gender"]:
        payload["gender"] = FORM_DEFAULTS["gender"]

    if payload["senior_citizen"] not in (0, 1):
        payload["senior_citizen"] = FORM_DEFAULTS["senior_citizen"]

    if payload["payment_method"] not in CHOICES["payment_method"]:
        payload["payment_method"] = FORM_DEFAULTS["payment_method"]

    if payload["contract_type"] not in CHOICES["contract_type"]:
        payload["contract_type"] = FORM_DEFAULTS["contract_type"]

    payload["tenure"] = max(payload["tenure"], 1)
    payload["monthly_charges"] = max(payload["monthly_charges"], 0.0)
    payload["total_charges"] = max(payload["total_charges"], 0.0)

    return payload


def _predict(payload: Dict[str, Any]) -> Tuple[str, float]:
    model = _load_model()
    input_frame = pd.DataFrame([payload], columns=FEATURE_COLUMNS)
    churn_probability = float(model.predict_proba(input_frame)[0][1])
    prediction = "Yes" if churn_probability >= 0.5 else "No"
    return prediction, churn_probability


def _save_prediction(payload: Dict[str, Any], prediction: str, probability: float, user_id: int | None = None) -> None:
    db = _get_db()
    db.execute(
        """
        INSERT INTO predictions (
            created_at,
            gender,
            senior_citizen,
            partner,
            dependents,
            tenure,
            online_security,
            online_backup,
            tech_support,
            streaming_tv,
            streaming_movies,
            payment_method,
            contract_type,
            paperless_billing,
            monthly_charges,
            total_charges,
            churn_prediction,
            churn_probability,
            user_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            payload["gender"],
            payload["senior_citizen"],
            payload["partner"],
            payload["dependents"],
            payload["tenure"],
            payload["online_security"],
            payload["online_backup"],
            payload["tech_support"],
            payload["streaming_tv"],
            payload["streaming_movies"],
            payload["payment_method"],
            payload["contract_type"],
            payload["paperless_billing"],
            payload["monthly_charges"],
            payload["total_charges"],
            prediction,
            probability,
            user_id,
        ),
    )
    db.commit()


def _prediction_summary(rows: List[sqlite3.Row]) -> Dict[str, Any]:
    total_predictions = len(rows)
    if total_predictions == 0:
        return {
            "total_predictions": 0,
            "churn_count": 0,
            "non_churn_count": 0,
            "avg_probability": 0,
            "churn_rate": 0,
            "contract_labels": [],
            "contract_churn_counts": [],
            "payment_labels": [],
            "payment_churn_counts": [],
            "tenure_bucket_labels": [],
            "tenure_bucket_counts": [],
        }

    churn_count = sum(1 for row in rows if row["churn_prediction"] == "Yes")
    non_churn_count = total_predictions - churn_count
    avg_probability = sum(float(row["churn_probability"]) for row in rows) / total_predictions

    contract_stats: Dict[str, int] = {}
    payment_stats: Dict[str, int] = {}
    tenure_buckets = {"0-12": 0, "13-24": 0, "25-48": 0, "49+": 0}

    for row in rows:
        if row["churn_prediction"] == "Yes":
            contract_key = row["contract_type"]
            payment_key = row["payment_method"]
            contract_stats[contract_key] = contract_stats.get(contract_key, 0) + 1
            payment_stats[payment_key] = payment_stats.get(payment_key, 0) + 1

            tenure_value = int(row["tenure"])
            if tenure_value <= 12:
                tenure_buckets["0-12"] += 1
            elif tenure_value <= 24:
                tenure_buckets["13-24"] += 1
            elif tenure_value <= 48:
                tenure_buckets["25-48"] += 1
            else:
                tenure_buckets["49+"] += 1

    return {
        "total_predictions": total_predictions,
        "churn_count": churn_count,
        "non_churn_count": non_churn_count,
        "avg_probability": round(avg_probability * 100, 2),
        "churn_rate": round((churn_count / total_predictions) * 100, 2),
        "contract_labels": list(contract_stats.keys()),
        "contract_churn_counts": list(contract_stats.values()),
        "payment_labels": list(payment_stats.keys()),
        "payment_churn_counts": list(payment_stats.values()),
        "tenure_bucket_labels": list(tenure_buckets.keys()),
        "tenure_bucket_counts": list(tenure_buckets.values()),
    }


def _classification_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    report = metrics.get("classification_report", {})
    rows: List[Dict[str, Any]] = []

    for label in ("0", "1", "macro avg", "weighted avg"):
        if label not in report:
            continue
        entry = report[label]
        rows.append(
            {
                "label": label,
                "precision": float(entry.get("precision", 0)),
                "recall": float(entry.get("recall", 0)),
                "f1_score": float(entry.get("f1-score", 0)),
                "support": int(entry.get("support", 0)),
            }
        )

    return rows


def _allowed_upload(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_UPLOAD_EXTENSIONS


def _recent_training_runs(limit: int = 10) -> List[sqlite3.Row]:
    return _get_db().execute(
        "SELECT * FROM training_runs ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()


def _bootstrap_training_log() -> None:
    existing = _get_db().execute("SELECT COUNT(*) AS count FROM training_runs").fetchone()
    if existing and int(existing["count"]) > 0:
        return

    metrics = _load_metrics()
    if metrics:
        _record_training_run(metrics, note="Initial model record")


_initialize_database()

with app.app_context():
    _ensure_model_exists()
    _bootstrap_training_log()


@app.route("/register", methods=["GET", "POST"])
def register() -> str:
    if g.current_user is not None:
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("register.html")

    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not full_name:
        flash("Full name is required.")
        return redirect(url_for("register"))

    if not _is_valid_email(email):
        flash("Please enter a valid email address.")
        return redirect(url_for("register"))

    if len(password) < 6:
        flash("Password must be at least 6 characters.")
        return redirect(url_for("register"))

    if password != confirm_password:
        flash("Password and confirm password do not match.")
        return redirect(url_for("register"))

    if _fetch_user_by_email(email) is not None:
        flash("This email is already registered.")
        return redirect(url_for("register"))

    password_hash = generate_password_hash(password)
    db = _get_db()
    db.execute(
        """
        INSERT INTO users (full_name, email, password_hash, role, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            full_name,
            email,
            password_hash,
            "user",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    db.commit()

    user = _fetch_user_by_email(email)
    if user is None:
        flash("Registration failed. Please try again.")
        return redirect(url_for("register"))

    _set_logged_in_user(int(user["id"]))
    flash("Registration successful. Welcome!")
    return redirect(url_for("index"))


@app.route("/login", methods=["GET", "POST"])
def login() -> str:
    if g.current_user is not None:
        return redirect(url_for("index"))

    if request.method == "GET":
        return render_template("login.html", next_url=request.args.get("next", ""))

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    user = _fetch_user_by_email(email)
    if user is None or not check_password_hash(str(user["password_hash"]), password):
        flash("Invalid email or password.")
        return redirect(url_for("login"))

    _set_logged_in_user(int(user["id"]))
    flash("Login successful.")
    return redirect(_next_url("index"))


@app.route("/logout", methods=["POST"])
def logout() -> str:
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index() -> str:
    metrics = _load_metrics()
    return render_template(
        "index.html",
        defaults=FORM_DEFAULTS,
        choices=CHOICES,
        metrics=metrics,
    )


@app.route("/predict", methods=["POST"])
@login_required
def predict() -> str:
    try:
        payload = _parse_payload(dict(request.form))
        prediction, churn_probability = _predict(payload)
        _save_prediction(payload, prediction, churn_probability, user_id=_current_user_id())

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(churn_probability * 100, 2),
            payload=payload,
        )
    except Exception as exc:
        flash(f"Prediction failed: {exc}")
        return redirect(url_for("index"))


@app.route("/history")
@login_required
def history() -> str:
    rows = _get_db().execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC LIMIT 200",
        (_current_user_id(),),
    ).fetchall()
    return render_template("history.html", rows=rows)


@app.route("/history/export")
@login_required
def export_history() -> Response:
    rows = _get_db().execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
        (_current_user_id(),),
    ).fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "created_at",
            *FEATURE_COLUMNS,
            "churn_prediction",
            "churn_probability",
        ]
    )

    for row in rows:
        writer.writerow(
            [
                row["id"],
                row["created_at"],
                row["gender"],
                row["senior_citizen"],
                row["partner"],
                row["dependents"],
                row["tenure"],
                row["online_security"],
                row["online_backup"],
                row["tech_support"],
                row["streaming_tv"],
                row["streaming_movies"],
                row["payment_method"],
                row["contract_type"],
                row["paperless_billing"],
                row["monthly_charges"],
                row["total_charges"],
                row["churn_prediction"],
                row["churn_probability"],
            ]
        )

    response = Response(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=prediction_history.csv"
    return response


@app.route("/dashboard")
@login_required
def dashboard() -> str:
    rows = _get_db().execute(
        "SELECT * FROM predictions WHERE user_id = ? ORDER BY id DESC",
        (_current_user_id(),),
    ).fetchall()
    summary = _prediction_summary(rows)
    return render_template("dashboard.html", summary=summary)


@app.route("/metrics")
@login_required
def metrics() -> str:
    model_metrics = _load_metrics()
    report_rows = _classification_rows(model_metrics)
    training_runs = _recent_training_runs(limit=15)
    return render_template(
        "metrics.html",
        metrics=model_metrics,
        report_rows=report_rows,
        training_runs=training_runs,
    )


@app.route("/train", methods=["GET", "POST"])
@login_required
def train_model_view() -> str:
    if request.method == "GET":
        return render_template(
            "train.html",
            default_dataset=DEFAULT_DATASET_PATH,
            training_runs=_recent_training_runs(),
        )

    try:
        dataset_mode = request.form.get("dataset_mode", "default")
        selected_dataset = DEFAULT_DATASET_PATH
        note = f"Manual training by {g.current_user['email']} with default dataset"

        if dataset_mode == "upload":
            uploaded_file = request.files.get("dataset_file")
            if uploaded_file is None or uploaded_file.filename is None or uploaded_file.filename.strip() == "":
                raise ValueError("Please choose a CSV file to upload.")

            if not _allowed_upload(uploaded_file.filename):
                raise ValueError("Only CSV files are allowed.")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cleaned_name = secure_filename(uploaded_file.filename)
            final_name = f"{timestamp}_{cleaned_name}"
            selected_dataset = UPLOAD_DIR / final_name
            uploaded_file.save(selected_dataset)
            note = f"Manual training by {g.current_user['email']} with uploaded dataset: {final_name}"

        from train_model import train_and_save_model

        train_and_save_model(selected_dataset)
        metrics_data = _load_metrics()
        _record_training_run(metrics_data, note=note)
        _load_model(force_refresh=True)

        flash("Model training completed successfully.")
        return redirect(url_for("metrics"))

    except Exception as exc:
        flash(f"Training failed: {exc}")
        return redirect(url_for("train_model_view"))


@app.route("/api/health")
def api_health() -> Response:
    metrics_data = _load_metrics()
    return jsonify(
        {
            "status": "ok",
            "model_exists": MODEL_PATH.exists(),
            "database_exists": DATABASE_PATH.exists(),
            "last_known_accuracy": metrics_data.get("accuracy"),
            "last_known_roc_auc": metrics_data.get("roc_auc"),
        }
    )


@app.route("/api/predict", methods=["POST"])
def api_predict() -> Response:
    try:
        payload = _parse_payload(request.get_json(force=True, silent=False) or {})
        prediction, churn_probability = _predict(payload)

        should_save = str(request.args.get("save", "1")).strip() != "0"
        if should_save:
            _save_prediction(payload, prediction, churn_probability, user_id=_current_user_id())

        return jsonify(
            {
                "prediction": prediction,
                "churn_probability": round(churn_probability, 6),
                "churn_probability_percent": round(churn_probability * 100, 2),
                "saved": should_save,
                "saved_by_user": bool(_current_user_id()),
                "input": payload,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(debug=True)
