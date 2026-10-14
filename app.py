
from flask import Flask, render_template, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Load saved model and scaler
MODEL_PATH = BASE_DIR / "model" / "kmeans_model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"

kmeans = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    income = ""
    spending = ""

    if request.method == "POST":
        try:
            income = request.form.get("annual_income", "").strip()
            spending = request.form.get("spending_score", "").strip()

            annual_income = float(income)
            spending_score = float(spending)

            if annual_income < 0:
                raise ValueError("Annual income cannot be negative.")

            if not 1 <= spending_score <= 100:
                raise ValueError("Spending score must be between 1 and 100.")

            # Prepare input using the same features as training
            customer = pd.DataFrame(
                [[annual_income, spending_score]],
                columns=[
                    "Annual Income (k$)",
                    "Spending Score (1-100)"
                ]
            )

            # Scale and predict
            customer_scaled = scaler.transform(customer)
            prediction = int(kmeans.predict(customer_scaled)[0])

        except ValueError as e:
            error = str(e) or "Please enter valid numbers."

        except Exception as e:
            app.logger.exception("Prediction failed")
            error = "Prediction failed. Check your saved model and scaler."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error,
        income=income,
        spending=spending
    )


if __name__ == "__main__":
    app.run(debug=True)
