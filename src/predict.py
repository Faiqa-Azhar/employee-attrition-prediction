"""
Standalone prediction script.
Loads the best trained pipeline and predicts attrition for new employee data,
either from a CSV file or a single hardcoded example.

Usage:
    python predict.py                      # runs the built-in example
    python predict.py --csv path/to.csv    # predicts for every row in a CSV
"""

import argparse
import pandas as pd
import joblib
import os

from utils import MODELS_DIR


def load_model():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run train.py first."
        )
    return joblib.load(model_path)


def predict(pipeline, X: pd.DataFrame, threshold: float = 0.30):
    """
    Returns a DataFrame with prediction, probability, and risk tier
    for each row in X. Threshold default (0.30) matches the value
    chosen during train.py's threshold tuning for the best model —
    update this if your best model's chosen threshold differs.
    """
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = X.copy()
    results["Attrition_Prediction"] = ["Leave" if p == 1 else "Stay" for p in predictions]
    results["Flight_Risk_Probability"] = probabilities.round(4)
    results["Risk_Tier"] = pd.cut(
        probabilities,
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
        include_lowest=True
    )
    return results


def get_example_employee():
    """A single example employee dict, matching all raw input columns
    the pipeline's FeatureEngineer step expects before transformation."""
    return pd.DataFrame([{
        "Age": 29, "MonthlyIncome": 2800, "JobLevel": 1,
        "YearsAtCompany": 3, "TotalWorkingYears": 5,
        "OverTime": "Yes", "JobSatisfaction": 2,
        "EnvironmentSatisfaction": 2, "WorkLifeBalance": 2,
        "RelationshipSatisfaction": 3, "Department": "Sales",
        "JobRole": "Sales Representative", "MaritalStatus": "Single",
        "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 2,
        "Gender": "Male", "BusinessTravel": "Travel_Frequently",
        "DailyRate": 700, "DistanceFromHome": 12, "Education": 2,
        "EducationField": "Marketing", "HourlyRate": 55, "JobInvolvement": 2,
        "MonthlyRate": 14000, "NumCompaniesWorked": 3, "PercentSalaryHike": 11,
        "PerformanceRating": 3, "StockOptionLevel": 0, "TrainingTimesLastYear": 1,
        "YearsInCurrentRole": 2
    }])


def main():
    parser = argparse.ArgumentParser(description="Predict employee attrition risk.")
    parser.add_argument("--csv", type=str, default=None,
                         help="Path to a CSV of employees to predict on. "
                              "If omitted, runs on a built-in example employee.")
    parser.add_argument("--threshold", type=float, default=0.30,
                         help="Probability threshold for Leave/Stay classification.")
    parser.add_argument("--output", type=str, default=None,
                         help="Optional path to save predictions as CSV.")
    args = parser.parse_args()

    pipeline = load_model()

    if args.csv:
        print(f"Loading employees from {args.csv}...")
        X = pd.read_csv(args.csv)
    else:
        print("No --csv provided. Running on built-in example employee.")
        X = get_example_employee()

    results = predict(pipeline, X, threshold=args.threshold)

    print("\n=== PREDICTION RESULTS ===")
    display_cols = ["Attrition_Prediction", "Flight_Risk_Probability", "Risk_Tier"]
    print(results[display_cols].to_string(index=True))

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nSaved full results to {args.output}")


if __name__ == "__main__":
    main()