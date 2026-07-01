"""
Preprocessing pipeline: feature engineering + encoding + scaling.
Built as a sklearn-compatible pipeline to prevent data leakage.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils import (DROP_COLS, TARGET_COL, NOMINAL_COLS,
                    BINARY_MAP, BUSINESS_TRAVEL_MAP)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer: drops useless cols, creates interaction features,
    encodes binary/ordinal text columns. Fits into a sklearn Pipeline cleanly."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Drop constant / ID columns if still present
        X = X.drop(columns=[c for c in DROP_COLS if c in X.columns], errors="ignore")

        # Binary mapping
        for col, mapping in BINARY_MAP.items():
            if col in X.columns:
                X[col] = X[col].map(mapping)

        # BusinessTravel ordinal mapping
        if "BusinessTravel" in X.columns:
            X["BusinessTravel"] = X["BusinessTravel"].map(BUSINESS_TRAVEL_MAP)

        # ── INTERACTION FEATURES ──
        X["PromotionStagnationIndex"] = X["YearsSinceLastPromotion"] / (X["YearsAtCompany"] + 1)
        X["IncomePerExperience"] = X["MonthlyIncome"] / (X["TotalWorkingYears"] + 1)
        X["LoyaltyScore"] = X["YearsWithCurrManager"] / (X["YearsAtCompany"] + 1)
        X["CareerVelocity"] = X["JobLevel"] / (X["TotalWorkingYears"] + 1)
        X["SatisfactionComposite"] = (
            X["JobSatisfaction"] + X["EnvironmentSatisfaction"] +
            X["RelationshipSatisfaction"] + X["WorkLifeBalance"]
        ) / 4
        X["OvertimeBurdenFlag"] = ((X["OverTime"] == 1) & (X["WorkLifeBalance"] <= 2)).astype(int)
        X["RoleStability"] = X["YearsInCurrentRole"] / (X["YearsAtCompany"] + 1)

        return X


def get_preprocessing_pipeline(scale_for_logreg=False):
    """
    Returns a single ColumnTransformer (NOT a nested Pipeline) that one-hot
    encodes nominal features and optionally scales numeric features.
    Must be a single transformer because imblearn.Pipeline does not allow
    nested Pipelines as intermediate steps.
    """
    nominal_present = [c for c in NOMINAL_COLS]

    if scale_for_logreg:
        transformers = [
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal_present),
            ("scale", StandardScaler(), make_column_selector(dtype_include=np.number))
        ]
    else:
        transformers = [
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal_present)
        ]

    ct = ColumnTransformer(transformers=transformers, remainder="passthrough")

    full_pipeline = Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("encode", ct)
    ])

    return full_pipeline

def load_and_prepare_target(df):
    """Splits dataframe into X and binary-encoded y."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].map({"Yes": 1, "No": 0})
    return X, y