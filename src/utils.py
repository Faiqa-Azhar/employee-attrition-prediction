"""
Shared constants and configuration for the Employee Attrition project.
All other scripts import from here to avoid duplication.
"""

import os

# ── PATHS ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "HR-Attrition.csv")
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

# ── COLUMNS TO DROP (zero variance / non-informative) ────
DROP_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

# ── TARGET ────────────────────────────────────────────────
TARGET_COL = "Attrition"

# ── ORDINAL FEATURES (already 1-4 / 1-5 scales — natural order) ──
ORDINAL_COLS = [
    "Education", "EnvironmentSatisfaction", "JobInvolvement",
    "JobLevel", "JobSatisfaction", "PerformanceRating",
    "RelationshipSatisfaction", "StockOptionLevel", "WorkLifeBalance"
]

# BusinessTravel needs manual ordinal mapping (text -> order)
BUSINESS_TRAVEL_MAP = {
    "Non-Travel": 0,
    "Travel_Rarely": 1,
    "Travel_Frequently": 2
}

# ── NOMINAL FEATURES (no order — one-hot encode) ─────────
NOMINAL_COLS = ["Department", "JobRole", "MaritalStatus", "EducationField"]

# ── BINARY FEATURES (simple 0/1 mapping) ─────────────────
BINARY_MAP = {
    "Gender": {"Male": 1, "Female": 0},
    "OverTime": {"Yes": 1, "No": 0}
}

RANDOM_STATE = 42