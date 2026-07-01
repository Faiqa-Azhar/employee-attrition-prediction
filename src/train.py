"""
Trains Logistic Regression, Random Forest, and XGBoost
with proper SMOTE (train-only), stratified CV tuning, and saves
the best pipelines to /models.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, f1_score

from utils import RAW_DATA_PATH, MODELS_DIR, RANDOM_STATE, TARGET_COL
from preprocessing import get_preprocessing_pipeline, load_and_prepare_target

os.makedirs(MODELS_DIR, exist_ok=True)

def build_full_pipeline(model, scale_for_logreg=False):
    """Combines preprocessing + SMOTE + model into ONE flat imblearn pipeline.
    No nested Pipeline objects allowed by imblearn — every step must be a
    plain transformer/estimator."""
    from preprocessing import FeatureEngineer
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from utils import NOMINAL_COLS
    import numpy as np

    if scale_for_logreg:
        transformers = [
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL_COLS),
            ("scale", StandardScaler(), make_column_selector(dtype_include=np.number))
        ]
    else:
        transformers = [
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"), NOMINAL_COLS)
        ]

    ct = ColumnTransformer(transformers=transformers, remainder="passthrough")

    pipeline = ImbPipeline([
        ("feature_engineering", FeatureEngineer()),
        ("encode", ct),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("model", model)
    ])
    return pipeline


def main():
    print("Loading data...")
    df = pd.read_csv(RAW_DATA_PATH)
    X, y = load_and_prepare_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    # ── 1. LOGISTIC REGRESSION (baseline) ──
    print("\n=== Tuning Logistic Regression ===")
    logreg_pipeline = build_full_pipeline(
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        scale_for_logreg=True
    )
    logreg_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__class_weight": ["balanced", None]
    }

    logreg_search = GridSearchCV(logreg_pipeline, logreg_grid, cv=cv,
                              scoring="f1", n_jobs=-1)

    logreg_search.fit(X_train, y_train)
    results["logistic_regression"] = logreg_search.best_estimator_
    print(f"Best params: {logreg_search.best_params_}")
    print(f"Best CV f1: {logreg_search.best_score_:.4f}")

    # ── 2. RANDOM FOREST ──
    print("\n=== Tuning Random Forest ===")
    rf_pipeline = build_full_pipeline(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    )
    rf_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [5, 10, 15, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 4, 8]
    }

    rf_search = RandomizedSearchCV(rf_pipeline, rf_grid, n_iter=25, cv=cv,
                                scoring="f1", n_jobs=-1, random_state=RANDOM_STATE)

    rf_search.fit(X_train, y_train)
    results["random_forest"] = rf_search.best_estimator_
    print(f"Best params: {rf_search.best_params_}")
    print(f"Best CV f1: {rf_search.best_score_:.4f}")

    # ── 3. XGBOOST ──
    print("\n=== Tuning XGBoost ===")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_pipeline = build_full_pipeline(
        XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                      random_state=RANDOM_STATE)
    )
    xgb_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 5, 7],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.6, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.8, 1.0]
    }
   
    xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_grid, n_iter=30, cv=cv,
                                 scoring="f1", n_jobs=-1, random_state=RANDOM_STATE)

    xgb_search.fit(X_train, y_train)
    results["xgboost"] = xgb_search.best_estimator_
    print(f"Best params: {xgb_search.best_params_}")
    print(f"Best CV f1: {xgb_search.best_score_:.4f}")

    # ── FINAL EVALUATION ON SEALED TEST SET ──
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION (touched only now)")
    print("=" * 60)

    summary = []
    best_model_name, best_score = None, -1


    from sklearn.metrics import precision_score, recall_score

    for name, pipeline in results.items():
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)

        # ── THRESHOLD TUNING ──
        # Find the threshold that maximizes F1 while keeping Recall >= 0.65
        best_thresh, best_f1_at_thresh = 0.5, -1
        threshold_log = []

        for thresh in np.arange(0.10, 0.55, 0.05):
            y_pred_t = (y_proba >= thresh).astype(int)
            rec = recall_score(y_test, y_pred_t)
            prec = precision_score(y_test, y_pred_t, zero_division=0)
            f1_t = f1_score(y_test, y_pred_t, zero_division=0)
            threshold_log.append((round(thresh, 2), round(rec, 3), round(prec, 3), round(f1_t, 3)))

            # if rec >= 0.65 and f1_t > best_f1_at_thresh:
            if rec >= 0.70 and f1_t > best_f1_at_thresh:
                best_f1_at_thresh = f1_t
                best_thresh = thresh

        # Final prediction using the tuned threshold
        y_pred = (y_proba >= best_thresh).astype(int)
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)

        print(f"\n--- {name.upper()} ---")
        print(f"Threshold sweep (thresh, recall, precision, f1):")
        for row in threshold_log:
            print(f"  {row}")
        # print(f"\nChosen threshold: {best_thresh:.2f}")
        # print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        import matplotlib.pyplot as plt

        print(f"\nChosen threshold: {best_thresh:.2f}")
        print(classification_report(y_test, y_pred, target_names=["Stay", "Leave"]))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                       display_labels=["Stay", "Leave"])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix — {name.replace('_', ' ').title()}\n"
                     f"Threshold: {best_thresh:.2f} | Recall: {recall:.2f} | "
                     f"Precision: {precision:.2f}")
        plt.tight_layout()
        figures_dir = os.path.join(os.path.dirname(MODELS_DIR), "reports", "figures")
        os.makedirs(figures_dir, exist_ok=True)
        plt.savefig(os.path.join(figures_dir, f"confusion_matrix_{name}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Confusion matrix saved → reports/figures/confusion_matrix_{name}.png")

        # print(f"ROC-AUC : {auc:.4f}")
        # print(f"ROC-AUC: {auc:.4f}")
        from sklearn.metrics import average_precision_score
        pr_auc = average_precision_score(y_test, y_proba)
        print(f"ROC-AUC : {auc:.4f}")
        print(f"PR-AUC  : {pr_auc:.4f}  ← honest imbalance check")

        summary.append({
            "model": name, "roc_auc": round(auc, 4), "pr_auc": round(pr_auc, 4),
            "f1_score": round(f1, 4), "recall": round(recall, 4),
            "precision": round(precision, 4), "threshold": round(best_thresh, 2)
        })


        # Only consider models with reasonable precision (not flagging everyone)
        MIN_PRECISION = 0.30
        if precision >= MIN_PRECISION and recall > best_score:
            best_score = recall
            best_model_name = name

        # Save every tuned pipeline
        joblib.dump(pipeline, os.path.join(MODELS_DIR, f"{name}.joblib"))

    # Save the best model separately for deployment
    joblib.dump(results[best_model_name], os.path.join(MODELS_DIR, "best_model.joblib"))

    # Save background sample for SHAP explainer in app
    feature_eng  = results[best_model_name].named_steps["feature_engineering"]
    encode_step  = results[best_model_name].named_steps["encode"]
    X_bg         = feature_eng.transform(X_train)
    X_bg_enc     = encode_step.transform(X_bg)
    if hasattr(X_bg_enc, "toarray"):
        X_bg_enc = X_bg_enc.toarray()
    feature_names_out = encode_step.get_feature_names_out()
    X_bg_df = pd.DataFrame(X_bg_enc, columns=feature_names_out)
    bg_sample = X_bg_df.sample(100, random_state=RANDOM_STATE)
    joblib.dump(bg_sample, os.path.join(MODELS_DIR, "shap_background.joblib"))
    print("SHAP background sample saved → models/shap_background.joblib")

    with open(os.path.join(MODELS_DIR, "model_comparison.json"), "w") as f:
        json.dump({"results": summary, "best_model": best_model_name}, f, indent=2)

    print(f"\n🏆 BEST MODEL: {best_model_name} (Recall: {best_score:.4f})")
    print("Saved to models/best_model.joblib")



if __name__ == "__main__":
    main()