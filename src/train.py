"""Titanic survival prediction.

Pipeline:
1. Load data and print a short EDA summary
2. Feature engineering (title, family size, is_alone)
3. Preprocessing (impute, scale, one-hot) inside a scikit-learn pipeline
4. Compare Logistic Regression, Decision Tree and Random Forest (5-fold CV, held-out test set)
5. Tune the best model with GridSearchCV; SMOTE is applied inside the CV folds so it never leaks
6. Save the final pipeline with joblib, plus metrics and plots

Run:  python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "titanic.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

NUMERIC = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title", "IsAlone"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create Title, FamilySize and IsAlone. Does not use the target, so it is leakage-safe."""
    df = df.copy()
    title = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    rare = {t for t, c in title.value_counts().items() if c < 10}
    df["Title"] = title.replace(list(rare), "Rare").replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Pclass"] = df["Pclass"].astype(str)
    df["IsAlone"] = df["IsAlone"].astype(str)
    return df


def eda_summary(df: pd.DataFrame) -> dict:
    summary = {
        "rows": int(len(df)),
        "survival_rate": round(float(df["Survived"].mean()), 3),
        "missing_pct": df[["Age", "Cabin", "Embarked"]].isna().mean().mul(100).round(1).to_dict(),
        "survival_by_sex": df.groupby("Sex")["Survived"].mean().round(3).to_dict(),
        "survival_by_class": df.groupby("Pclass")["Survived"].mean().round(3).to_dict(),
    }
    print("EDA summary:", json.dumps(summary, indent=2))
    return summary


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }


def scores(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = add_features(load_data())
    eda = eda_summary(df)

    X = df[NUMERIC + CATEGORICAL]
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---- 1. Compare models (cross-validated on the training split only)
    comparison = {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        comparison[name] = {
            "cv_roc_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_roc_auc_std": round(float(cv_auc.std()), 4),
            "test": scores(y_test, pipe.predict(X_test), proba),
        }
        print(f"{name:20s} CV AUC {cv_auc.mean():.3f} +/- {cv_auc.std():.3f} | test {comparison[name]['test']}")

    best_name = max(comparison, key=lambda n: comparison[n]["cv_roc_auc_mean"])
    print(f"\nBest by CV ROC-AUC: {best_name}")

    # ---- 2. Tune the best model; SMOTE lives inside the pipeline so it only touches training folds
    base = candidate_models()[best_name]
    tune_pipe = ImbPipeline(
        [
            ("prep", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", base),
        ]
    )
    grids = {
        "Random Forest": {
            "model__n_estimators": [100, 300],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "Decision Tree": {"model__max_depth": [3, 4, 5, 7], "model__min_samples_leaf": [1, 5, 10]},
        "Logistic Regression": {"model__C": [0.01, 0.1, 1, 10]},
    }
    search = GridSearchCV(tune_pipe, grids[best_name], scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_, "| best CV AUC:", round(search.best_score_, 4))

    final = search.best_estimator_
    proba = final.predict_proba(X_test)[:, 1]
    pred = final.predict(X_test)
    final_scores = scores(y_test, pred, proba)
    print("Tuned model on held-out test set:", final_scores)

    # ---- 3. Save artifacts
    joblib.dump(final, MODELS_DIR / "titanic_pipeline.joblib")
    results = {
        "eda": eda,
        "model_comparison": comparison,
        "best_model": best_name,
        "best_params": {k: (v if v is not None else "None") for k, v in search.best_params_.items()},
        "tuned_cv_roc_auc": round(float(search.best_score_), 4),
        "tuned_test_scores": final_scores,
        "train_class_balance": y_train.value_counts(normalize=True).round(3).to_dict(),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2, default=str))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axes[0], cmap="Blues")
    axes[0].set_title("Confusion matrix (test set)")
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1])
    axes[1].set_title("ROC curve (test set)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "evaluation.png", dpi=130)
    plt.close()

    names = comparison.keys()
    auc = [comparison[n]["test"]["roc_auc"] for n in names]
    f1 = [comparison[n]["test"]["f1"] for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, auc, 0.4, label="ROC-AUC")
    plt.bar(x + 0.2, f1, 0.4, label="F1")
    plt.xticks(x, list(names), rotation=15)
    plt.ylim(0.6, 1.0)
    plt.legend()
    plt.title("Model comparison (held-out test set)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=130)
    plt.close()
    print("Saved model to models/ and reports to reports/")


if __name__ == "__main__":
    main()
