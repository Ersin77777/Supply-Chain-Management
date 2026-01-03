from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# ---------- Data filters ----------
def filter_train_data_for_delivery(df: pd.DataFrame) -> pd.DataFrame:
    """
    For delivery_status model:
    - drop invalid, extreme_date, open orders
    - require delivery_status non-null
    """
    return df[(df["is_invalid"] == 0) & (df["is_extreme_date"] == 0) & (df["is_open_order"] == 0)].dropna(
        subset=["delivery_status"]
    )


def filter_train_data_for_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """
    For quantity_status model:
    - drop invalid, extreme_date, open orders
    - require quantity_status non-null
    """
    return df[(df["is_invalid"] == 0) & (df["is_extreme_date"] == 0) & (df["is_open_order"] == 0)].dropna(
        subset=["quantity_status"]
    )


# ---------- Evaluation ----------
def evaluate_model(model, X, y, stratified: bool = True) -> float:
    cv = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if stratified
        else KFold(n_splits=5, shuffle=True, random_state=42)
    )
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    return float(np.mean(scores))


def model_report(model, X, y, title: str = "") -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n" + "=" * 70)
    print(f"{title} - Classification Report")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("=" * 70)


@dataclass
class BestModelResult:
    name: str
    model: object
    cv_macro_f1: float


def evaluate_all_models(models: dict, X, y, company: str, taskname: str, stratified: bool = True) -> BestModelResult:
    print(f"\n====== {company} - {taskname} (5-Fold CV macro-F1) ======")
    best_name, best_model, best_score = None, None, -1.0

    for name, model in models.items():
        f1 = evaluate_model(model, X, y, stratified=stratified)
        print(f"{name}: {f1:.3f}")
        if f1 > best_score:
            best_score = f1
            best_name = name
            best_model = model

    assert best_name is not None and best_model is not None
    print(f"\n>> Best Model for {company}: {best_name} (macro-F1={best_score:.3f})")
    model_report(best_model, X, y, title=f"{company} {taskname} ({best_name})")

    return BestModelResult(name=best_name, model=best_model, cv_macro_f1=best_score)


# ---------- Persistence ----------
def save_model(model, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)


def load_model(path: str | Path):
    return joblib.load(path)
