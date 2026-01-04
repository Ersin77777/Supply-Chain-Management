# from __future__ import annotations
# from dataclasses import dataclass
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import joblib

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
# from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# # ---------- Data filters ----------
# def filter_train_data_for_delivery(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     For delivery_status model:
#     - drop invalid, extreme_date, open orders
#     - require delivery_status non-null
#     """
#     return df[(df["is_invalid"] == 0) & (df["is_extreme_date"] == 0) & (df["is_open_order"] == 0)].dropna(
#         subset=["delivery_status"]
#     )


# def filter_train_data_for_quantity(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     For quantity_status model:
#     - drop invalid, extreme_date, open orders
#     - require quantity_status non-null
#     """
#     return df[(df["is_invalid"] == 0) & (df["is_extreme_date"] == 0) & (df["is_open_order"] == 0)].dropna(
#         subset=["quantity_status"]
#     )


# # ---------- Evaluation ----------
# def evaluate_model(model, X, y, stratified: bool = True) -> float:
#     cv = (
#         StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         if stratified
#         else KFold(n_splits=5, shuffle=True, random_state=42)
#     )

#     scores = cross_val_score(
#         estimator=model,
#         X=X,
#         y=y,                      
#         cv=cv,
#         scoring="f1_macro",
#         n_jobs=-1,
#         error_score="raise"      
#     )
#     return scores.mean()



# def model_report(model, X, y, title: str = "") -> None:
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     print("\n" + "=" * 70)
#     print(f"{title} - Classification Report")
#     print(classification_report(y_test, y_pred, digits=3))
#     print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
#     print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
#     print("=" * 70)


# @dataclass
# class BestModelResult:
#     name: str
#     model: object
#     cv_macro_f1: float


# def evaluate_all_models(
#     models: dict,
#     X,
#     y,
#     company: str,
#     taskname: str,
#     stratified: bool = True,
# ):
#     print(f"\n====== {company} - {taskname} (5-Fold CV macro-F1) ======")

#     # ✅ 强制：这里统一做 label encoding
#     le = LabelEncoder()
#     y_enc = le.fit_transform(y)

#     best_name, best_model, best_score = None, None, -1.0

#     for name, model in models.items():
#         f1 = evaluate_model(
#             model=model,
#             X=X,
#             y=y_enc,              # 🔑 只传数值标签
#             stratified=stratified
#         )
#         print(f"{name}: {f1:.3f}")

#         if f1 > best_score:
#             best_name = name
#             best_model = model
#             best_score = f1

#     print(f"\n>> Best Model for {company} {taskname}: {best_name} (F1={best_score:.3f})")

#     return {
#         "best_model_name": best_name,
#         "best_model": best_model,
#         "label_encoder": le,
#         "best_score": best_score,
#     }



# # ---------- Persistence ----------
# def save_model(model, out_path: str | Path) -> None:
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     joblib.dump(model, out_path)


# def load_model(path: str | Path):
#     return joblib.load(path)




from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import collections

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ---------- Data filters ----------
def filter_train_data_for_delivery(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["is_invalid"] == 0) & (df["is_extreme_date"] == 0) & (df["is_open_order"] == 0)].dropna(
        subset=["delivery_status"]
    )

def filter_train_data_for_quantity(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["is_invalid"] == 0) & (df["is_extreme_date"] == 0) & (df["is_open_order"] == 0)].dropna(
        subset=["quantity_status"]
    )


# ---------- Helper: Robust SMOTE ----------
def get_robust_smote(y_train):
    """
    智能获取 SMOTE 对象。
    如果最小类别的样本数非常少（例如少于6个），
    SMOTE 默认的 k_neighbors=5 会报错。
    此函数自动将 k_neighbors 调低，确保不会报错。
    """
    # 统计各类别样本数
    counts = collections.Counter(y_train)
    min_samples = min(counts.values())
    
    # 动态调整 k_neighbors
    # 如果最少类别只有 2 个样本，k只能设为 1
    # 如果有 6 个以上，k 设为默认 5
    k = min(min_samples - 1, 5)
    if k < 1: k = 1
    
    return SMOTE(random_state=42, k_neighbors=k)


# ---------- Evaluation ----------
def evaluate_model(model, X, y, stratified: bool = True, use_smote: bool = False) -> float:
    cv = (
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if stratified
        else KFold(n_splits=5, shuffle=True, random_state=42)
    )

    if use_smote:
        # 在 Pipeline 中无法动态获取 y 来调整 k_neighbors，
        # 所以这里我们使用一个保守的 SMOTE 配置 (k_neighbors=1) 
        # 或者你也可以编写自定义采样器，但 k=1 对于极少样本是安全的。
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=1)), 
            ('model', model)
        ])
        estimator = pipeline
    else:
        estimator = model

    scores = cross_val_score(
        estimator=estimator,
        X=X,
        y=y,                      
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        error_score="raise"      
    )
    return scores.mean()


def model_report(model, X, y, label_encoder: LabelEncoder = None, title: str = "", use_smote: bool = False) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if use_smote:
        # 在这里我们可以根据切分后的 y_train 动态调整 SMOTE
        sm = get_robust_smote(y_train)
        print(f"Applying SMOTE with k_neighbors={sm.k_neighbors}...")
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        model.fit(X_train_res, y_train_res)
    else:
        model.fit(X_train, y_train)
        
    y_pred = model.predict(X_test)

    if label_encoder is not None:
        y_test_labels = label_encoder.inverse_transform(y_test)
        y_pred_labels = label_encoder.inverse_transform(y_pred)
    else:
        y_test_labels = y_test
        y_pred_labels = y_pred

    print("\n" + "=" * 70)
    print(f"{title} - Classification Report")
    print(classification_report(y_test_labels, y_pred_labels, digits=3))
    print(f"Accuracy: {accuracy_score(y_test_labels, y_pred_labels):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test_labels, y_pred_labels))
    print("=" * 70)


def evaluate_all_models(
    models: dict,
    X,
    y,
    company: str,
    taskname: str,
    stratified: bool = True,
    use_smote: bool = False
):
    print(f"\n====== {company} - {taskname} (5-Fold CV macro-F1) [SMOTE={use_smote}] ======")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    best_name = None
    best_model = None
    best_score = -1.0

    for name, model in models.items():
        try:
            f1 = evaluate_model(
                model=model,
                X=X,
                y=y_enc,
                stratified=stratified,
                use_smote=use_smote
            )
            print(f"{name}: {f1:.3f}")

            if f1 > best_score:
                best_name = name
                best_model = model
                best_score = f1
        except Exception as e:
            print(f"{name}: Failed - {str(e)}")

    if best_model is not None:
        print(f"\n>> Best Model for {company} {taskname}: {best_name} (F1={best_score:.3f})")
        model_report(
            best_model, 
            X, 
            y_enc, 
            label_encoder=le, 
            title=f"{company} {taskname} ({best_name})",
            use_smote=use_smote
        )
        return {
            "best_model_name": best_name,
            "best_model": best_model,
            "label_encoder": le,
            "best_score": best_score,
        }
    else:
        print("\n>> No models were successfully evaluated.")
        return None

# ---------- Persistence ----------
def save_model(model, out_path: Union[str, Path]) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)

def load_model(path: Union[str, Path]):
    return joblib.load(path)