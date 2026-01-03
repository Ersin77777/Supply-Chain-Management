from __future__ import annotations
import pandas as pd


def predict_single_order(model, input_dict: dict, features: list[str], preprocessor=None):
    """
    Predict one order.
    - input_dict: keys should include required features
    - preprocessor: optional, e.g., OneHotEncoder ColumnTransformer for quantity model
    """
    X = pd.DataFrame([input_dict])[features].copy().fillna(0)
    if preprocessor is not None:
        X = preprocessor.transform(X)
    return model.predict(X)[0]