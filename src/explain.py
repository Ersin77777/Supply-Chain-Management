from __future__ import annotations
import shap


def shap_summary_bar(model, X, feature_names=None, title: str = "") -> None:
    """
    TreeExplainer summary bar plot.
    For tree-based models: RF, XGBoost, LightGBM, CatBoost.
    """
    print(title)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        plot_type="bar",
        show=True,
    )
