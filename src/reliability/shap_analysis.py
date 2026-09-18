import shap
import matplotlib.pyplot as plt

def generate_shap_summary(xgb_model, X, feature_names=None, output_path='shap_summary.png'):
    """
    Generates and saves a SHAP summary plot for the XGBoost detector.
    """
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return shap_values
