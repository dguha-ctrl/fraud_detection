# Step 1: hybrid_utils.py
# Save this as hybrid_utils.py in your project directory

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class HybridModel(BaseEstimator, ClassifierMixin):
    def __init__(self, xgb_model, iso_model, lof_model, alpha, beta, gamma):
        self.xgb_model = xgb_model
        self.iso_model = iso_model
        self.lof_model = lof_model
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        try:
            # Ensure input is a NumPy array
            if hasattr(X, 'to_numpy'):
                X = X.to_numpy()

            xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
            iso_score = self.iso_model.decision_function(X)
            lof_score = self.lof_model.decision_function(X)

            # Normalize anomaly scores
            iso_score = (iso_score - np.min(iso_score)) / (np.max(iso_score) - np.min(iso_score) + 1e-8)
            lof_score = (lof_score - np.min(lof_score)) / (np.max(lof_score) - np.min(lof_score) + 1e-8)

            hybrid_score = (self.alpha * xgb_prob) + (self.beta * iso_score) + (self.gamma * lof_score)

            print("\n✅ HybridModel.predict_proba():")
            print("XGBoost Probability:", xgb_prob)
            print("Isolation Forest Score:", iso_score)
            print("LOF Score:", lof_score)
            print("Hybrid Score:", hybrid_score)

            return np.vstack([1 - hybrid_score, hybrid_score]).T

        except Exception as e:
            print("❌ ERROR in predict_proba:", str(e))
            return np.array([[0.5, 0.5]])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
