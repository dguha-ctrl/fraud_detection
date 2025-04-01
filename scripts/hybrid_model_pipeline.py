import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, ClassifierMixin
from hyperopt import fmin, tpe, hp, Trials
from fraud_utils import FraudPreprocessor
import pyodbc

print("\U0001F195 Loading data...")
df = pd.read_csv("C:/Users/Debrachoubey/PycharmProjects/pythonProject/first_100k_records_europe.csv")
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Preprocess first
preprocessor = FraudPreprocessor()
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print("\U0001F195 Tuning XGBoost...")
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=42
)

param_dist = {
    'n_estimators': [100, 300],
    'max_depth': [6, 8],
    'learning_rate': [0.05, 0.1],
    'min_child_weight': [1, 3],
    'gamma': [0, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring='f1',
    cv=2,
    verbose=2,
    n_jobs=-1,
    random_state=42
)
search.fit(X_train_proc, y_train)
best_xgb_model = search.best_estimator_

print("\U0001F195 Fitting Isolation Forest...")
iso_forest = IsolationForest(random_state=42)
iso_forest.fit(X_train_proc)
iso_scores = iso_forest.decision_function(X_test_proc)

print("\U0001F195 Fitting LOF...")
def objective_lof(params):
    n_neighbors = int(params['n_neighbors'])
    contamination = float(params['contamination'])
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, novelty=True)
    lof.fit(X_train_proc)
    lof_labels = np.where(lof.predict(X_test_proc) == -1, 1, 0)
    return -roc_auc_score(y_test, lof_labels)

space_lof = {
    'n_neighbors': hp.quniform('n_neighbors', 5, 30, 5),
    'contamination': hp.uniform('contamination', 0.005, 0.02)
}

best_lof_params = fmin(fn=objective_lof, space=space_lof, algo=tpe.suggest, max_evals=5, trials=Trials())
best_lof = LocalOutlierFactor(
    n_neighbors=int(best_lof_params['n_neighbors']),
    contamination=float(best_lof_params['contamination']),
    novelty=True
)
best_lof.fit(X_train_proc)
lof_scores = best_lof.decision_function(X_test_proc)

print("\U0001F195 Optimizing Hybrid Weights...")
y_pred_proba = best_xgb_model.predict_proba(X_test_proc)[:, 1]
iso_scores_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
lof_scores_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min())

def objective_hybrid(params):
    alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']
    hybrid_score = (alpha * y_pred_proba) + (beta * iso_scores_norm) + (gamma * lof_scores_norm)
    y_pred = (hybrid_score >= 0.5).astype(int)
    return -roc_auc_score(y_test, y_pred)

space_hybrid = {
    'alpha': hp.uniform('alpha', 0.5, 0.9),
    'beta': hp.uniform('beta', 0.1, 0.3),
    'gamma': hp.uniform('gamma', 0.05, 0.2)
}

best_hybrid_params = fmin(fn=objective_hybrid, space=space_hybrid, algo=tpe.suggest, max_evals=10, trials=Trials())
alpha = best_hybrid_params['alpha']
beta = best_hybrid_params['beta']
gamma = best_hybrid_params['gamma']
print(f"\n✅ Best weights: alpha={alpha:.3f}, beta={beta:.3f}, gamma={gamma:.3f}")

# Define HybridModel class
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
        xgb_prob = self.xgb_model.predict_proba(X)[:, 1]
        iso_score = (self.iso_model.decision_function(X) - np.min(self.iso_model.decision_function(X))) / (np.ptp(self.iso_model.decision_function(X)))
        lof_score = (self.lof_model.decision_function(X) - np.min(self.lof_model.decision_function(X))) / (np.ptp(self.lof_model.decision_function(X)))
        hybrid_score = (self.alpha * xgb_prob) + (self.beta * iso_score) + (self.gamma * lof_score)
        return np.vstack([1 - hybrid_score, hybrid_score]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# Wrap in final pipeline
hybrid_model = HybridModel(best_xgb_model, iso_forest, best_lof, alpha, beta, gamma)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', hybrid_model)
])
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "hybrid_fraud_detection_pipeline.pkl")
print("✅ Saved hybrid_fraud_detection_pipeline.pkl")

# Evaluate and store metrics
y_pred = pipeline.predict(X_test)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=DESKTOP-EPOVRA0\\SQLEXPRESS;'
    'DATABASE=credit_card;'
    'Trusted_Connection=yes;'
)
cursor = conn.cursor()
cursor.execute("""
    INSERT INTO model_performance (model_name, recall, precision, f1_score, accuracy)
    VALUES (?, ?, ?, ?, ?)
""", ('hybrid_pipeline_v1', recall, precision, f1, accuracy))
conn.commit()
conn.close()
print("📊 Hybrid model performance logged to SQL Server")
