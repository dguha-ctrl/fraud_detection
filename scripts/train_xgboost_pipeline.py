# train_pipeline.py
import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from fraud_utils import FraudPreprocessor

# Load your dataset
df = pd.read_csv("C:/Users/Debrachoubey/PycharmProjects/pythonProject/first_100k_records_europe.csv")

# Split data
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Setup class weight for imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# XGBoost setup
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

# Final pipeline
pipeline = Pipeline([
    ('preprocessor', FraudPreprocessor()),
    ('classifier', search)
])

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Save the full pipeline
joblib.dump(pipeline, "fraud_detection_pipeline.pkl")
print("✅ Pipeline saved as fraud_detection_pipeline.pkl")