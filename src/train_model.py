import numpy as np
import pandas as pd
import os
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ========== 1. Load Data ==========
X = np.load("data/processed/processed.npy")
y = np.load("data/processed/target.npy")

# ========== 2. Train-Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========== 3. Setup MLflow ==========
mlflow.set_experiment("Credit Scoring Model")

# ========== 4. Define Models & Parameters ==========
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

param_grids = {
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10]
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    }
}

# ========== 5. Train Models ==========
best_model = None
best_score = 0
best_name = ""

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        print(f"\n🔍 Training {name}...")

        # Train model using GridSearchCV
        grid_search = GridSearchCV(
            model, param_grids[name], cv=3, scoring="f1", n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        final_model = grid_search.best_estimator_
        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]

        # Evaluate model
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        # Log to MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc
        })

        # Log model with input example to avoid warnings
        input_example = pd.DataFrame(X_test[:1])  # 1 row sample
        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            input_example=input_example
        )

        # Track best model
        if f1 > best_score:
            best_model = final_model
            best_score = f1
            best_name = name

# ========== 6. Save Best Model Locally ==========
os.makedirs("models", exist_ok=True)
model_path = f"models/{best_name}_best_model.pkl"
joblib.dump(best_model, model_path)

print(f"\n✅ Best model: {best_name} saved to {model_path} with F1 Score: {best_score:.4f}")
