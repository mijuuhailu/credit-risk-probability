import pandas as pd

df = pd.read_csv("../data/processed_data.csv")

from sklearn.model_selection import train_test_split

X = df.drop("is_high_risk", axis=1)

y = df["is_high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(
    random_state=42,
    max_iter=1000
)

log_model.fit(
    X_train,
    y_train
)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluate_model(
    model,
    X_test,
    y_test
):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(
        X_test
    )[:,1]

    metrics = {
        "accuracy":
            accuracy_score(
                y_test,
                y_pred
            ),

        "precision":
            precision_score(
                y_test,
                y_pred
            ),

        "recall":
            recall_score(
                y_test,
                y_pred
            ),

        "f1":
            f1_score(
                y_test,
                y_pred
            ),

        "roc_auc":
            roc_auc_score(
                y_test,
                y_prob
            )
    }

    return metrics

from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [100,200],
    "max_depth": [5,10,None],
    "min_samples_split": [2,5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    param_grid,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_rf = grid_search.best_estimator_

import mlflow
import mlflow.sklearn

mlflow.set_experiment(
    "Credit_Risk_Model"
)

with mlflow.start_run(
    run_name="LogisticRegression"
):

    log_model.fit(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        log_model,
        X_test,
        y_test
    )

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_metrics(
        metrics
    )

    mlflow.sklearn.log_model(
        log_model,
        "model"
    )

with mlflow.start_run(
    run_name="RandomForest"
):

    best_rf.fit(
        X_train,
        y_train
    )

    metrics = evaluate_model(
        best_rf,
        X_test,
        y_test
    )

    mlflow.log_params(
        grid_search.best_params_
    )

    mlflow.log_metrics(
        metrics
    )

    mlflow.sklearn.log_model(
        best_rf,
        "model"
    )

print(metrics)

import joblib

joblib.dump(
    best_rf,
    "best_model.pkl"
)


