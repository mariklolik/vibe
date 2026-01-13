#!/usr/bin/env python3
"""Run boosting comparison experiment: MLX vs XGBoost vs LightGBM vs scikit-learn."""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

N_ESTIMATORS = 50
MAX_DEPTH = 4
LEARNING_RATE = 0.1


def run_sklearn_gbdt(X_train, X_test, y_train, y_test, task="regression"):
    print("\n[sklearn GradientBoosting]")
    
    if task == "regression":
        model = GradientBoostingRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=42,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=42,
        )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    if task == "regression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
        return {"mse": mse, "r2": r2, "train_time": train_time, "predict_time": predict_time}
    else:
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}")
        return {"accuracy": acc, "train_time": train_time, "predict_time": predict_time}


def run_xgboost(X_train, X_test, y_train, y_test, task="regression"):
    print("\n[XGBoost]")
    try:
        import xgboost as xgb
    except ImportError:
        print("  XGBoost not installed, skipping")
        return None
    
    if task == "regression":
        model = xgb.XGBRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=42,
            verbosity=0,
        )
    else:
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=42,
            verbosity=0,
        )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    if task == "regression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
        return {"mse": mse, "r2": r2, "train_time": train_time, "predict_time": predict_time}
    else:
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}")
        return {"accuracy": acc, "train_time": train_time, "predict_time": predict_time}


def run_lightgbm(X_train, X_test, y_train, y_test, task="regression"):
    print("\n[LightGBM]")
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not installed, skipping")
        return None
    
    if task == "regression":
        model = lgb.LGBMRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=42,
            verbosity=-1,
        )
    else:
        model = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            random_state=42,
            verbosity=-1,
        )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    if task == "regression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
        return {"mse": mse, "r2": r2, "train_time": train_time, "predict_time": predict_time}
    else:
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.4f}")
        return {"accuracy": acc, "train_time": train_time, "predict_time": predict_time}


def run_mlx_gbdt(X_train, X_test, y_train, y_test, task="regression"):
    print("\n[MLX GradientBoosting]")
    try:
        import mlx.core as mx
        from mlx_gbdt import MLXGradientBoosting
    except ImportError as e:
        print(f"  MLX not available: {e}")
        return None
    
    if task != "regression":
        print("  MLX implementation only supports regression, skipping")
        return None
    
    model = MLXGradientBoosting(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  MSE: {mse:.4f}, R²: {r2:.4f}")
    
    return {"mse": mse, "r2": r2, "train_time": train_time, "predict_time": predict_time}


def run_regression_experiment():
    print("=" * 60)
    print("REGRESSION EXPERIMENT: California Housing")
    print("=" * 60)
    
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    results = {}
    
    results["sklearn"] = run_sklearn_gbdt(X_train, X_test, y_train, y_test, "regression")
    results["xgboost"] = run_xgboost(X_train, X_test, y_train, y_test, "regression")
    results["lightgbm"] = run_lightgbm(X_train, X_test, y_train, y_test, "regression")
    results["mlx"] = run_mlx_gbdt(X_train, X_test, y_train, y_test, "regression")
    
    results = {k: v for k, v in results.items() if v is not None}
    
    return results


def run_classification_experiment():
    print("\n" + "=" * 60)
    print("CLASSIFICATION EXPERIMENT: Breast Cancer")
    print("=" * 60)
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    results = {}
    
    results["sklearn"] = run_sklearn_gbdt(X_train, X_test, y_train, y_test, "classification")
    results["xgboost"] = run_xgboost(X_train, X_test, y_train, y_test, "classification")
    results["lightgbm"] = run_lightgbm(X_train, X_test, y_train, y_test, "classification")
    
    results = {k: v for k, v in results.items() if v is not None}
    
    return results


def main():
    print("Gradient Boosting Comparison Experiment")
    print(f"Config: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, lr={LEARNING_RATE}")
    
    regression_results = run_regression_experiment()
    classification_results = run_classification_experiment()
    
    all_results = {
        "config": {
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
        },
        "regression": regression_results,
        "classification": classification_results,
    }
    
    results_file = RESULTS_DIR / "boosting_comparison.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nRegression (California Housing):")
    print(f"{'Method':<12} {'R²':>8} {'MSE':>10} {'Train(s)':>10} {'Pred(s)':>10}")
    print("-" * 52)
    for method, res in regression_results.items():
        print(f"{method:<12} {res['r2']:>8.4f} {res['mse']:>10.4f} {res['train_time']:>10.3f} {res['predict_time']:>10.4f}")
    
    print("\nClassification (Breast Cancer):")
    print(f"{'Method':<12} {'Accuracy':>10} {'Train(s)':>10} {'Pred(s)':>10}")
    print("-" * 44)
    for method, res in classification_results.items():
        print(f"{method:<12} {res['accuracy']:>10.4f} {res['train_time']:>10.3f} {res['predict_time']:>10.4f}")
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
