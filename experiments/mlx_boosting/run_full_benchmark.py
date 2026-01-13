#!/usr/bin/env python3
"""Full benchmark: MLX-Boost Optimized vs baselines on multiple datasets."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

N_ESTIMATORS = 100
MAX_DEPTH = 6
LEARNING_RATE = 0.1
N_BINS = 256
RANDOM_STATE = 42


def load_california_housing() -> Tuple[np.ndarray, np.ndarray, str]:
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    return data.data, data.target, "california_housing"


def load_diabetes() -> Tuple[np.ndarray, np.ndarray, str]:
    from sklearn.datasets import load_diabetes
    data = load_diabetes()
    return data.data, data.target, "diabetes"


def load_boston_proxy() -> Tuple[np.ndarray, np.ndarray, str]:
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=506,
        n_features=13,
        n_informative=10,
        noise=10,
        random_state=RANDOM_STATE,
    )
    return X, y, "synthetic_regression"


def load_large_synthetic() -> Tuple[np.ndarray, np.ndarray, str]:
    from sklearn.datasets import make_regression
    X, y = make_regression(
        n_samples=50000,
        n_features=20,
        n_informative=15,
        noise=5,
        random_state=RANDOM_STATE,
    )
    return X, y, "large_synthetic_50k"


def run_sklearn(X_train, X_test, y_train, y_test) -> Dict[str, float]:
    print("  [sklearn]", end=" ", flush=True)
    model = GradientBoostingRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"R²={r2:.4f}, MSE={mse:.4f}, train={train_time:.2f}s")
    
    return {"mse": mse, "r2": r2, "train_time": train_time, "predict_time": predict_time}


def run_xgboost(X_train, X_test, y_train, y_test) -> Optional[Dict[str, float]]:
    try:
        import xgboost as xgb
    except ImportError:
        print("  [xgboost] Not installed, skipping")
        return None
    
    print("  [xgboost]", end=" ", flush=True)
    model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"R²={r2:.4f}, MSE={mse:.4f}, train={train_time:.2f}s")
    
    return {"mse": mse, "r2": r2, "train_time": train_time, "predict_time": predict_time}


def run_lightgbm(X_train, X_test, y_train, y_test) -> Optional[Dict[str, float]]:
    try:
        import lightgbm as lgb
    except ImportError:
        print("  [lightgbm] Not installed, skipping")
        return None
    
    print("  [lightgbm]", end=" ", flush=True)
    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=RANDOM_STATE,
        verbosity=-1,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"R²={r2:.4f}, MSE={mse:.4f}, train={train_time:.2f}s")
    
    return {"mse": mse, "r2": r2, "train_time": train_time, "predict_time": predict_time}


def run_mlx_original(X_train, X_test, y_train, y_test) -> Optional[Dict[str, float]]:
    try:
        import mlx.core as mx
        from mlx_gbdt import MLXGradientBoosting
    except ImportError as e:
        print(f"  [mlx_original] Not available: {e}")
        return None
    
    print("  [mlx_original]", end=" ", flush=True)
    model = MLXGradientBoosting(
        n_estimators=min(N_ESTIMATORS, 50),
        max_depth=min(MAX_DEPTH, 4),
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
    print(f"R²={r2:.4f}, MSE={mse:.4f}, train={train_time:.2f}s")
    
    return {"mse": mse, "r2": r2, "train_time": train_time, "predict_time": predict_time}


def run_mlx_optimized(X_train, X_test, y_train, y_test) -> Optional[Dict[str, float]]:
    try:
        import mlx.core as mx
        from mlx_gbdt_optimized import MLXGradientBoostingOptimized
    except ImportError as e:
        print(f"  [mlx_optimized] Not available: {e}")
        return None
    
    print("  [mlx_optimized]", end=" ", flush=True)
    model = MLXGradientBoostingOptimized(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        n_bins=N_BINS,
        verbose=0,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"R²={r2:.4f}, MSE={mse:.4f}, train={train_time:.2f}s")
    
    return {"mse": mse, "r2": r2, "train_time": train_time, "predict_time": predict_time}


def run_ablation_bins(X_train, X_test, y_train, y_test) -> Dict[str, Dict[str, float]]:
    try:
        from mlx_gbdt_optimized import MLXGradientBoostingOptimized
    except ImportError:
        return {}
    
    print("\n  Ablation: Number of histogram bins")
    results = {}
    
    for n_bins in [32, 64, 128, 256, 512]:
        print(f"    bins={n_bins}", end=" ", flush=True)
        model = MLXGradientBoostingOptimized(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE,
            n_bins=n_bins,
            verbose=0,
        )
        
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"R²={r2:.4f}, train={train_time:.2f}s")
        results[f"bins_{n_bins}"] = {"r2": r2, "mse": mse, "train_time": train_time}
    
    return results


def run_ablation_depth(X_train, X_test, y_train, y_test) -> Dict[str, Dict[str, float]]:
    try:
        from mlx_gbdt_optimized import MLXGradientBoostingOptimized
    except ImportError:
        return {}
    
    print("\n  Ablation: Tree depth")
    results = {}
    
    for depth in [3, 4, 5, 6, 7, 8]:
        print(f"    depth={depth}", end=" ", flush=True)
        model = MLXGradientBoostingOptimized(
            n_estimators=N_ESTIMATORS,
            max_depth=depth,
            learning_rate=LEARNING_RATE,
            n_bins=N_BINS,
            verbose=0,
        )
        
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"R²={r2:.4f}, train={train_time:.2f}s")
        results[f"depth_{depth}"] = {"r2": r2, "mse": mse, "train_time": train_time}
    
    return results


def run_dataset_benchmark(dataset_loader) -> Dict[str, Any]:
    X, y, dataset_name = dataset_loader()
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    results = {"dataset": dataset_name, "n_train": len(X_train), "n_test": len(X_test)}
    
    results["sklearn"] = run_sklearn(X_train, X_test, y_train, y_test)
    results["xgboost"] = run_xgboost(X_train, X_test, y_train, y_test)
    results["lightgbm"] = run_lightgbm(X_train, X_test, y_train, y_test)
    results["mlx_original"] = run_mlx_original(X_train, X_test, y_train, y_test)
    results["mlx_optimized"] = run_mlx_optimized(X_train, X_test, y_train, y_test)
    
    if dataset_name == "california_housing":
        results["ablation_bins"] = run_ablation_bins(X_train, X_test, y_train, y_test)
        results["ablation_depth"] = run_ablation_depth(X_train, X_test, y_train, y_test)
    
    results = {k: v for k, v in results.items() if v is not None}
    return results


def print_summary(all_results: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("SUMMARY: R² Scores Across Datasets")
    print("=" * 80)
    
    methods = ["sklearn", "xgboost", "lightgbm", "mlx_original", "mlx_optimized"]
    
    print(f"{'Dataset':<25}", end="")
    for method in methods:
        print(f"{method:>12}", end="")
    print()
    print("-" * 85)
    
    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<25}", end="")
        for method in methods:
            if method in results and results[method]:
                print(f"{results[method]['r2']:>12.4f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()
    
    print("\n" + "=" * 80)
    print("SUMMARY: Training Time (seconds)")
    print("=" * 80)
    
    print(f"{'Dataset':<25}", end="")
    for method in methods:
        print(f"{method:>12}", end="")
    print()
    print("-" * 85)
    
    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<25}", end="")
        for method in methods:
            if method in results and results[method]:
                print(f"{results[method]['train_time']:>12.2f}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()


def main():
    print("=" * 80)
    print("MLX-Boost Full Benchmark")
    print(f"Config: n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, lr={LEARNING_RATE}")
    print("=" * 80)
    
    datasets = [
        load_california_housing,
        load_diabetes,
        load_boston_proxy,
        load_large_synthetic,
    ]
    
    all_results = {}
    
    for loader in datasets:
        try:
            results = run_dataset_benchmark(loader)
            all_results[results["dataset"]] = results
        except Exception as e:
            print(f"Error running {loader.__name__}: {e}")
    
    print_summary(all_results)
    
    results_file = RESULTS_DIR / "full_benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
