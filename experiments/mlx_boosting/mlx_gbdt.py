"""MLX-accelerated Gradient Boosting Decision Tree implementation.

Uses MLX for vectorized operations on Apple Silicon.
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeNode:
    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: Optional[float] = None
    
    def is_leaf(self) -> bool:
        return self.value is not None


class MLXDecisionTree:
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def _variance_reduction(self, y: np.ndarray, left_mask: np.ndarray) -> float:
        if np.sum(left_mask) < self.min_samples_split or np.sum(~left_mask) < self.min_samples_split:
            return -np.inf
        
        left_y = y[left_mask]
        right_y = y[~left_mask]
        
        n = len(y)
        n_left = len(left_y)
        n_right = len(right_y)
        
        var_before = np.var(y)
        var_after = (n_left * np.var(left_y) + n_right * np.var(right_y)) / n
        
        return var_before - var_after
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            thresholds = np.percentile(feature_values, [10, 25, 50, 75, 90])
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                gain = self._variance_reduction(y, left_mask)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> TreeNode:
        n_samples = len(y)
        
        if depth >= self.max_depth or n_samples < self.min_samples_split:
            return TreeNode(value=float(np.mean(y)))
        
        feature_idx, threshold, gain = self._best_split(X, y)
        
        if feature_idx is None or gain <= 0:
            return TreeNode(value=float(np.mean(y)))
        
        left_mask = X[:, feature_idx] <= threshold
        
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        
        return TreeNode(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_node,
            right=right_node,
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x: np.ndarray, node: TreeNode) -> float:
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(x, self.root) for x in X])


class MLXGradientBoosting:
    """Gradient Boosting with MLX-accelerated residual computation."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.initial_prediction = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.initial_prediction = float(np.mean(y))
        
        y_mlx = mx.array(y.astype(np.float32))
        predictions_mlx = mx.full(y_mlx.shape, self.initial_prediction)
        
        for i in range(self.n_estimators):
            residuals_mlx = y_mlx - predictions_mlx
            residuals = np.array(residuals_mlx.tolist())
            
            tree = MLXDecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            update = tree.predict(X)
            update_mlx = mx.array(update.astype(np.float32))
            predictions_mlx = predictions_mlx + self.learning_rate * update_mlx
            
            if (i + 1) % 10 == 0:
                mse = float(mx.mean((y_mlx - predictions_mlx) ** 2))
                print(f"  Iteration {i+1}/{self.n_estimators}, MSE: {mse:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.trees:
            predictions = predictions + self.learning_rate * tree.predict(X)
        
        return predictions


def numpy_to_mlx(arr: np.ndarray) -> mx.array:
    return mx.array(arr.astype(np.float32))


def mlx_to_numpy(arr: mx.array) -> np.ndarray:
    return np.array(arr.tolist())
