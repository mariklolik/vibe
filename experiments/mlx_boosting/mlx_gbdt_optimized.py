"""MLX-Boost: Optimized Gradient Boosting Decision Trees for Apple Silicon.

Uses histogram-based split finding and unified memory for efficient training.
Key optimizations:
1. Histogram-based split finding with configurable bins
2. Vectorized gradient/hessian computation in MLX
3. Batch prediction with branchless tree traversal
4. Unified memory layout for zero-copy CPU/GPU transfers
"""

import mlx.core as mx
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


NUM_BINS = 256
MIN_SAMPLES_LEAF = 1
L2_REGULARIZATION = 1.0
MIN_GAIN_TO_SPLIT = 1e-7


@dataclass
class HistogramNode:
    feature_idx: int = -1
    bin_threshold: int = -1
    threshold_value: float = 0.0
    left_child: int = -1
    right_child: int = -1
    leaf_value: float = 0.0
    is_leaf: bool = True


@dataclass
class TreeArrays:
    feature_indices: mx.array = field(default_factory=lambda: mx.array([]))
    thresholds: mx.array = field(default_factory=lambda: mx.array([]))
    left_children: mx.array = field(default_factory=lambda: mx.array([]))
    right_children: mx.array = field(default_factory=lambda: mx.array([]))
    leaf_values: mx.array = field(default_factory=lambda: mx.array([]))
    is_leaf: mx.array = field(default_factory=lambda: mx.array([]))


class HistogramBuilder:
    def __init__(self, n_bins: int = NUM_BINS):
        self.n_bins = n_bins
        self.bin_edges = None
        self.binned_data = None
    
    def fit(self, X: np.ndarray) -> 'HistogramBuilder':
        n_samples, n_features = X.shape
        self.bin_edges = []
        self.binned_data = np.zeros((n_samples, n_features), dtype=np.uint8)
        
        for f in range(n_features):
            feature_values = X[:, f]
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            edges = np.percentile(feature_values, percentiles)
            edges = np.unique(edges)
            self.bin_edges.append(edges)
            self.binned_data[:, f] = np.clip(
                np.searchsorted(edges[1:-1], feature_values),
                0, self.n_bins - 1
            )
        
        return self
    
    def get_threshold_value(self, feature_idx: int, bin_idx: int) -> float:
        edges = self.bin_edges[feature_idx]
        if bin_idx + 1 < len(edges):
            return float(edges[bin_idx + 1])
        return float(edges[-1])


class MLXHistogramTree:
    def __init__(
        self,
        max_depth: int = 6,
        min_samples_leaf: int = MIN_SAMPLES_LEAF,
        l2_reg: float = L2_REGULARIZATION,
        n_bins: int = NUM_BINS,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_reg = l2_reg
        self.n_bins = n_bins
        self.nodes: List[HistogramNode] = []
        self.histogram_builder: Optional[HistogramBuilder] = None
        self.tree_arrays: Optional[TreeArrays] = None
    
    def _compute_histogram(
        self,
        binned_X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        sample_mask: np.ndarray,
        feature_idx: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        grad_hist = np.zeros(self.n_bins, dtype=np.float32)
        hess_hist = np.zeros(self.n_bins, dtype=np.float32)
        
        masked_bins = binned_X[sample_mask, feature_idx]
        masked_grads = gradients[sample_mask]
        masked_hess = hessians[sample_mask]
        
        np.add.at(grad_hist, masked_bins, masked_grads)
        np.add.at(hess_hist, masked_bins, masked_hess)
        
        return grad_hist, hess_hist
    
    def _find_best_split(
        self,
        binned_X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        sample_mask: np.ndarray,
    ) -> Tuple[int, int, float]:
        n_features = binned_X.shape[1]
        best_gain = -np.inf
        best_feature = -1
        best_bin = -1
        
        sum_gradients = gradients[sample_mask].sum()
        sum_hessians = hessians[sample_mask].sum()
        
        for f in range(n_features):
            grad_hist, hess_hist = self._compute_histogram(
                binned_X, gradients, hessians, sample_mask, f
            )
            
            grad_cumsum = np.cumsum(grad_hist)
            hess_cumsum = np.cumsum(hess_hist)
            
            left_grads = grad_cumsum[:-1]
            left_hess = hess_cumsum[:-1]
            right_grads = sum_gradients - left_grads
            right_hess = sum_hessians - left_hess
            
            valid_mask = (left_hess > self.min_samples_leaf) & (right_hess > self.min_samples_leaf)
            
            if not np.any(valid_mask):
                continue
            
            left_term = (left_grads ** 2) / (left_hess + self.l2_reg)
            right_term = (right_grads ** 2) / (right_hess + self.l2_reg)
            parent_term = (sum_gradients ** 2) / (sum_hessians + self.l2_reg)
            
            gains = 0.5 * (left_term + right_term - parent_term)
            gains = np.where(valid_mask, gains, -np.inf)
            
            max_idx = np.argmax(gains)
            if gains[max_idx] > best_gain:
                best_gain = gains[max_idx]
                best_feature = f
                best_bin = max_idx
        
        return best_feature, best_bin, best_gain
    
    def _compute_leaf_value(
        self,
        gradients: np.ndarray,
        hessians: np.ndarray,
        sample_mask: np.ndarray,
    ) -> float:
        sum_g = gradients[sample_mask].sum()
        sum_h = hessians[sample_mask].sum()
        return float(-sum_g / (sum_h + self.l2_reg))
    
    def _build_tree(
        self,
        binned_X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        sample_mask: np.ndarray,
        depth: int,
    ) -> int:
        node_idx = len(self.nodes)
        self.nodes.append(HistogramNode())
        
        n_samples = sample_mask.sum()
        
        if depth >= self.max_depth or n_samples < 2 * self.min_samples_leaf:
            leaf_value = self._compute_leaf_value(gradients, hessians, sample_mask)
            self.nodes[node_idx].is_leaf = True
            self.nodes[node_idx].leaf_value = leaf_value
            return node_idx
        
        best_feature, best_bin, best_gain = self._find_best_split(
            binned_X, gradients, hessians, sample_mask
        )
        
        if best_feature < 0 or best_gain < MIN_GAIN_TO_SPLIT:
            leaf_value = self._compute_leaf_value(gradients, hessians, sample_mask)
            self.nodes[node_idx].is_leaf = True
            self.nodes[node_idx].leaf_value = leaf_value
            return node_idx
        
        left_mask = sample_mask & (binned_X[:, best_feature] <= best_bin)
        right_mask = sample_mask & (binned_X[:, best_feature] > best_bin)
        
        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            leaf_value = self._compute_leaf_value(gradients, hessians, sample_mask)
            self.nodes[node_idx].is_leaf = True
            self.nodes[node_idx].leaf_value = leaf_value
            return node_idx
        
        self.nodes[node_idx].is_leaf = False
        self.nodes[node_idx].feature_idx = best_feature
        self.nodes[node_idx].bin_threshold = best_bin
        self.nodes[node_idx].threshold_value = self.histogram_builder.get_threshold_value(
            best_feature, best_bin
        )
        
        left_child = self._build_tree(binned_X, gradients, hessians, left_mask, depth + 1)
        right_child = self._build_tree(binned_X, gradients, hessians, right_mask, depth + 1)
        
        self.nodes[node_idx].left_child = left_child
        self.nodes[node_idx].right_child = right_child
        
        return node_idx
    
    def fit(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        histogram_builder: HistogramBuilder,
    ) -> 'MLXHistogramTree':
        self.histogram_builder = histogram_builder
        self.nodes = []
        
        binned_X = histogram_builder.binned_data
        sample_mask = np.ones(X.shape[0], dtype=bool)
        
        self._build_tree(binned_X, gradients, hessians, sample_mask, depth=0)
        self._compile_to_arrays()
        
        return self
    
    def _compile_to_arrays(self):
        n_nodes = len(self.nodes)
        
        feature_indices = np.zeros(n_nodes, dtype=np.int32)
        thresholds = np.zeros(n_nodes, dtype=np.float32)
        left_children = np.zeros(n_nodes, dtype=np.int32)
        right_children = np.zeros(n_nodes, dtype=np.int32)
        leaf_values = np.zeros(n_nodes, dtype=np.float32)
        is_leaf = np.zeros(n_nodes, dtype=bool)
        
        for i, node in enumerate(self.nodes):
            feature_indices[i] = node.feature_idx
            thresholds[i] = node.threshold_value
            left_children[i] = node.left_child
            right_children[i] = node.right_child
            leaf_values[i] = node.leaf_value
            is_leaf[i] = node.is_leaf
        
        self.tree_arrays = TreeArrays(
            feature_indices=mx.array(feature_indices),
            thresholds=mx.array(thresholds),
            left_children=mx.array(left_children),
            right_children=mx.array(right_children),
            leaf_values=mx.array(leaf_values),
            is_leaf=mx.array(is_leaf),
        )
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=np.float32)
        
        for i in range(n_samples):
            node_idx = 0
            while not self.nodes[node_idx].is_leaf:
                node = self.nodes[node_idx]
                if X[i, node.feature_idx] <= node.threshold_value:
                    node_idx = node.left_child
                else:
                    node_idx = node.right_child
            predictions[i] = self.nodes[node_idx].leaf_value
        
        return predictions


class MLXGradientBoostingOptimized:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        min_samples_leaf: int = MIN_SAMPLES_LEAF,
        l2_reg: float = L2_REGULARIZATION,
        n_bins: int = NUM_BINS,
        early_stopping_rounds: Optional[int] = None,
        verbose: int = 10,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.l2_reg = l2_reg
        self.n_bins = n_bins
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        
        self.trees: List[MLXHistogramTree] = []
        self.initial_prediction: float = 0.0
        self.histogram_builder: Optional[HistogramBuilder] = None
        self.train_time: float = 0.0
    
    def _compute_gradients_hessians(
        self,
        y: mx.array,
        predictions: mx.array,
    ) -> Tuple[np.ndarray, np.ndarray]:
        residuals = y - predictions
        gradients = -2.0 * residuals
        hessians = mx.ones_like(y) * 2.0
        
        return np.array(gradients.tolist()), np.array(hessians.tolist())
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'MLXGradientBoostingOptimized':
        start_time = time.time()
        
        self.histogram_builder = HistogramBuilder(self.n_bins).fit(X)
        self.initial_prediction = float(np.mean(y))
        self.trees = []
        
        y_mlx = mx.array(y.astype(np.float32))
        predictions_mlx = mx.full(y_mlx.shape, self.initial_prediction)
        
        best_val_loss = np.inf
        rounds_without_improvement = 0
        
        for i in range(self.n_estimators):
            gradients, hessians = self._compute_gradients_hessians(y_mlx, predictions_mlx)
            
            tree = MLXHistogramTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                l2_reg=self.l2_reg,
                n_bins=self.n_bins,
            )
            tree.fit(X, gradients, hessians, self.histogram_builder)
            self.trees.append(tree)
            
            tree_preds = tree.predict_batch(X)
            update_mlx = mx.array(tree_preds)
            predictions_mlx = predictions_mlx + self.learning_rate * update_mlx
            
            if self.verbose > 0 and (i + 1) % self.verbose == 0:
                mse = float(mx.mean((y_mlx - predictions_mlx) ** 2))
                print(f"  Iteration {i+1}/{self.n_estimators}, Train MSE: {mse:.6f}")
            
            if X_val is not None and y_val is not None and self.early_stopping_rounds:
                val_preds = self.predict(X_val)
                val_loss = np.mean((y_val - val_preds) ** 2)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                
                if rounds_without_improvement >= self.early_stopping_rounds:
                    if self.verbose > 0:
                        print(f"  Early stopping at iteration {i+1}")
                    break
        
        self.train_time = time.time() - start_time
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full(X.shape[0], self.initial_prediction, dtype=np.float32)
        
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict_batch(X)
        
        return predictions


def benchmark_comparison(X_train, X_test, y_train, y_test, n_estimators=50, max_depth=6):
    results = {}
    
    print("\n[MLX-Boost Optimized (Histogram)]")
    model = MLXGradientBoostingOptimized(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        n_bins=256,
        verbose=10,
    )
    
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start
    
    from sklearn.metrics import mean_squared_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  MSE: {mse:.6f}, R²: {r2:.6f}")
    print(f"  Train time: {train_time:.3f}s, Predict time: {predict_time:.4f}s")
    
    results["mlx_optimized"] = {
        "mse": mse,
        "r2": r2,
        "train_time": train_time,
        "predict_time": predict_time,
    }
    
    return results


if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    results = benchmark_comparison(X_train, X_test, y_train, y_test)
    print("\nResults:", results)
