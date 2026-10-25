"""Binary CART with stable sorting and cumulative class counts; NumPy only."""

from dataclasses import dataclass
import math
from numbers import Integral, Real

import numpy as np


TREE_SCHEMA = "catalogmatch.tree.v1"


@dataclass
class TreeNode:
    sample_count: int
    positive_count: int
    positive_probability: float
    feature_index: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None


@dataclass
class _Split:
    feature_index: int
    threshold: float
    weighted_impurity: float
    left_count: int


def _gini(positive_count, sample_count):
    """Counts may be scalars or arrays; callers guarantee nonempty populations."""
    probability = positive_count / sample_count
    return 2.0 * probability * (1.0 - probability)


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate_X(X, n_features: int | None = None) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] == 0 or X.dtype.kind not in "iuf":
        raise ValueError("X must be a two-dimensional real numerical matrix with at least one feature")
    with np.errstate(over="ignore", invalid="ignore"):
        X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite float64 feature values")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X must have {n_features} features, got {X.shape[1]}")
    return X


def _validate_training(X, y) -> tuple[np.ndarray, np.ndarray]:
    X = _validate_X(X)
    y = np.asarray(y)
    if X.shape[0] == 0:
        raise ValueError("Training requires at least one sample")
    if y.ndim != 1 or y.dtype.kind not in "biuf" or not np.all((y == 0) | (y == 1)):
        raise ValueError("y must be a one-dimensional binary array containing only 0 and 1")
    if len(y) != len(X):
        raise ValueError("X and y must have the same number of rows")
    return X, np.asarray(y, dtype=np.int64)


def _validate_indices(indices, n_samples: int) -> np.ndarray:
    if indices is None:
        return np.arange(n_samples, dtype=np.int64)
    indices = np.asarray(indices)
    if (indices.ndim != 1 or indices.size == 0 or indices.dtype.kind not in "iu"
            or np.any(indices < 0) or np.any(indices >= n_samples)):
        raise ValueError("sample_indices must be a nonempty one-dimensional array of valid integer row indices")
    return np.asarray(indices, dtype=np.int64)


def _threshold(lo: float, hi: float) -> float:
    midpoint = lo / 2.0 + hi / 2.0
    # Rounding can produce hi, especially for adjacent floats or subnormals.
    return midpoint if lo <= midpoint < hi else lo


def _best_split(X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                features: np.ndarray, min_samples_leaf: int) -> _Split | None:
    """Score distinct boundaries in O(s*m*log(m)), without per-boundary partitions.

    Strictly smaller computed impurity wins. Exact ties prefer lower feature,
    then earlier sorted boundary. Only the final winning split is partitioned.
    """
    sample_count = len(indices)
    best = None
    for feature in np.sort(features):
        values = X[indices, feature]
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        positives = np.cumsum(y[indices[order]], dtype=np.int64)
        # Each boundary is identified by the number of occurrences on its left.
        left_counts = np.arange(min_samples_leaf, sample_count - min_samples_leaf + 1)
        distinct = sorted_values[left_counts - 1] < sorted_values[left_counts]
        left_counts = left_counts[distinct]
        if left_counts.size == 0:
            continue
        right_counts = sample_count - left_counts
        left_positives = positives[left_counts - 1]
        right_positives = positives[-1] - left_positives
        impurities = (left_counts / sample_count * _gini(left_positives, left_counts)
                      + right_counts / sample_count * _gini(right_positives, right_counts))
        position = int(np.argmin(impurities))  # First minimum is the earliest boundary.
        impurity = float(impurities[position])
        if best is None or impurity < best.weighted_impurity:
            left_count = int(left_counts[position])
            threshold = _threshold(float(sorted_values[left_count - 1]),
                                   float(sorted_values[left_count]))
            best = _Split(int(feature), threshold, impurity, left_count)
    return best


def _require_keys(value, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} has missing, unknown, or malformed fields")


def _finite_number(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


class DecisionTreeClassifier:
    """Binary CART; root depth is zero, and probability ties predict class zero.

    max_depth is None or an integer >= 0 (default 8); min_samples_split >= 2;
    min_samples_leaf >= 1; max_features is None, 'sqrt', or an integer in [1,d].
    random_state is a nonnegative integer. Real inputs are converted to float64.
    """

    def __init__(self, max_depth: int | None = 8, min_samples_split: int = 2,
                 min_samples_leaf: int = 2, max_features: int | str | None = None,
                 random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self._validate_parameters()
        self.root_ = None
        self.n_features_in_ = None

    def _validate_parameters(self) -> None:
        if self.max_depth is not None:
            self.max_depth = _integer(self.max_depth, "max_depth", 0)
        self.min_samples_split = _integer(self.min_samples_split, "min_samples_split", 2)
        self.min_samples_leaf = _integer(self.min_samples_leaf, "min_samples_leaf", 1)
        self.random_state = _integer(self.random_state, "random_state", 0)
        if self.max_features is not None:
            if isinstance(self.max_features, str):
                if self.max_features != "sqrt":
                    raise ValueError("max_features must be None, 'sqrt', or a positive integer")
            else:
                self.max_features = _integer(self.max_features, "max_features", 1)

    def _feature_count(self, n_features: int) -> int:
        if self.max_features is None:
            return n_features
        if self.max_features == "sqrt":
            return max(1, math.isqrt(n_features))
        if self.max_features > n_features:
            raise ValueError("max_features cannot exceed the number of features")
        return self.max_features

    def _select_features(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        count = self._feature_count(n_features)
        if count == n_features:
            return np.arange(n_features)
        return np.sort(rng.choice(n_features, size=count, replace=False))

    def fit(self, X, y, sample_indices=None) -> "DecisionTreeClassifier":
        self._validate_parameters()
        X, y = _validate_training(X, y)
        self._feature_count(X.shape[1])
        indices = _validate_indices(sample_indices, len(X))
        return self._fit_with_indices(X, y, indices, np.random.default_rng(self.random_state))

    def _fit_with_indices(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                          rng: np.random.Generator) -> "DecisionTreeClassifier":
        """Internal entry: callers validate once; forest continues its bootstrap RNG."""
        root = self._build_tree(X, y, indices, rng, depth=0)
        self.root_ = root
        self.n_features_in_ = X.shape[1]
        return self

    def _build_tree(self, X, y, indices, rng, depth: int) -> TreeNode:
        sample_count = len(indices)
        positive_count = int(np.sum(y[indices], dtype=np.int64))
        node = TreeNode(sample_count, positive_count, positive_count / sample_count)
        if (positive_count in (0, sample_count)
                or (self.max_depth is not None and depth >= self.max_depth)
                or sample_count < self.min_samples_split
                or sample_count < 2 * self.min_samples_leaf):
            return node
        features = self._select_features(X.shape[1], rng)
        split = _best_split(X, y, indices, features, self.min_samples_leaf)
        if split is None:
            return node
        gain = _gini(positive_count, sample_count) - split.weighted_impurity
        if gain < -1e-12:
            raise RuntimeError("Split impurity exceeds parent impurity beyond numerical roundoff")
        # Partition once, after scoring; repeated row indices remain occurrences.
        goes_left = X[indices, split.feature_index] <= split.threshold
        left_indices = indices[goes_left]
        right_indices = indices[~goes_left]
        if len(left_indices) != split.left_count or not 0 < len(left_indices) < sample_count:
            raise RuntimeError("Split threshold does not reproduce the scored partition")
        node.feature_index = split.feature_index
        node.threshold = split.threshold
        # Release this frame's input/mask; callers can still hold child indices.
        del indices, goes_left, features
        node.left = self._build_tree(X, y, left_indices, rng, depth + 1)
        del left_indices
        node.right = self._build_tree(X, y, right_indices, rng, depth + 1)
        return node

    def _check_fitted(self) -> None:
        if self.root_ is None:
            raise ValueError("DecisionTreeClassifier has not been fitted")

    def _positive_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Traverse a validated matrix iteratively; also used by the forest."""
        result = np.empty(len(X), dtype=np.float64)
        for position, row in enumerate(X):
            node = self.root_
            while node.feature_index is not None:
                if row[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            result[position] = node.positive_probability
        return result

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        X = _validate_X(X, self.n_features_in_)
        positive = self._positive_probabilities(X)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(np.int64)

    def _parameters(self) -> dict:
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features,
                "random_state": self.random_state}

    def to_dict(self) -> dict:
        self._check_fitted()

        def encode(node):
            return {"sample_count": node.sample_count, "positive_count": node.positive_count,
                    "positive_probability": node.positive_probability,
                    "feature_index": node.feature_index, "threshold": node.threshold,
                    "left": None if node.left is None else encode(node.left),
                    "right": None if node.right is None else encode(node.right)}

        return {"schema": TREE_SCHEMA, "parameters": self._parameters(),
                "n_features": self.n_features_in_, "root": encode(self.root_)}

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionTreeClassifier":
        _require_keys(data, {"schema", "parameters", "n_features", "root"}, "Tree model")
        if data["schema"] != TREE_SCHEMA:
            raise ValueError("Unknown tree schema version")
        _require_keys(data["parameters"], {"max_depth", "min_samples_split", "min_samples_leaf",
                                          "max_features", "random_state"}, "Tree parameters")
        model = cls(**data["parameters"])
        model.n_features_in_ = _integer(data["n_features"], "n_features", 1)
        model._feature_count(model.n_features_in_)
        seen = set()

        def decode(saved, depth, bounds):
            _require_keys(saved, {"sample_count", "positive_count", "positive_probability",
                                  "feature_index", "threshold", "left", "right"}, "Tree node")
            if id(saved) in seen:
                raise ValueError("Tree nodes cannot be shared or cyclic")
            seen.add(id(saved))
            count = _integer(saved["sample_count"], "sample_count", 1)
            positive = _integer(saved["positive_count"], "positive_count", 0)
            probability = _finite_number(saved["positive_probability"], "positive_probability")
            if positive > count or not 0 <= probability <= 1 or probability != positive / count:
                raise ValueError("Node probability and positive count must agree with sample count")
            node = TreeNode(count, positive, probability)
            if saved["feature_index"] is None:
                if any(saved[key] is not None for key in ("threshold", "left", "right")):
                    raise ValueError("Leaf must have no split threshold or children")
                return node
            feature = _integer(saved["feature_index"], "feature_index", 0)
            if feature >= model.n_features_in_:
                raise ValueError("Node feature_index exceeds n_features")
            if (positive in (0, count) or count < model.min_samples_split
                    or count < 2 * model.min_samples_leaf
                    or (model.max_depth is not None and depth >= model.max_depth)):
                raise ValueError("Internal node violates tree stopping conditions")
            node.feature_index = feature
            node.threshold = _finite_number(saved["threshold"], "threshold")
            lower, upper = bounds.get(feature, (-math.inf, math.inf))
            if not lower < node.threshold < upper or node.threshold == np.finfo(float).max:
                raise ValueError("Split threshold makes a child unreachable under ancestor routing")
            left_bounds = dict(bounds)
            left_bounds[feature] = (lower, node.threshold)
            right_bounds = dict(bounds)
            right_bounds[feature] = (node.threshold, upper)
            node.left = decode(saved["left"], depth + 1, left_bounds)
            node.right = decode(saved["right"], depth + 1, right_bounds)
            if (node.left.sample_count + node.right.sample_count != count
                    or node.left.positive_count + node.right.positive_count != positive
                    or min(node.left.sample_count, node.right.sample_count) < model.min_samples_leaf):
                raise ValueError("Child counts must sum to parent counts and respect min_samples_leaf")
            weighted = (node.left.sample_count / count * _gini(node.left.positive_count, node.left.sample_count)
                        + node.right.sample_count / count * _gini(node.right.positive_count, node.right.sample_count))
            if _gini(positive, count) - weighted < -1e-12:
                raise ValueError("Internal node has negative information gain")
            return node

        try:
            model.root_ = decode(data["root"], 0, {})
        except RecursionError as error:
            raise ValueError("Serialized tree exceeds Python's recursion limit") from error
        return model
