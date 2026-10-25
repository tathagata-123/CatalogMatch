"""Independent direct-count split oracle, node invariants, and sklearn references."""

import copy
import json
import weakref

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier as SklearnTree

from catalogmatch.tree import DecisionTreeClassifier, _best_split, _gini


def brute_force_split(X, y, indices, features, min_samples_leaf):
    """Deliberately slow: enumerate thresholds and recount each partition directly."""
    best = None
    for feature in sorted(features):
        distinct = sorted(set(float(X[row, feature]) for row in indices))
        for lo, hi in zip(distinct, distinct[1:]):
            threshold = lo / 2 + hi / 2
            if not lo <= threshold < hi:
                threshold = lo
            left = [row for row in indices if X[row, feature] <= threshold]
            right = [row for row in indices if X[row, feature] > threshold]
            if min(len(left), len(right)) < min_samples_leaf:
                continue
            left_probability = sum(int(y[row]) for row in left) / len(left)
            right_probability = sum(int(y[row]) for row in right) / len(right)
            impurity = (len(left) / len(indices) * (2 * left_probability * (1 - left_probability))
                        + len(right) / len(indices) * (2 * right_probability * (1 - right_probability)))
            candidate = (impurity, feature, threshold, len(left))
            if best is None or candidate < best:
                best = candidate
    return best


def assert_partition_tree(node, X, y, indices, model, depth=0):
    """Independently route every occurrence and verify every stored statistic."""
    assert node.sample_count == len(indices) > 0
    assert node.positive_count == sum(int(y[row]) for row in indices)
    assert node.positive_probability == node.positive_count / node.sample_count
    assert not any(isinstance(value, np.ndarray) for value in vars(node).values())
    if node.feature_index is None:
        assert node.left is node.right is node.threshold is None
        return 1
    assert node.left is not None and node.right is not None
    assert model.max_depth is None or depth < model.max_depth
    assert node.sample_count >= model.min_samples_split
    left = indices[X[indices, node.feature_index] <= node.threshold]
    right = indices[X[indices, node.feature_index] > node.threshold]
    assert min(len(left), len(right)) >= model.min_samples_leaf
    assert len(left) + len(right) == len(indices)
    assert max(len(left), len(right)) < len(indices)
    return (1 + assert_partition_tree(node.left, X, y, left, model, depth + 1)
            + assert_partition_tree(node.right, X, y, right, model, depth + 1))


@pytest.mark.parametrize("positive,count,expected", [(0, 4, 0), (4, 4, 0),
                                                      (2, 4, .5), (1, 4, .375), (3, 4, .375)])
def test_gini_hand_calculation(positive, count, expected):
    assert _gini(positive, count) == expected


def test_weighted_gini_hand_calculation():
    # Parent [0,0,0,1,1,1], children [0,0] and [0,1,1,1].
    weighted = 2 / 6 * _gini(0, 2) + 4 / 6 * _gini(3, 4)
    assert weighted == .25
    assert _gini(3, 6) - weighted == .25


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("leaf_size", [1, 2, 3, 5])
def test_sorted_split_matches_independent_oracle(seed, leaf_size):
    rng = np.random.default_rng(seed)
    X = rng.integers(-3, 4, size=(13, 5)).astype(np.float64)
    X[:, 3] = X[:, 0]  # Tied features, along with many duplicate values.
    X[:, 4] = 7         # Constant feature.
    y = rng.integers(0, 2, size=len(X))
    features = np.array([4, 3, 1, 0])  # Search must not depend on supplied order.
    for indices in (np.arange(len(X)), rng.integers(0, len(X), size=17)):
        expected = brute_force_split(X, y, indices, features, leaf_size)
        actual = _best_split(X, y, indices, features, leaf_size)
        assert actual is not None and expected is not None
        impurity, feature, threshold, left_count = expected
        assert actual.weighted_impurity == pytest.approx(impurity, abs=1e-15)
        assert (actual.feature_index, actual.threshold, actual.left_count) == (feature, threshold, left_count)
        assert np.count_nonzero(X[indices, feature] <= actual.threshold) == left_count
        model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=leaf_size).fit(X, y, indices)
        assert_partition_tree(model.root_, X, y, indices, model)


@pytest.mark.parametrize("X,y,leaf_size", [
    ([[1], [1], [1]], [0, 1, 0], 1),
    ([[0], [1], [2]], [0, 1, 0], 2),
    ([[0]], [1], 1),
])
def test_oracle_and_optimized_agree_when_no_split(X, y, leaf_size):
    X, y = np.array(X, dtype=float), np.array(y)
    indices = np.arange(len(X))
    assert brute_force_split(X, y, indices, [0], leaf_size) is None
    assert _best_split(X, y, indices, np.array([0]), leaf_size) is None


def test_exact_ties_choose_lower_feature_and_earliest_boundary():
    X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    y = np.array([0, 1, 0])
    split = _best_split(X, y, np.arange(3), np.array([1, 0]), 1)
    assert split.feature_index == 0 and split.threshold == .5


@pytest.mark.parametrize("X,y", [([[3]], [1]), ([[3]], [0]),
                                    ([[0], [1], [2]], [1, 1, 1]),
                                    ([[0], [1], [2]], [0, 0, 0]),
                                    ([[1, 2]] * 4, [0, 1, 0, 1])])
def test_single_sample_one_class_and_constant_features(X, y):
    model = DecisionTreeClassifier(min_samples_leaf=1).fit(X, y)
    assert model.root_.feature_index is None
    assert model.root_.positive_probability == sum(y) / len(y)
    assert model.predict_proba(X).shape == (len(X), 2)


def test_perfect_separation_and_known_leaf_probabilities():
    X = np.arange(6).reshape(-1, 1)
    pure = DecisionTreeClassifier(min_samples_leaf=2).fit(X, [0, 0, 0, 1, 1, 1])
    assert pure.root_.threshold == 2.5
    np.testing.assert_array_equal(pure.predict(X), [0, 0, 0, 1, 1, 1])
    mixed = DecisionTreeClassifier(max_depth=1, min_samples_leaf=2).fit(X, [0, 0, 1, 1, 0, 1])
    assert mixed.root_.threshold == 1.5
    np.testing.assert_array_equal(mixed.predict_proba([[1.5], [1.6]]), [[1, 0], [.25, .75]])


@pytest.mark.parametrize("parameters", [{"max_depth": 0}, {"min_samples_split": 7},
                                         {"min_samples_leaf": 4}])
def test_root_stopping_conditions(parameters):
    model = DecisionTreeClassifier(**parameters).fit(np.arange(6).reshape(-1, 1), [0, 0, 0, 1, 1, 1])
    assert model.root_.feature_index is None
    assert model.root_.sample_count == 6
    assert model.root_.positive_probability == .5


def test_zero_gain_xor_and_depth_limit():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0])
    shallow = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1).fit(X, y)
    assert shallow.root_.feature_index == 0
    assert shallow.root_.left.positive_probability == shallow.root_.right.positive_probability == .5
    assert shallow.root_.left.left is shallow.root_.right.left is None
    for depth in (2, None):
        model = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1).fit(X, y)
        np.testing.assert_array_equal(model.predict(X), y)
        assert assert_partition_tree(model.root_, X, y, np.arange(4), model) == 7


def test_bootstrap_multiplicity_affects_leaf_and_split_constraints():
    X, y = np.array([[0], [1]], dtype=float), np.array([0, 1])
    indices = np.array([0, 0, 0, 1, 1])
    model = DecisionTreeClassifier(min_samples_split=5, min_samples_leaf=2).fit(X, y, indices)
    assert model.root_.sample_count == 5 and model.root_.positive_count == 2
    assert model.root_.left.sample_count == 3 and model.root_.right.sample_count == 2
    assert_partition_tree(model.root_, X, y, indices, model)
    constant = DecisionTreeClassifier(max_depth=0).fit(X, y, [0, 1, 1, 1])
    assert constant.root_.positive_probability == .75


@pytest.mark.parametrize("lo,hi", [
    (1., np.nextafter(1., np.inf)),
    (np.nextafter(1., -np.inf), 1.),
    (-1., np.nextafter(-1., np.inf)),
    (0., np.nextafter(0., np.inf)),
    (-np.nextafter(0., np.inf), 0.),
    (np.finfo(float).max / 2, np.finfo(float).max),
    (-np.finfo(float).max, np.finfo(float).max),
])
def test_threshold_reproduces_scored_partition_at_float_boundaries(lo, hi):
    X = np.array([[hi], [lo], [lo], [hi]])
    y = np.array([1, 0, 0, 1])
    with np.errstate(over="raise", invalid="raise"):
        model = DecisionTreeClassifier(min_samples_leaf=2).fit(X, y)
    assert lo <= model.root_.threshold < hi
    assert model.root_.left.sample_count == model.root_.right.sample_count == 2
    np.testing.assert_array_equal(model.predict(X), y)
    assert_partition_tree(model.root_, X, y, np.arange(4), model)


def test_empty_predictions_ties_and_arbitrary_feature_dimensions():
    for dimension in (1, 2, 12, 19):
        model = DecisionTreeClassifier().fit(np.ones((2, dimension)), [0, 1])
        assert model.n_features_in_ == dimension
        assert model.predict_proba(np.empty((0, dimension))).shape == (0, 2)
        assert model.predict(np.empty((0, dimension))).shape == (0,)
        assert model.predict(np.zeros((1, dimension))).tolist() == [0]
        np.testing.assert_array_equal(model.predict_proba(np.zeros((1, dimension))), [[.5, .5]])


@pytest.mark.parametrize("method", ["predict", "predict_proba", "to_dict"])
def test_unfitted_errors(method):
    model = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="fitted"):
        getattr(model, method)(*(() if method == "to_dict" else ([[0]],)))


@pytest.mark.parametrize("X,y", [
    ([], []), (np.empty((0, 2)), []), (np.empty((2, 0)), [0, 1]),
    ([0, 1], [0, 1]), (np.zeros((2, 1, 1)), [0, 1]),
    ([[0], [1]], [[0], [1]]), ([[0], [1]], [0]),
    ([[0], [1]], [0, 2]), ([[0], [1]], [0, -.1]),
    ([[0], [1]], [0, np.nan]), ([[0], [1]], [0, np.inf]),
    ([[0], [1]], ["0", "1"]), ([[0], [1]], [0j, 1j]),
    ([[np.nan], [1]], [0, 1]), ([[np.inf], [1]], [0, 1]),
    ([[-np.inf], [1]], [0, 1]), ([["0"], ["1"]], [0, 1]),
    ([[0j], [1j]], [0, 1]), (np.array([[0], [1]], dtype=object), [0, 1]),
    ([[0], [1, 2]], [0, 1]),
])
def test_invalid_training_inputs(X, y):
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(X, y)


@pytest.mark.parametrize("X", [[], [0], [[0, 1]], [[np.nan]], [[np.inf]], [[-np.inf]],
                               [["0"]], [[1j]], np.empty((1, 0)), np.empty((0, 2)),
                               np.zeros((1, 1, 1))])
def test_invalid_prediction_inputs(X):
    model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        model.predict_proba(X)


@pytest.mark.parametrize("parameters", [
    {"max_depth": -1}, {"max_depth": True}, {"max_depth": 1.5},
    {"min_samples_split": 1}, {"min_samples_split": False}, {"min_samples_split": 2.5},
    {"min_samples_leaf": 0}, {"min_samples_leaf": True}, {"min_samples_leaf": 1.5},
    {"max_features": 0}, {"max_features": -1}, {"max_features": True},
    {"max_features": "log2"}, {"max_features": .5}, {"max_features": []},
    {"random_state": -1}, {"random_state": True}, {"random_state": 1.5}, {"random_state": None},
])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        DecisionTreeClassifier(**parameters)


def test_feature_count_limit_and_parameter_revalidation():
    with pytest.raises(ValueError, match="max_features"):
        DecisionTreeClassifier(max_features=3).fit([[0, 0]], [0])
    model = DecisionTreeClassifier()
    model.min_samples_leaf = 0
    with pytest.raises(ValueError, match="min_samples_leaf"):
        model.fit([[0]], [0])


@pytest.mark.parametrize("indices", [[], [-1], [2], [0.0], [True], [[0]], ["0"], [np.nan],
                                     np.array([2**64 - 1], dtype=np.uint64)])
def test_invalid_sample_indices(indices):
    with pytest.raises(ValueError, match="sample_indices"):
        DecisionTreeClassifier().fit([[0], [1]], [0, 1], indices)


def test_training_arrays_not_retained_and_refit_replaces_tree():
    X = np.array([[0.], [1.]])
    y = np.array([0, 1], dtype=np.int64)
    indices = np.array([0, 1])
    references = [weakref.ref(value) for value in (X, y, indices)]
    model = DecisionTreeClassifier(min_samples_leaf=1).fit(X, y, indices)
    del X, y, indices
    assert all(reference() is None for reference in references)
    assert model.fit([[5, 6]], [1]) is model
    assert model.n_features_in_ == 2 and model.root_.sample_count == 1


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_real_numeric_inputs_and_numpy_parameters(dtype):
    model = DecisionTreeClassifier(max_depth=np.int64(2), min_samples_leaf=np.int64(1),
                                   max_features=np.int64(1), random_state=np.int64(42))
    model.fit(np.array([[0], [1]], dtype=dtype), [False, True])
    np.testing.assert_array_equal(model.predict([[0], [1]]), [0, 1])
    json.dumps(model.to_dict(), allow_nan=False)


@pytest.mark.parametrize("label", [None, 0, 1])
def test_json_roundtrip_exact_and_independent(label):
    rng = np.random.default_rng(27)
    X = rng.normal(size=(80, 7))
    y = rng.integers(0, 2, size=80) if label is None else np.full(80, label)
    model = DecisionTreeClassifier(max_features="sqrt", min_samples_leaf=1).fit(X, y)
    saved = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    restored = DecisionTreeClassifier.from_dict(saved)
    np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    assert restored.to_dict() == model.to_dict()
    saved["root"]["positive_count"] = -1
    assert restored.root_.positive_count >= 0  # No aliases to caller dictionaries.


def split_model_dict():
    return DecisionTreeClassifier(max_depth=2, min_samples_leaf=2).fit(
        np.arange(6).reshape(-1, 1), [0, 0, 0, 1, 1, 1]).to_dict()


@pytest.mark.parametrize("path,value", [
    (("schema",), "catalogmatch.tree.v2"), (("n_features",), 0), (("n_features",), True),
    (("parameters", "max_depth"), -1), (("parameters", "max_features"), 2),
    (("parameters", "max_depth"), 0), (("parameters", "min_samples_split"), 7),
    (("parameters", "min_samples_leaf"), 4),
    (("root", "feature_index"), 1), (("root", "feature_index"), -1),
    (("root", "feature_index"), True), (("root", "feature_index"), .5),
    (("root", "threshold"), np.nan), (("root", "threshold"), np.inf),
    (("root", "threshold"), "2.5"), (("root", "threshold"), None),
    (("root", "sample_count"), 0), (("root", "sample_count"), -1),
    (("root", "sample_count"), True), (("root", "sample_count"), 6.),
    (("root", "sample_count"), 8), (("root", "positive_count"), 7),
    (("root", "positive_count"), -1), (("root", "positive_count"), True),
    (("root", "positive_probability"), np.nan), (("root", "positive_probability"), np.inf),
    (("root", "positive_probability"), -.1), (("root", "positive_probability"), 1.1),
    (("root", "positive_probability"), .25), (("root", "positive_probability"), True),
    (("root", "left"), None), (("root", "right"), None), (("root", "left"), []),
    (("root", "feature_index"), None), (("root", "left", "threshold"), 0.),
    (("root", "left", "feature_index"), 0),
    (("root", "left", "sample_count"), 2),
])
def test_corrupted_serialized_tree_rejected(path, value):
    data = split_model_dict()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        DecisionTreeClassifier.from_dict(data)


@pytest.mark.parametrize("location", [(), ("root",), ("parameters",)])
def test_missing_and_unknown_serialized_fields(location):
    for missing in (True, False):
        data = split_model_dict()
        target = data
        for key in location:
            target = target[key]
        if missing:
            target.pop(next(iter(target)))
        else:
            target["extra"] = 1
        with pytest.raises(ValueError, match="fields"):
            DecisionTreeClassifier.from_dict(data)


def test_corrupt_counts_purity_and_cyclic_nodes():
    data = split_model_dict()
    data["root"]["right"]["positive_count"] = 2
    data["root"]["right"]["positive_probability"] = 2 / 3
    with pytest.raises(ValueError, match="Child counts"):
        DecisionTreeClassifier.from_dict(data)
    data = split_model_dict()
    data["root"]["positive_count"] = 0
    data["root"]["positive_probability"] = 0.
    with pytest.raises(ValueError, match="stopping"):
        DecisionTreeClassifier.from_dict(data)
    data = split_model_dict()
    data["root"]["left"] = data["root"]
    with pytest.raises(ValueError, match="cyclic"):
        DecisionTreeClassifier.from_dict(data)
    data = split_model_dict()
    data["root"]["left"]["left"] = copy.deepcopy(data["root"]["right"])
    with pytest.raises(ValueError, match="Leaf"):
        DecisionTreeClassifier.from_dict(data)


def test_serialized_thresholds_must_allow_both_children():
    X = np.arange(8).reshape(-1, 1)
    data = DecisionTreeClassifier(max_depth=3, min_samples_leaf=1).fit(
        X, [0, 0, 1, 1, 0, 0, 1, 1]).to_dict()
    assert data["root"]["right"]["feature_index"] == 0
    data["root"]["right"]["threshold"] = data["root"]["threshold"]
    with pytest.raises(ValueError, match="unreachable"):
        DecisionTreeClassifier.from_dict(data)
    data = split_model_dict()
    data["root"]["threshold"] = np.finfo(float).max
    with pytest.raises(ValueError, match="unreachable"):
        DecisionTreeClassifier.from_dict(data)


def test_serialized_children_cannot_violate_minimum_leaf_size():
    data = split_model_dict()
    data["root"].update(positive_count=5, positive_probability=5 / 6)
    data["root"]["left"].update(sample_count=1)
    data["root"]["right"].update(sample_count=5, positive_count=5)
    with pytest.raises(ValueError, match="min_samples_leaf"):
        DecisionTreeClassifier.from_dict(data)


@pytest.mark.parametrize("labels,depth", [([0, 0, 0, 1, 1, 1], 2),
                                           ([0, 0, 1, 1, 0, 1], 1)])
def test_sklearn_unambiguous_tree_comparison(labels, depth):
    X = np.arange(6, dtype=float).reshape(-1, 1)
    settings = dict(max_depth=depth, min_samples_leaf=2, min_samples_split=2, random_state=42)
    custom = DecisionTreeClassifier(**settings).fit(X, labels)
    reference = SklearnTree(**settings).fit(X, labels)
    assert custom.root_.feature_index == reference.tree_.feature[0]
    assert custom.root_.threshold == pytest.approx(reference.tree_.threshold[0])
    query = np.linspace(-1, 6, 29).reshape(-1, 1)
    np.testing.assert_allclose(custom.predict_proba(query), reference.predict_proba(query), atol=1e-15)
    np.testing.assert_array_equal(custom.predict(query), reference.predict(query))


def test_sklearn_physically_expanded_bootstrap_comparison():
    X = np.array([[0.], [1.]])
    y = np.array([0, 1])
    indices = np.array([0, 0, 0, 1, 1])
    custom = DecisionTreeClassifier(min_samples_split=5, min_samples_leaf=2).fit(X, y, indices)
    reference = SklearnTree(min_samples_split=5, min_samples_leaf=2).fit(X[indices], y[indices])
    np.testing.assert_array_equal(custom.predict_proba(X), reference.predict_proba(X))
    assert custom.root_.sample_count == reference.tree_.n_node_samples[0] == 5
    # With weights, sklearn sees just two distinct rows for min_samples_split.
    weighted = SklearnTree(min_samples_split=5, min_samples_leaf=2).fit(X, y, sample_weight=[3, 2])
    assert weighted.tree_.node_count == 1


def test_documented_precision_difference_from_sklearn():
    X = np.array([[1.], [np.nextafter(1., np.inf)]])
    custom = DecisionTreeClassifier(min_samples_leaf=1).fit(X, [0, 1])
    reference = SklearnTree(min_samples_leaf=1).fit(X, [0, 1])
    np.testing.assert_array_equal(custom.predict(X), [0, 1])
    # sklearn converts to float32: these two distinct float64 values coalesce.
    assert reference.tree_.node_count == 1
