"""Bootstrap/RNG instrumentation and independent expanded-row sklearn ensembles."""

import json
import weakref

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier as SklearnForest
from sklearn.tree import DecisionTreeClassifier as SklearnTree

import catalogmatch.tree as tree_module
from catalogmatch.forest import RandomForestClassifier
from catalogmatch.tree import DecisionTreeClassifier


def test_bootstrap_indices_seeds_and_continuing_rng(monkeypatch):
    original = DecisionTreeClassifier._fit_with_indices
    calls = []

    def record(self, X, y, indices, rng):
        calls.append((self.random_state, indices.copy(), rng, rng.bit_generator.state))
        return original(self, X, y, indices, rng)

    monkeypatch.setattr(DecisionTreeClassifier, "_fit_with_indices", record)
    X = np.arange(60, dtype=float).reshape(20, 3)
    y = np.arange(20) % 2
    model = RandomForestClassifier(n_estimators=5, min_samples_leaf=1).fit(X, y)
    master = np.random.default_rng(42)
    assert len({id(call[2]) for call in calls}) == 5
    for tree, (seed, indices, _, state) in zip(model.estimators_, calls):
        assert seed == int(master.integers(0, np.iinfo(np.int64).max))
        rng = np.random.default_rng(seed)
        expected = rng.integers(0, len(X), size=len(X), dtype=np.int64)
        np.testing.assert_array_equal(indices, expected)
        assert state == rng.bit_generator.state
        assert len(np.unique(indices)) < len(indices)
        assert tree.root_.sample_count == len(X)
        assert tree.root_.positive_count == int(np.sum(y[indices]))
        # Full replay proves feature sampling continues after the bootstrap draw.
        replay = DecisionTreeClassifier(min_samples_leaf=1, max_features="sqrt", random_state=seed)
        replay._fit_with_indices(X, y, expected, rng)
        assert replay.to_dict() == tree.to_dict()
    assert len({tuple(call[1]) for call in calls}) == 5


def test_no_bootstrap_uses_all_rows_once(monkeypatch):
    original = DecisionTreeClassifier._fit_with_indices
    calls = []

    def record(self, X, y, indices, rng):
        calls.append(indices.copy())
        return original(self, X, y, indices, rng)

    monkeypatch.setattr(DecisionTreeClassifier, "_fit_with_indices", record)
    model = RandomForestClassifier(n_estimators=4, bootstrap=False).fit(
        np.arange(9).reshape(-1, 1), [0, 0, 0, 1, 1, 1, 1, 1, 1])
    for tree, indices in zip(model.estimators_, calls):
        np.testing.assert_array_equal(indices, np.arange(9))
        assert tree.root_.sample_count == 9 and tree.root_.positive_count == 6


def test_feature_subsets_are_fresh_sorted_and_without_replacement(monkeypatch):
    original = DecisionTreeClassifier._select_features
    selections = []
    searches = []
    search = tree_module._best_split

    def record_selection(self, count, rng):
        features = original(self, count, rng)
        selections.append(tuple(features))
        return features

    def record_search(X, y, indices, features, leaf_size):
        searches.append(tuple(features))
        return search(X, y, indices, features, leaf_size)

    monkeypatch.setattr(DecisionTreeClassifier, "_select_features", record_selection)
    monkeypatch.setattr(tree_module, "_best_split", record_search)
    rng = np.random.default_rng(18)
    X, y = rng.normal(size=(100, 12)), rng.integers(0, 2, size=100)
    forest = RandomForestClassifier(n_estimators=2, max_depth=3, min_samples_leaf=1).fit(X, y)
    assert searches == selections and len(selections) > 2
    assert len(set(selections)) > 2
    for features in selections:
        assert len(features) == len(set(features)) == 3
        assert features == tuple(sorted(features))
        assert all(0 <= feature < 12 for feature in features)

    def eligible_count(node, depth):
        if (node.positive_count in (0, node.sample_count) or depth >= 3
                or node.sample_count < 2):
            return 0
        if node.feature_index is None:
            return 1
        return 1 + eligible_count(node.left, depth + 1) + eligible_count(node.right, depth + 1)

    assert len(selections) == sum(eligible_count(tree.root_, 0) for tree in forest.estimators_)


@pytest.mark.parametrize("dimensions,setting,count", [(1, "sqrt", 1), (2, "sqrt", 1),
                                                       (12, "sqrt", 3), (19, "sqrt", 4),
                                                       (5, None, 5), (5, 2, 2)])
def test_feature_count_and_sampling_with_known_rng(dimensions, setting, count):
    model = DecisionTreeClassifier(max_features=setting)
    actual_rng, expected_rng = np.random.default_rng(91), np.random.default_rng(91)
    for _ in range(3):
        actual = model._select_features(dimensions, actual_rng)
        expected = (np.arange(dimensions) if count == dimensions else
                    np.sort(expected_rng.choice(dimensions, size=count, replace=False)))
        np.testing.assert_array_equal(actual, expected)


def test_exact_feature_subset_has_no_additional_feature_fallback(monkeypatch):
    calls = []

    def constant_feature(self, count, rng):
        calls.append(count)
        return np.array([0])

    monkeypatch.setattr(DecisionTreeClassifier, "_select_features", constant_feature)
    X = [[0, 0], [0, 0], [0, 1], [0, 1]]
    forest = RandomForestClassifier(n_estimators=3, bootstrap=False, min_samples_leaf=1).fit(X, [0, 0, 1, 1])
    assert len(calls) == 3
    assert all(tree.root_.feature_index is None for tree in forest.estimators_)
    np.testing.assert_array_equal(forest.predict_proba(X), [[.5, .5]] * 4)


def test_one_class_bootstraps_keep_two_probability_columns():
    X, y = np.array([[0.], [1.]]), np.array([0, 1])
    forest = RandomForestClassifier(n_estimators=32, min_samples_leaf=1).fit(X, y)
    pure = [tree for tree in forest.estimators_ if tree.root_.positive_count in (0, 2)]
    assert pure and {tree.root_.positive_count for tree in pure} == {0, 2}
    for tree in pure:
        expected = [1., 0.] if tree.root_.positive_count == 0 else [0., 1.]
        np.testing.assert_array_equal(tree.predict_proba(X), [expected, expected])
    assert forest.predict_proba(X).shape == (2, 2)


@pytest.mark.parametrize("bootstrap", [True, False])
def test_same_seed_reproduces_whole_forest_and_refit(bootstrap):
    rng = np.random.default_rng(111)
    X, y = rng.normal(size=(60, 9)), rng.integers(0, 2, size=60)
    first = RandomForestClassifier(n_estimators=4, bootstrap=bootstrap).fit(X, y)
    second = RandomForestClassifier(n_estimators=4, bootstrap=bootstrap).fit(X, y)
    assert first.to_dict() == second.to_dict()
    saved = first.to_dict()
    assert first.fit(X, y) is first
    assert first.to_dict() == saved


def test_probability_averaging_and_difference_from_hard_voting():
    # First verify averaging for an actually bootstrapped ensemble.
    X, y = np.zeros((5, 1)), np.array([0, 0, 0, 1, 1])
    forest = RandomForestClassifier(n_estimators=3, max_depth=0, random_state=9).fit(X, y)
    probabilities = np.array([tree.predict_proba([[0]])[0, 1] for tree in forest.estimators_])
    expected = float(np.mean(probabilities))
    np.testing.assert_allclose(forest.predict_proba([[0]]), [[1 - expected, expected]])
    assert forest.predict([[0]])[0] == int(expected > .5)
    # A separate constructed ensemble makes the distinction numerical and explicit.
    forest.estimators_ = [DecisionTreeClassifier(max_depth=0).fit(np.zeros((10, 1)), labels)
                          for labels in ([1] * 6 + [0] * 4, [1] * 6 + [0] * 4, [0] * 10)]
    np.testing.assert_allclose(forest.predict_proba([[0]]), [[.6, .4]])
    assert sum(tree.predict([[0]])[0] for tree in forest.estimators_) == 2
    assert forest.predict([[0]])[0] == 0


def test_tie_empty_and_one_class_forest():
    X = np.zeros((2, 4))
    forest = RandomForestClassifier(n_estimators=3, bootstrap=False).fit(X, [0, 1])
    np.testing.assert_array_equal(forest.predict_proba(X), [[.5, .5]] * 2)
    np.testing.assert_array_equal(forest.predict(X), [0, 0])
    assert forest.predict_proba(np.empty((0, 4))).shape == (0, 2)
    assert forest.predict(np.empty((0, 4))).shape == (0,)
    for label in (0, 1):
        forest.fit([[3]], [label])
        np.testing.assert_array_equal(forest.predict_proba([[3]]), [[1 - label, label]])
        restored = RandomForestClassifier.from_dict(json.loads(json.dumps(forest.to_dict())))
        np.testing.assert_array_equal(restored.predict_proba([[3]]), [[1 - label, label]])


@pytest.mark.parametrize("parameters", [
    {"n_estimators": 0}, {"n_estimators": True}, {"n_estimators": 2.5},
    {"bootstrap": 1}, {"bootstrap": "yes"}, {"bootstrap": None},
    {"max_depth": -1}, {"max_depth": True}, {"min_samples_split": 1},
    {"min_samples_leaf": 0}, {"max_features": 0}, {"max_features": True},
    {"max_features": "log2"}, {"max_features": .5}, {"random_state": -1},
])
def test_invalid_forest_parameters(parameters):
    with pytest.raises(ValueError):
        RandomForestClassifier(**parameters)


def test_forest_feature_count_and_changed_parameters():
    with pytest.raises(ValueError, match="max_features"):
        RandomForestClassifier(max_features=3).fit([[0, 0]], [0])
    forest = RandomForestClassifier()
    forest.n_estimators = 0
    with pytest.raises(ValueError, match="n_estimators"):
        forest.fit([[0]], [0])


@pytest.mark.parametrize("X,y", [
    (np.empty((0, 2)), []), (np.empty((2, 0)), [0, 1]), ([0, 1], [0, 1]),
    (np.zeros((2, 1, 1)), [0, 1]), ([[0], [1]], [[0], [1]]), ([[0]], [0, 1]),
    ([[0], [1]], [0, 2]), ([[0], [1]], [0, np.nan]), ([[0], [1]], [0, np.inf]),
    ([[np.nan]], [0]), ([[np.inf]], [0]), ([["1"]], [1]), ([[1j]], [1]),
])
def test_invalid_forest_training_inputs(X, y):
    with pytest.raises(ValueError):
        RandomForestClassifier().fit(X, y)


@pytest.mark.parametrize("X", [[], [0], [[0, 1]], [[np.nan]], [[np.inf]], [["0"]], [[1j]],
                               np.empty((0, 2)), np.empty((1, 0)), np.zeros((1, 1, 1))])
def test_invalid_forest_prediction_inputs(X):
    forest = RandomForestClassifier(n_estimators=2).fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        forest.predict_proba(X)


@pytest.mark.parametrize("method", ["predict", "predict_proba", "to_dict"])
def test_unfitted_forest(method):
    with pytest.raises(ValueError, match="fitted"):
        getattr(RandomForestClassifier(), method)(*(() if method == "to_dict" else ([[0]],)))


def test_forest_releases_training_arrays():
    X, y = np.ones((10, 2)), np.arange(10, dtype=np.int64) % 2
    references = weakref.ref(X), weakref.ref(y)
    forest = RandomForestClassifier(n_estimators=2).fit(X, y)
    del X, y
    assert all(reference() is None for reference in references)
    assert not any(isinstance(value, np.ndarray) for value in vars(forest).values())


@pytest.mark.parametrize("bootstrap", [True, False])
def test_forest_json_roundtrip_exact(bootstrap):
    rng = np.random.default_rng(22)
    X, y = rng.normal(size=(60, 12)), rng.integers(0, 2, size=60)
    forest = RandomForestClassifier(n_estimators=5, bootstrap=bootstrap).fit(X, y)
    data = json.loads(json.dumps(forest.to_dict(), allow_nan=False))
    restored = RandomForestClassifier.from_dict(data)
    np.testing.assert_array_equal(restored.predict_proba(X), forest.predict_proba(X))
    np.testing.assert_array_equal(restored.predict(X), forest.predict(X))
    assert restored.to_dict() == forest.to_dict()
    assert restored.predict_proba(np.empty((0, 12))).shape == (0, 2)
    data["tree_seeds"][0] = -1
    assert restored.tree_seeds_[0] >= 0


@pytest.mark.parametrize("damage", ["version", "missing", "unknown", "parameters", "features",
                                    "tree_count", "seed_count", "tree_type", "seed_type", "seed",
                                    "tree_features", "tree_parameters", "tree_seed", "node",
                                    "root_counts", "nonbootstrap_positives"])
def test_corrupt_forest_rejected(damage):
    forest = RandomForestClassifier(n_estimators=2, max_depth=0, bootstrap=False).fit(
        [[0], [1], [2], [3]], [0, 0, 1, 1])
    data = forest.to_dict()
    if damage == "version":
        data["schema"] = "v999"
    elif damage == "missing":
        del data["parameters"]
    elif damage == "unknown":
        data["extra"] = 1
    elif damage == "parameters":
        data["parameters"]["bootstrap"] = "yes"
    elif damage == "features":
        data["n_features"] = 0
    elif damage == "tree_count":
        data["trees"].pop()
    elif damage == "seed_count":
        data["tree_seeds"].pop()
    elif damage == "tree_type":
        data["trees"] = {}
    elif damage == "seed_type":
        data["tree_seeds"][0] = True
    elif damage == "seed":
        data["tree_seeds"][0] += 1
    elif damage == "tree_features":
        data["trees"][0]["n_features"] = 2
    elif damage == "tree_parameters":
        data["trees"][0]["parameters"]["min_samples_leaf"] = 1
    elif damage == "tree_seed":
        data["trees"][0]["parameters"]["random_state"] += 1
    elif damage == "node":
        data["trees"][0]["root"]["positive_probability"] = np.nan
    elif damage == "root_counts":
        data["trees"][0]["root"].update(sample_count=6, positive_count=3)
    else:
        data["trees"][0]["root"].update(positive_count=1, positive_probability=.25)
    with pytest.raises(ValueError):
        RandomForestClassifier.from_dict(data)


def test_sklearn_equivalent_expanded_bootstrap_ensemble():
    # One binary feature: a unique legal boundary, including noisy leaves.
    X = np.repeat([[0.], [10.]], 10, axis=0)
    y = np.array([0] * 8 + [1] * 2 + [0] * 2 + [1] * 8)
    forest = RandomForestClassifier(n_estimators=8, max_depth=2, max_features=None).fit(X, y)
    query = np.array([[-1.], [0.], [4.], [6.], [10.], [11.]])
    reference_probabilities = []
    for tree, seed in zip(forest.estimators_, forest.tree_seeds_):
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(X), size=len(X), dtype=np.int64)
        reference = SklearnTree(max_depth=2, min_samples_leaf=2, random_state=42)
        reference.fit(X[indices], y[indices])
        positive = np.zeros(len(query))
        if 1 in reference.classes_:
            positive = reference.predict_proba(query)[:, list(reference.classes_).index(1)]
        np.testing.assert_allclose(tree.predict_proba(query)[:, 1], positive, atol=1e-15)
        reference_probabilities.append(positive)
    expected = np.mean(reference_probabilities, axis=0)
    np.testing.assert_allclose(forest.predict_proba(query)[:, 1], expected, atol=1e-15)
    np.testing.assert_array_equal(forest.predict(query), expected > .5)


def test_sklearn_forest_comparison_with_aligned_nonbootstrap_settings():
    X = np.repeat([[0.], [10.]], 10, axis=0)
    y = np.array([0] * 8 + [1] * 2 + [0] * 2 + [1] * 8)
    parameters = dict(n_estimators=4, max_depth=2, min_samples_leaf=2,
                      max_features=None, bootstrap=False, random_state=42)
    custom = RandomForestClassifier(**parameters).fit(X, y)
    reference = SklearnForest(**parameters).fit(X, y)
    np.testing.assert_allclose(custom.predict_proba(X), reference.predict_proba(X), atol=1e-15)
    np.testing.assert_array_equal(custom.predict(X), reference.predict(X))


@pytest.mark.parametrize("depth", [8, 12])
@pytest.mark.parametrize("leaf_size", [2, 5])
def test_intended_bounded_grid_is_supported(depth, leaf_size):
    rng = np.random.default_rng(66)
    X, y = rng.normal(size=(40, 12)), rng.integers(0, 2, size=40)
    forest = RandomForestClassifier(max_depth=depth, min_samples_leaf=leaf_size).fit(X, y)
    assert len(forest.estimators_) == 32
    assert forest.max_features == "sqrt" and forest.bootstrap and forest.random_state == 42
    probabilities = forest.predict_proba(X)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.)


def test_synthetic_holdout_quality_against_sklearn():
    # Fixed synthetic signal, no product data or parameter search. RNGs need not agree.
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(1000, 12))
    y = (X[:, 0] + .6 * X[:, 1] - .4 * X[:, 2] > 0).astype(np.int64)
    query = rng.normal(size=(400, 12))
    labels = (query[:, 0] + .6 * query[:, 1] - .4 * query[:, 2] > 0).astype(np.int64)
    settings = dict(n_estimators=32, max_depth=8, min_samples_leaf=2,
                    max_features="sqrt", bootstrap=True, random_state=42)
    custom = RandomForestClassifier(**settings).fit(X, y)
    reference = SklearnForest(**settings, n_jobs=1).fit(X, y)
    custom_accuracy = np.mean(custom.predict(query) == labels)
    reference_accuracy = np.mean(reference.predict(query) == labels)
    # A broad quality check supplements (never replaces) the exact algorithmic tests.
    assert min(custom_accuracy, reference_accuracy) >= .85
    assert abs(custom_accuracy - reference_accuracy) < .08
