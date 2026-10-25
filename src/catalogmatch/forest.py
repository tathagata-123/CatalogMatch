"""Sequential Random Forest built exclusively from the custom CART learner."""

import numpy as np

from catalogmatch.tree import (
    DecisionTreeClassifier, _integer, _require_keys, _validate_training, _validate_X,
)


FOREST_SCHEMA = "catalogmatch.forest.v1"


class RandomForestClassifier:
    """Average leaf probabilities from independently seeded, bootstrapped trees.

    n_estimators >= 1, bootstrap is bool; other limits match DecisionTreeClassifier.
    Exactly max_features are inspected at every eligible node, with no fallback.
    """

    def __init__(self, n_estimators: int = 32, max_depth: int | None = 8,
                 min_samples_split: int = 2, min_samples_leaf: int = 2,
                 max_features: int | str | None = "sqrt", bootstrap: bool = True,
                 random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self._validate_parameters()
        self.estimators_ = []
        self.tree_seeds_ = []
        self.n_features_in_ = None

    def _tree_parameters(self) -> dict:
        return {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf, "max_features": self.max_features,
                "random_state": self.random_state}

    def _validate_parameters(self) -> None:
        self.n_estimators = _integer(self.n_estimators, "n_estimators", 1)
        if not isinstance(self.bootstrap, (bool, np.bool_)):
            raise ValueError("bootstrap must be a boolean")
        self.bootstrap = bool(self.bootstrap)
        # Share all CART parameter rules without an estimator framework.
        parameters = DecisionTreeClassifier(**self._tree_parameters())._parameters()
        self.max_depth = parameters["max_depth"]
        self.min_samples_split = parameters["min_samples_split"]
        self.min_samples_leaf = parameters["min_samples_leaf"]
        self.max_features = parameters["max_features"]
        self.random_state = parameters["random_state"]

    def fit(self, X, y) -> "RandomForestClassifier":
        self._validate_parameters()
        X, y = _validate_training(X, y)
        DecisionTreeClassifier(**self._tree_parameters())._feature_count(X.shape[1])
        master_rng = np.random.default_rng(self.random_state)
        estimators = []
        seeds = []
        for _ in range(self.n_estimators):
            seed = int(master_rng.integers(0, np.iinfo(np.int64).max))
            rng = np.random.default_rng(seed)
            if self.bootstrap:
                indices = rng.integers(0, len(X), size=len(X), dtype=np.int64)
            else:
                indices = np.arange(len(X), dtype=np.int64)
            parameters = self._tree_parameters()
            parameters["random_state"] = seed
            tree = DecisionTreeClassifier(**parameters)
            # Continue the same RNG after the bootstrap draw for node feature choices.
            tree._fit_with_indices(X, y, indices, rng)
            estimators.append(tree)
            seeds.append(seed)
        self.estimators_ = estimators
        self.tree_seeds_ = seeds
        self.n_features_in_ = X.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not self.estimators_:
            raise ValueError("RandomForestClassifier has not been fitted")

    def predict_proba(self, X) -> np.ndarray:
        self._check_fitted()
        X = _validate_X(X, self.n_features_in_)
        positive = np.zeros(len(X), dtype=np.float64)
        for tree in self.estimators_:
            positive += tree._positive_probabilities(X)
        positive /= len(self.estimators_)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(np.int64)

    def to_dict(self) -> dict:
        self._check_fitted()
        parameters = self._tree_parameters()
        parameters.update(n_estimators=self.n_estimators, bootstrap=self.bootstrap)
        return {"schema": FOREST_SCHEMA, "parameters": parameters,
                "n_features": self.n_features_in_, "tree_seeds": list(self.tree_seeds_),
                "trees": [tree.to_dict() for tree in self.estimators_]}

    @classmethod
    def from_dict(cls, data: dict) -> "RandomForestClassifier":
        _require_keys(data, {"schema", "parameters", "n_features", "tree_seeds", "trees"}, "Forest model")
        if data["schema"] != FOREST_SCHEMA:
            raise ValueError("Unknown forest schema version")
        _require_keys(data["parameters"], {"n_estimators", "bootstrap", "max_depth", "min_samples_split",
                                          "min_samples_leaf", "max_features", "random_state"}, "Forest parameters")
        model = cls(**data["parameters"])
        model.n_features_in_ = _integer(data["n_features"], "n_features", 1)
        DecisionTreeClassifier(**model._tree_parameters())._feature_count(model.n_features_in_)
        if (not isinstance(data["trees"], list) or not isinstance(data["tree_seeds"], list)
                or len(data["trees"]) != model.n_estimators
                or len(data["tree_seeds"]) != model.n_estimators):
            raise ValueError("Forest tree and seed counts must equal n_estimators")
        master_rng = np.random.default_rng(model.random_state)
        for saved, seed in zip(data["trees"], data["tree_seeds"]):
            seed = _integer(seed, "tree seed", 0)
            if seed != int(master_rng.integers(0, np.iinfo(np.int64).max)):
                raise ValueError("Tree seed does not agree with forest random_state")
            tree = DecisionTreeClassifier.from_dict(saved)
            expected = model._tree_parameters()
            expected["random_state"] = seed
            if tree.n_features_in_ != model.n_features_in_ or tree._parameters() != expected:
                raise ValueError("Tree features or parameters do not agree with forest")
            if model.estimators_ and tree.root_.sample_count != model.estimators_[0].root_.sample_count:
                raise ValueError("Forest trees must have equal root sample counts")
            if (not model.bootstrap and model.estimators_
                    and tree.root_.positive_count != model.estimators_[0].root_.positive_count):
                raise ValueError("Nonbootstrap trees must have equal root positive counts")
            model.estimators_.append(tree)
            model.tree_seeds_.append(seed)
        return model
