"""Checked candidate matrices, custom model bundles, and gold-free matching.

Only preparation/labeling functions consume a Dataset. Runtime matching needs
raw product attributes, a target catalog, NumPy, and the saved custom forest.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from time import perf_counter

import numpy as np

from catalogmatch.data import (
    DataValidationError, RawRecord, PreparedRecord, SPLIT_VERSION,
    load_prepared_partition, prepare_record, sha256_file,
)
from catalogmatch.features import (
    FEATURE_NAMES, FEATURE_SCHEMA_VERSION, cached_feature_matrix, feature_matrix,
)
from catalogmatch.forest import RandomForestClassifier
from catalogmatch.normalize import NORMALIZER_VERSION
from catalogmatch.retrieval import (
    InvertedIndex, RetrievalConfig, RETRIEVAL_CONFIGS, RETRIEVAL_VERSION,
    retrieve_candidates, select_configuration,
)
from catalogmatch.tree import DecisionTreeClassifier


BUNDLE_VERSION = "catalogmatch.bundle.v1"


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def input_provenance(processed_dir, dataset, audit, selected_path) -> dict:
    return {
        "raw_sha256": dataset.raw_sha256,
        "manifest_sha256": audit["manifest_sha256"],
        "prepared_sha256": {name: sha256_file(processed_dir / name) for name in
                            ("records.jsonl", "split_memberships.json", "audit.json")},
        "selected_sha256": sha256_file(selected_path),
        "split_version": SPLIT_VERSION, "split_seed": 42,
    }


def build_catalog_indexes(right, config):
    """Build only enabled channels, always from the supplied target catalog."""
    word = InvertedIndex()
    gram = InvertedIndex()
    if config.word_k is not None:
        word = InvertedIndex.build(record.title_word_tokens for record in right)
    if config.gram_k is not None:
        gram = InvertedIndex.build(record.title_trigrams for record in right)
    return word, gram


def candidate_pairs(left, indexes, config) -> np.ndarray:
    """Preserve query/retriever order; deduplicate without introducing new pairs."""
    pairs = []
    for position, record in enumerate(left):
        seen = set()
        for candidate in retrieve_candidates(record, *indexes, config):
            target = candidate.target_position
            if target not in seen:
                pairs.append((position, target))
                seen.add(target)
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def label_candidates(pairs, gold_positions) -> np.ndarray:
    """Evaluation/training boundary: labels attach AFTER candidate retrieval."""
    keys = [tuple(pair) for pair in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate candidate pairs")
    return np.asarray([int(pair in gold_positions) for pair in keys], dtype=np.int64)


@dataclass
class CandidateData:
    split: str
    left: list[PreparedRecord]
    right: list[PreparedRecord]
    pairs: np.ndarray
    X: np.ndarray
    y: np.ndarray
    gold: set[tuple[int, int]]
    report: dict


def prepare_partition(processed_dir, dataset, partitions, audit, split, config,
                      cache_dir, provenance, *, allow_test=False) -> CandidateData:
    started = perf_counter()
    left, right = load_prepared_partition(processed_dir, dataset, partitions, audit,
                                          split, allow_test=allow_test)
    preparation_seconds = perf_counter() - started
    started = perf_counter()
    indexes = build_catalog_indexes(right, config)
    index_seconds = perf_counter() - started
    started = perf_counter()
    pairs = candidate_pairs(left, indexes, config)
    retrieval_seconds = perf_counter() - started
    context = {**provenance, "split": split, "retrieval_config": asdict(config),
               "retrieval_version": RETRIEVAL_VERSION}
    X, cache = cached_feature_matrix(Path(cache_dir) / f"{split}.npz", pairs, left, right,
                                     provenance=context)
    validate_features(X)
    left_positions = {record.raw.raw_id: i for i, record in enumerate(left)}
    right_positions = {record.raw.raw_id: i for i, record in enumerate(right)}
    gold = set()
    for a, b in dataset.gold_pairs:
        if (a in left_positions) != (b in right_positions):
            raise DataValidationError("Gold pair crosses partition")
        if a in left_positions:
            gold.add((left_positions[a], right_positions[b]))
    y = label_candidates(pairs, gold)
    positives = int(y.sum())
    report = {
        "split": split, "left_records": len(left), "right_records": len(right),
        "candidate_pairs": len(pairs), "positive_candidates": positives,
        "negative_candidates": len(pairs) - positives, "gold_pairs": len(gold),
        "retrieval_missed_gold": len(gold) - positives,
        "prevalence": positives / len(pairs) if len(pairs) else 0.0,
        "feature_names": list(FEATURE_NAMES), "shape": list(X.shape), "dtype": str(X.dtype),
        "X_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
        "y_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
        "pairs_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
        "feature_nbytes": X.nbytes, "candidate_nbytes": pairs.nbytes, "label_nbytes": y.nbytes,
        "preparation_seconds": preparation_seconds, "index_build_seconds": index_seconds,
        "retrieval_seconds": retrieval_seconds, **cache,
    }
    # Every learner receives the same objects; accidental in-place edits fail.
    for array in (pairs, X, y):
        array.setflags(write=False)
    return CandidateData(split, left, right, pairs, X, y, gold, report)


def validate_features(X) -> None:
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or X.shape[1] != 12
            or X.dtype != np.float64 or len(FEATURE_NAMES) != 12
            or not np.all(np.isfinite(X)) or np.any(X < 0) or np.any(X > 1)):
        raise ValueError("Expected finite float64 features with shape (n, 12) in [0, 1]")


@dataclass
class ModelBundle:
    metadata: dict
    model: DecisionTreeClassifier | RandomForestClassifier | None

    def score(self, X):
        validate_features(X)
        if self.metadata["model_type"] == "lexical":
            return np.maximum(X[:, 0], X[:, 2])
        return self.model.predict_proba(X)[:, 1]


def build_bundle(model, model_type, threshold, config, provenance, dependencies=None):
    parameters = {} if model is None else model.to_dict()["parameters"]
    metadata = {
        "bundle_version": BUNDLE_VERSION, "preprocessing_version": NORMALIZER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES),
        "n_features": 12, "retrieval_version": RETRIEVAL_VERSION,
        "retrieval_config": asdict(config), "model_type": model_type,
        "hyperparameters": parameters, "threshold": float(threshold), "random_seed": 42,
        "provenance": provenance, "dependencies": dependencies or {
            "python": platform.python_version(), "numpy": np.__version__},
        "training_split": "train", "selection_split": "validation", "frozen": True,
    }
    bundle = ModelBundle(metadata, model)
    validate_bundle(bundle)
    return bundle


def train_bundle(X, y, validation_X, validation_y, validation_gold_count, *, config,
                 provenance, parameters, model_type="custom_forest", dependencies=None):
    """Fit a custom learner on training rows, then select its validation threshold.

    The run-level comparison checks partition identities before calling this
    numerical helper. No test data is accepted by the run-level tuning function.
    """
    from catalogmatch.evaluate import select_threshold

    validate_features(X)
    validate_features(validation_X)
    if model_type == "custom_cart":
        model = DecisionTreeClassifier(**parameters)
    elif model_type == "custom_forest":
        model = RandomForestClassifier(**parameters)
    else:
        raise ValueError("train_bundle requires a custom CART or forest")
    started = perf_counter()
    model.fit(X, y)
    training_seconds = perf_counter() - started
    scores = model.predict_proba(validation_X)[:, 1]
    threshold, metrics = select_threshold(validation_y, scores, validation_gold_count)
    bundle = build_bundle(model, model_type, threshold, config, provenance, dependencies)
    return bundle, {"training_seconds": training_seconds, "validation": metrics}, scores


def _check_hash(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_bundle(bundle) -> None:
    """Check the inference envelope; tree structures are checked on load/save."""
    metadata = bundle.metadata
    expected = {
        "bundle_version": BUNDLE_VERSION, "preprocessing_version": NORMALIZER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_names": list(FEATURE_NAMES),
        "n_features": 12, "retrieval_version": RETRIEVAL_VERSION,
        "training_split": "train", "selection_split": "validation", "frozen": True,
        "random_seed": 42,
    }
    required = set(expected) | {"retrieval_config", "model_type", "hyperparameters",
                               "threshold", "provenance", "dependencies"}
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError("Bundle metadata has missing or unknown fields")
    for key, value in expected.items():
        if metadata[key] != value or type(metadata[key]) is not type(value):
            raise ValueError(f"Incompatible bundle {key}")
    try:
        config = RetrievalConfig(**metadata["retrieval_config"])
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid bundle retrieval configuration") from error
    if config not in RETRIEVAL_CONFIGS:
        raise ValueError("Unsupported retrieval configuration")
    threshold = metadata["threshold"]
    if (type(threshold) not in (float, int) or not math.isfinite(threshold)
            or not 0 <= threshold <= np.nextafter(1.0, np.inf)):
        raise ValueError("Invalid bundle threshold")
    provenance = metadata["provenance"]
    if (not isinstance(provenance, dict)
            or set(provenance) != {"raw_sha256", "manifest_sha256", "prepared_sha256",
                                  "selected_sha256", "split_version", "split_seed"}
            or provenance["split_version"] != SPLIT_VERSION or provenance["split_seed"] != 42
            or not _check_hash(provenance["manifest_sha256"])
            or not _check_hash(provenance["selected_sha256"])):
        raise ValueError("Invalid bundle provenance")
    for field, names in (("raw_sha256", {"Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv"}),
                         ("prepared_sha256", {"records.jsonl", "split_memberships.json", "audit.json"})):
        hashes = provenance[field]
        if not isinstance(hashes, dict) or set(hashes) != names or not all(map(_check_hash, hashes.values())):
            raise ValueError(f"Invalid bundle {field}")
    versions = metadata["dependencies"]
    if (not isinstance(versions, dict) or not {"python", "numpy"} <= versions.keys()
            or not all(isinstance(v, str) and v for v in versions.values())):
        raise ValueError("Missing dependency versions")
    model_type = metadata["model_type"]
    if model_type == "lexical":
        if bundle.model is not None or metadata["hyperparameters"] != {}:
            raise ValueError("Lexical bundle must have no trained model")
    else:
        expected_class = {"custom_cart": DecisionTreeClassifier,
                          "custom_forest": RandomForestClassifier}.get(model_type)
        if expected_class is None or type(bundle.model) is not expected_class:
            raise ValueError("Incompatible bundle model type")
        if bundle.model.n_features_in_ != 12:
            raise ValueError("Bundle model must have twelve features")
        # Compare scalar parameters without serializing all nodes on warm queries.
        params = (bundle.model._parameters() if model_type == "custom_cart"
                  else {**bundle.model._tree_parameters(), "n_estimators": bundle.model.n_estimators,
                        "bootstrap": bundle.model.bootstrap})
        if params != metadata["hyperparameters"] or params["random_state"] != metadata["random_seed"]:
            raise ValueError("Bundle model parameters disagree with metadata")


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    validate_bundle(bundle)
    state = None if bundle.model is None else bundle.model.to_dict()
    if state is not None:
        type(bundle.model).from_dict(state)
    write_json(path, {"metadata": bundle.metadata, "model": state})


def load_bundle(path: Path) -> ModelBundle:
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or set(saved) != {"metadata", "model"}:
        raise ValueError("Malformed model bundle")
    metadata = saved["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("Malformed bundle metadata")
    model_type = metadata.get("model_type")
    if model_type == "custom_cart":
        model = DecisionTreeClassifier.from_dict(saved["model"])
    elif model_type == "custom_forest":
        model = RandomForestClassifier.from_dict(saved["model"])
    elif model_type == "lexical" and saved["model"] is None:
        model = None
    else:
        raise ValueError("Unsupported bundle model type")
    bundle = ModelBundle(metadata, model)
    validate_bundle(bundle)
    return bundle


@dataclass
class PreparedCatalog:
    records: list[PreparedRecord]
    indexes: tuple[InvertedIndex, InvertedIndex]
    config: RetrievalConfig


def prepare_catalog(raw_records, config) -> PreparedCatalog:
    records = []
    seen = set()
    # ID order stabilizes retrieval tie-breaking even if callers reorder the CSV.
    for raw in sorted(raw_records, key=lambda record: record.raw_id):
        if raw.source != "buy" or raw.raw_id in seen:
            raise ValueError("Target catalog requires unique Buy record IDs")
        seen.add(raw.raw_id)
        records.append(prepare_record(raw))
    return PreparedCatalog(records, build_catalog_indexes(records, config), config)


def match_record(left_record: RawRecord, right_catalog, model_bundle):
    """Many-to-many matching; returned features are evidence, not causal explanations.

    Pass a PreparedCatalog to reuse indexes across queries. Scores are average
    leaf fractions, not guaranteed calibrated probabilities of product identity.
    """
    bundle = load_bundle(model_bundle) if isinstance(model_bundle, (str, Path)) else model_bundle
    validate_bundle(bundle)
    if bundle.metadata["model_type"] != "custom_forest":
        raise ValueError("Product matching requires the saved custom forest")
    config = RetrievalConfig(**bundle.metadata["retrieval_config"])
    catalog = (right_catalog if isinstance(right_catalog, PreparedCatalog)
               else prepare_catalog(right_catalog, config))
    if catalog.config != config:
        raise ValueError("Catalog retrieval configuration differs from bundle")
    query = prepare_record(left_record)
    pairs = candidate_pairs([query], catalog.indexes, config)
    return score_pairs(query, catalog.records, pairs, bundle)


def score_pairs(query, right, pairs, bundle):
    """Compute the same features, decisions, and output for indexed/exhaustive pairs."""
    X = feature_matrix(pairs, [query], right)
    scores = bundle.score(X)
    matches = []
    for pair, values, score in zip(pairs, X, scores):
        if score >= bundle.metadata["threshold"]:
            target = right[pair[1]].raw
            matches.append({"target_id": target.raw_id, "title": target.name,
                            "score": float(score), "features": dict(zip(FEATURE_NAMES, values.tolist()))})
    return sorted(matches, key=lambda match: (-match["score"], match["target_id"]))


def load_selected_config(path: Path, processed_dir: Path, dataset, audit: dict) -> RetrievalConfig:
    """Validate saved selection/provenance without rerunning retrieval evaluation."""
    report = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "split": "validation", "query_source": "abt", "target_source": "buy",
        "raw_sha256": dataset.raw_sha256, "manifest_sha256": audit["manifest_sha256"],
        "split_version": SPLIT_VERSION, "split_seed": 42,
        "normalizer_version": NORMALIZER_VERSION, "retrieval_version": RETRIEVAL_VERSION,
        "prepared_sha256": {
            filename: sha256_file(processed_dir / filename)
            for filename in ("records.jsonl", "split_memberships.json", "audit.json")
        },
    }
    if not isinstance(report, dict):
        raise DataValidationError("Selected retrieval configuration must be a JSON object")
    for key, value in expected.items():
        if report.get(key) != value:
            raise DataValidationError(f"Selected retrieval configuration has stale or invalid {key}")
    try:
        config = RetrievalConfig(**report["selected_config"])
        if config not in RETRIEVAL_CONFIGS or config != select_configuration(report["comparison"]):
            raise ValueError("Configuration differs from the saved validation selection")
    except (KeyError, TypeError, ValueError) as error:
        raise DataValidationError(f"Invalid selected retrieval configuration: {error}") from error
    return config

