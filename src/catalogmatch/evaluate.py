"""Validation selection, frozen test evaluation, error examples, and benchmarks.

Scikit-learn imports and reference-model persistence live only in this module.
No test attributes are prepared until load_frozen_run has verified the selection.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import platform
import subprocess
from time import perf_counter
import tracemalloc

import numpy as np

from catalogmatch.data import prepare_record, sha256_file, verify_original
from catalogmatch.features import FEATURE_NAMES, feature_matrix, pair_features
from catalogmatch.pipeline import (
    ModelBundle, PreparedCatalog, build_bundle, build_catalog_indexes, candidate_pairs,
    input_provenance, load_bundle, load_selected_config, match_record, prepare_partition,
    save_bundle, score_pairs, train_bundle, validate_features, write_json,
)
from catalogmatch.retrieval import RetrievalConfig


PREDICT_NONE = float(np.nextafter(1.0, np.inf))
MODEL_NAMES = ("lexical", "custom_cart", "custom_forest", "sklearn_forest")
RUN_VERSION = "catalogmatch.selection.v1"


def _checked_scores(y, scores, total_gold):
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    if (y.ndim != 1 or scores.ndim != 1 or len(y) != len(scores)
            or not np.all((y == 0) | (y == 1))):
        raise ValueError("Expected aligned one-dimensional binary labels and scores")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0) or np.any(scores > 1):
        raise ValueError("Scores must be finite and in [0, 1]")
    if type(total_gold) is not int or total_gold < int(y.sum()):
        raise ValueError("Total gold count must include every retrieved positive")
    return y.astype(np.int64, copy=False), scores


def metrics_from_counts(tp, fp, retrieved_gold, total_gold):
    fn = total_gold - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gold if total_gold else 0.0
    candidate_recall = retrieved_gold / total_gold if total_gold else 0.0
    conditional_recall = tp / retrieved_gold if retrieved_gold else 0.0
    if recall > candidate_recall or not math.isclose(
            recall, candidate_recall * conditional_recall, rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("Recall accounting is inconsistent")
    return {
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "gold_pairs": total_gold, "retrieved_gold": retrieved_gold,
        "candidate_recall": candidate_recall, "conditional_classifier_recall": conditional_recall,
        "retrieval_missed_gold": total_gold - retrieved_gold,
        "classifier_rejected_gold": retrieved_gold - tp,
    }


def classification_metrics(y, scores, threshold, total_gold):
    y, scores = _checked_scores(y, scores, total_gold)
    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite")
    accepted = scores >= threshold
    tp = int(y[accepted].sum())
    return metrics_from_counts(tp, int(accepted.sum()) - tp, int(y.sum()), total_gold)


def retrieval_metrics(data):
    possible = len(data.left) * len(data.right)
    retrieved = int(data.y.sum())
    return {
        "candidate_pairs": len(data.pairs), "cartesian_pairs": possible,
        "candidate_recall": retrieved / len(data.gold) if data.gold else 0.0,
        "candidate_reduction_ratio": 1 - len(data.pairs) / possible if possible else 0.0,
        "retrieval_missed_gold": len(data.gold) - retrieved,
    }


def select_threshold(y, scores, total_gold):
    """O(n log n) sorting, then one O(n) sweep over complete equal-score groups.

    Retrieval misses remain in the recall/F1 denominator at EVERY threshold.
    The tuple comparison implements F1, precision, then higher-threshold ties.
    """
    y, scores = _checked_scores(y, scores, total_gold)
    retrieved = int(y.sum())
    best_threshold = PREDICT_NONE
    best = metrics_from_counts(0, 0, retrieved, total_gold)
    best_key = (best["f1"], best["precision"], best_threshold)
    order = np.argsort(-scores, kind="stable")
    tp = fp = offset = 0
    while offset < len(order):
        score = float(scores[order[offset]])
        end = offset
        while end < len(order) and scores[order[end]] == score:
            if y[order[end]]:
                tp += 1
            else:
                fp += 1
            end += 1
        metrics = metrics_from_counts(tp, fp, retrieved, total_gold)
        key = (metrics["f1"], metrics["precision"], score)
        if key > best_key:
            best_threshold, best, best_key = score, metrics, key
        offset = end
    return best_threshold, best


def candidate_average_precision(y, scores):
    from sklearn.metrics import average_precision_score

    if len(y) == 0 or not np.any(y):
        return 0.0
    return float(average_precision_score(y, scores))


def parameter_grid(model_type):
    if model_type not in MODEL_NAMES:
        raise ValueError("Unknown comparison model")
    if model_type == "lexical":
        return [{}]
    grid = []
    for depth in (8, 12):
        for leaf in (2, 5):
            parameters = {"max_depth": depth, "min_samples_leaf": leaf,
                          "min_samples_split": 2, "random_state": 42}
            if model_type == "custom_cart":
                parameters["max_features"] = None
            else:
                parameters.update(n_estimators=32, max_features="sqrt", bootstrap=True)
            if model_type == "sklearn_forest":
                parameters.update(criterion="gini", n_jobs=1)
            grid.append(parameters)
    return grid


def configuration_key(result):
    parameters = result["hyperparameters"]
    return (result["validation"]["f1"], -parameters.get("max_depth", 0),
            parameters.get("min_samples_leaf", 0))


def dependency_versions():
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scikit-learn": sklearn.__version__}


def machine_info():
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                text=True, capture_output=True, check=False)
        if result.returncode == 0:
            cpu = result.stdout.strip()
    return {**dependency_versions(), "os": platform.platform(), "cpu": cpu,
            "logical_cpu_count": os.cpu_count(), "processes": 1}


def score_model(model, X):
    if isinstance(model, ModelBundle):
        return model.score(X)
    if len(X) == 0:
        return np.empty(0, dtype=np.float64)
    # sklearn can have a single class on synthetic all-negative/all-positive fits.
    if 1 not in model.classes_:
        return np.zeros(len(X), dtype=np.float64)
    column = list(model.classes_).index(1)
    return model.predict_proba(X)[:, column]


def timing_summary(samples):
    return {"samples_seconds": samples, "median_seconds": float(np.median(samples)),
            "p95_seconds": float(np.percentile(samples, 95)),
            "percentile_method": "NumPy linear percentile of actual samples"}


def time_inference(model, X):
    score_model(model, X)  # One untimed warmup.
    samples = []
    for _ in range(5):
        started = perf_counter()
        score_model(model, X)
        samples.append(perf_counter() - started)
    return {**timing_summary(samples), "rows_per_pass": len(X), "warmups": 1,
            "passes": 5, "scope": "batch scoring; excludes retrieval/features and loading"}


def _check_selection_data(training, validation):
    if training.split != "train" or validation.split != "validation":
        raise ValueError("Selection accepts train and validation only")
    train_keys = {record.raw.key for record in training.left + training.right}
    val_keys = {record.raw.key for record in validation.left + validation.right}
    if train_keys & val_keys:
        raise ValueError("Training/validation record leakage")
    if not validation.gold:
        raise ValueError("Validation selection requires gold pairs")
    for data in (training, validation):
        validate_features(data.X)
        if data.report["feature_names"] != list(FEATURE_NAMES):
            raise ValueError("Feature ordering differs between partitions")
        if len(data.X) != len(data.y) or len(data.X) != len(data.pairs):
            raise ValueError("Candidate rows and labels are misaligned")
        keys = [tuple(pair) for pair in data.pairs]
        if len(set(keys)) != len(keys) or not np.array_equal(data.y, [int(p in data.gold) for p in keys]):
            raise ValueError("Invalid candidate labels or duplicate pairs")


def compare_models(training, validation, config, provenance, *, progress=None):
    """Fixed thirteen trials; shared matrices; train only, select on validation."""
    from sklearn.ensemble import RandomForestClassifier as SklearnForest

    _check_selection_data(training, validation)
    dependencies = dependency_versions()
    tried = []
    selected = {}
    models = {}
    selected_scores = {}
    # Measure a complete full-data custom forest before considering any fallback.
    for name in ("custom_forest", "custom_cart", "sklearn_forest", "lexical"):
        for parameters in parameter_grid(name):
            if progress:
                progress(f"Fitting {name}: {parameters}")
            if name in ("custom_cart", "custom_forest"):
                model, result, scores = train_bundle(
                    training.X, training.y, validation.X, validation.y, len(validation.gold),
                    config=config, provenance=provenance, parameters=parameters,
                    model_type=name, dependencies=dependencies)
                if name == "custom_forest" and result["training_seconds"] > 120:
                    raise RuntimeError("Full custom forest exceeded 120 seconds. Profile training before "
                                       "deciding whether a shared sampled training list is necessary.")
                threshold = model.metadata["threshold"]
            else:
                if name == "lexical":
                    model = build_bundle(None, name, PREDICT_NONE, config, provenance, dependencies)
                    training_seconds = 0.0
                else:
                    model = SklearnForest(**parameters)
                    started = perf_counter()
                    model.fit(training.X, training.y)
                    training_seconds = perf_counter() - started
                scores = score_model(model, validation.X)
                threshold, metrics = select_threshold(validation.y, scores, len(validation.gold))
                result = {"training_seconds": training_seconds, "validation": metrics}
                if name == "lexical":
                    model.metadata["threshold"] = threshold
            result.update(model_type=name, hyperparameters=parameters, threshold=threshold,
                          training_X_sha256=training.report["X_sha256"],
                          training_y_sha256=training.report["y_sha256"],
                          validation_X_sha256=validation.report["X_sha256"],
                          inference=time_inference(model, validation.X))
            result["validation"].update(retrieval_metrics(validation))
            result["validation"]["candidate_only_average_precision"] = candidate_average_precision(validation.y, scores)
            tried.append(result)
            if name not in selected or configuration_key(result) > configuration_key(selected[name]):
                selected[name], models[name], selected_scores[name] = result, model, scores
    matched_gaps = []
    for custom in (row for row in tried if row["model_type"] == "custom_forest"):
        params = custom["hyperparameters"]
        reference = next(row for row in tried if row["model_type"] == "sklearn_forest"
                         and row["hyperparameters"]["max_depth"] == params["max_depth"]
                         and row["hyperparameters"]["min_samples_leaf"] == params["min_samples_leaf"])
        matched_gaps.append({"max_depth": params["max_depth"], "min_samples_leaf": params["min_samples_leaf"],
                             "sklearn_minus_custom_f1": reference["validation"]["f1"] - custom["validation"]["f1"]})
    diagnostics = {
        "comparable_forest_gaps": matched_gaps,
        "forest_below_lexical": selected["custom_forest"]["validation"]["f1"] < selected["lexical"]["validation"]["f1"],
        "investigation_required": any(row["sklearn_minus_custom_f1"] > 0.03 for row in matched_gaps),
        "training_sampling": "none; all retrieved training pairs used by all learners",
    }
    return {"tried": tried, "selected": selected, "diagnostics": diagnostics}, models, selected_scores


def error_examples(data, scores, threshold, model, limit=10):
    """Bounded observed errors; diagnostic scores on misses never enter metrics."""
    rows = {tuple(pair): i for i, pair in enumerate(data.pairs)}
    false_positives = [tuple(pair) for pair, y, score in zip(data.pairs, data.y, scores)
                       if y == 0 and score >= threshold]
    rejected = [tuple(pair) for pair, y, score in zip(data.pairs, data.y, scores)
                if y == 1 and score < threshold]
    missed = sorted(data.gold - rows.keys())
    false_positives.sort(key=lambda pair: (-scores[rows[pair]], pair))
    rejected.sort(key=lambda pair: (scores[rows[pair]], pair))
    examples = []
    for category, pairs in (("false_positive", false_positives),
                            ("retrieved_rejected", rejected), ("retrieval_miss", missed)):
        for pair in pairs[:limit]:
            left, right = data.left[pair[0]], data.right[pair[1]]
            values = np.asarray(pair_features(left, right), dtype=np.float64)
            score = (float(scores[rows[pair]]) if pair in rows
                     else float(score_model(model, values.reshape(1, 12))[0]))
            factors = []
            if values[5]:
                factors.append("model_number_conflict")
            if not values[8]:
                factors.append("missing_description")
            if not values[10]:
                factors.append("missing_price")
            if category == "retrieval_miss":
                factors.append("retrieval_exclusion; inspect selected-channel overlap and top-k rank")
            examples.append({"abt_id": left.raw.raw_id, "buy_id": right.raw.raw_id,
                             "abt_title": left.raw.name, "buy_title": right.raw.name,
                             "gold_label": int(pair in data.gold), "score": score,
                             "threshold": threshold, "error_category": category,
                             "score_is_diagnostic_only": pair not in rows,
                             "features": dict(zip(FEATURE_NAMES, values.tolist())),
                             "possible_factors_not_causal_explanations": factors})
    return {"split": data.split, "limit_per_category": limit, "examples": examples}


def tune_run(processed_dir, raw_dir, manifest_path, selected_path, output_dir, *, progress=None):
    output_dir = Path(output_dir)
    # Preserve earlier selections/results. A new experiment gets a new run directory.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Tuning requires an empty output directory; preserve frozen runs")
    started = perf_counter()
    dataset, partitions, audit = verify_original(raw_dir, manifest_path)
    config = load_selected_config(selected_path, processed_dir, dataset, audit)
    provenance = input_provenance(processed_dir, dataset, audit, selected_path)
    verification_seconds = perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    training = prepare_partition(processed_dir, dataset, partitions, audit, "train", config,
                                  output_dir / "features", provenance)
    validation = prepare_partition(processed_dir, dataset, partitions, audit, "validation", config,
                                    output_dir / "features", provenance)
    report, models, scores = compare_models(training, validation, config, provenance, progress=progress)
    if (report["diagnostics"]["investigation_required"] or report["diagnostics"]["forest_below_lexical"]):
        # Keep evidence reviewable before allowing the held-out stage.
        report["diagnostics"]["review_status"] = "required before freezing"
    else:
        report["diagnostics"]["review_status"] = "no quality-gap trigger"
    report.update(schema=RUN_VERSION, provenance=provenance, retrieval_config=asdict(config),
                  training=training.report, validation=validation.report, environment=machine_info(),
                  verification_seconds=verification_seconds, training_split="train",
                  selection_split="validation", refit_on_validation=False, negative_sampling=False)
    files = ["retrieval.json", "selection.json", "validation_scores.npz"]
    (output_dir / "retrieval.json").write_bytes(Path(selected_path).read_bytes())
    for name in MODEL_NAMES:
        result = report["selected"][name]
        filename = f"{name}.pkl" if name == "sklearn_forest" else f"{name}.json"
        if name == "sklearn_forest":
            # Reference only: load exclusively from this trusted, hash-checked run.
            (output_dir / filename).write_bytes(pickle.dumps(models[name], protocol=pickle.HIGHEST_PROTOCOL))
        else:
            save_bundle(models[name], output_dir / filename)
            restored = load_bundle(output_dir / filename)
            if not np.array_equal(restored.score(validation.X), scores[name]):
                raise RuntimeError("Bundle round trip changed validation scores")
        result["model_file"] = filename
        files.append(filename)
        error_file = f"validation_errors_{name}.json"
        write_json(output_dir / error_file, error_examples(validation, scores[name], result["threshold"], models[name]))
        files.append(error_file)
    np.savez_compressed(output_dir / "validation_scores.npz", pairs=validation.pairs,
                        labels=validation.y, **scores)
    write_json(output_dir / "selection.json", report)
    if report["diagnostics"]["review_status"] == "required before freezing":
        raise ValueError("Validation quality diagnostic triggered. Review saved validation errors and "
                         "comparison before explicitly freezing this run with freeze_run(..., diagnostic_review=...).")
    freeze_run(output_dir, files=files)
    return report


def freeze_run(run_dir, *, files=None, diagnostic_review=None):
    """Commit selection last, after validation diagnostics. Never modify a frozen run."""
    run_dir = Path(run_dir)
    if (run_dir / "frozen.json").exists():
        raise ValueError("Run is already frozen")
    selection = json.loads((run_dir / "selection.json").read_text())
    if selection["diagnostics"]["review_status"] == "required before freezing":
        if not isinstance(diagnostic_review, str) or not diagnostic_review.strip():
            raise ValueError("A concrete validation diagnostic review is required")
        selection["diagnostics"].update(review_status="reviewed", review=diagnostic_review)
        write_json(run_dir / "selection.json", selection)
    if files is None:
        files = ["selection.json", "retrieval.json", "validation_scores.npz"]
        files += [selection["selected"][name]["model_file"] for name in MODEL_NAMES]
        files += [f"validation_errors_{name}.json" for name in MODEL_NAMES]
    write_json(run_dir / "frozen.json", {
        "schema": RUN_VERSION, "frozen": True, "frozen_at": datetime.now(timezone.utc).isoformat(),
        "files_sha256": {name: sha256_file(run_dir / name) for name in files},
    })


def load_frozen_run(run_dir):
    run_dir = Path(run_dir)
    frozen = json.loads((run_dir / "frozen.json").read_text())
    if frozen.get("schema") != RUN_VERSION or frozen.get("frozen") is not True:
        raise ValueError("Test evaluation requires a frozen selection")
    hashes = frozen.get("files_sha256", {})
    required = {"selection.json", "retrieval.json", "validation_scores.npz", "sklearn_forest.pkl"}
    required |= {f"{name}.json" for name in MODEL_NAMES if name != "sklearn_forest"}
    required |= {f"validation_errors_{name}.json" for name in MODEL_NAMES}
    if not isinstance(hashes, dict) or set(hashes) != required:
        raise ValueError("Frozen run has incomplete artifact identities")
    for name, digest in hashes.items():
        if sha256_file(run_dir / name) != digest:
            raise ValueError(f"Frozen artifact changed: {name}")
    selection = json.loads((run_dir / "selection.json").read_text())
    if (selection["schema"] != RUN_VERSION or selection["training_split"] != "train"
            or selection["selection_split"] != "validation" or selection["refit_on_validation"] is not False
            or set(selection["selected"]) != set(MODEL_NAMES)
            or selection["diagnostics"]["review_status"] == "required before freezing"):
        raise ValueError("Invalid frozen selection protocol")
    models = {}
    for name in MODEL_NAMES:
        result = selection["selected"][name]
        filename = f"{name}.pkl" if name == "sklearn_forest" else f"{name}.json"
        if result["model_file"] != filename:
            raise ValueError("Invalid frozen model filename")
        if name == "sklearn_forest":
            from sklearn.ensemble import RandomForestClassifier as SklearnForest
            # Only these three versions, not OS/CPU, constrain the reference artifact.
            for key, value in dependency_versions().items():
                if selection["environment"].get(key) != value:
                    raise ValueError(f"Reference model dependency version changed: {key}")
            model = pickle.loads((run_dir / filename).read_bytes())
            if type(model) is not SklearnForest or model.n_features_in_ != 12:
                raise ValueError("Invalid sklearn reference model")
            if any(model.get_params()[key] != value for key, value in result["hyperparameters"].items()):
                raise ValueError("Reference parameters differ from frozen selection")
        else:
            model = load_bundle(run_dir / filename)
            if (model.metadata["threshold"] != result["threshold"]
                    or model.metadata["hyperparameters"] != result["hyperparameters"]
                    or model.metadata["provenance"] != selection["provenance"]
                    or model.metadata["retrieval_config"] != selection["retrieval_config"]):
                raise ValueError("Bundle differs from frozen selection")
        models[name] = model
    return selection, models


def evaluate_run(processed_dir, raw_dir, manifest_path, run_dir, split="test"):
    if split != "test":
        raise ValueError("Frozen evaluation supports the held-out test partition only")
    run_dir = Path(run_dir)
    selection, models = load_frozen_run(run_dir)
    if (run_dir / "test_metrics.json").exists():
        raise ValueError("Test results already exist; preserve the first evaluation")
    started = perf_counter()
    dataset, partitions, audit = verify_original(raw_dir, manifest_path)
    config = load_selected_config(run_dir / "retrieval.json", processed_dir, dataset, audit)
    provenance = input_provenance(processed_dir, dataset, audit, run_dir / "retrieval.json")
    if provenance != selection["provenance"]:
        raise ValueError("Evaluation inputs differ from frozen training provenance")
    verification_seconds = perf_counter() - started
    test = prepare_partition(processed_dir, dataset, partitions, audit, "test", config,
                              run_dir / "features", provenance, allow_test=True)
    results = {}
    all_scores = {}
    for name in MODEL_NAMES:
        result = selection["selected"][name]
        scores = score_model(models[name], test.X)
        all_scores[name] = scores
        metrics = classification_metrics(test.y, scores, result["threshold"], len(test.gold))
        metrics.update(retrieval_metrics(test))
        metrics["candidate_only_average_precision"] = candidate_average_precision(test.y, scores)
        results[name] = {"hyperparameters": result["hyperparameters"], "threshold": result["threshold"],
                         "metrics": metrics, "training_seconds": result["training_seconds"],
                         "inference": time_inference(models[name], test.X)}
        write_json(run_dir / f"test_errors_{name}.json",
                   error_examples(test, scores, result["threshold"], models[name]))
    np.savez_compressed(run_dir / "test_scores.npz", pairs=test.pairs, labels=test.y, **all_scores)
    report = {"split": "test", "retuned": False, "refit": False,
              "frozen_sha256": sha256_file(run_dir / "frozen.json"),
              "evaluated_at": datetime.now(timezone.utc).isoformat(),
              "provenance": provenance, "partition": test.report, "models": results,
              "verification_seconds": verification_seconds, "environment": machine_info()}
    write_json(run_dir / "test_metrics.json", report)
    return report


def exhaustive_match(raw_query, catalog, bundle):
    query = prepare_record(raw_query)
    pairs = np.asarray([(0, position) for position in range(len(catalog.records))],
                       dtype=np.int64).reshape(-1, 2)
    return score_pairs(query, catalog.records, pairs, bundle)


def _query_timings(queries, catalog, bundle):
    """Alternate method order across passes; collect individual query samples."""
    methods = {"indexed": match_record, "exhaustive": exhaustive_match}
    for method in methods.values():
        for query in queries:
            method(query, catalog, bundle)
    samples = {name: [] for name in methods}
    totals = {name: [] for name in methods}
    accepted = {}
    for timing_pass in range(5):
        order = ("indexed", "exhaustive") if timing_pass % 2 == 0 else ("exhaustive", "indexed")
        for name in order:
            pass_samples = []
            count = 0
            for query in queries:
                started = perf_counter()
                matches = methods[name](query, catalog, bundle)
                pass_samples.append(perf_counter() - started)
                count += len(matches)
            samples[name].extend(pass_samples)
            totals[name].append(sum(pass_samples))
            accepted[name] = count
    return {name: {"query_latency": timing_summary(samples[name]),
                   "pass_totals": timing_summary(totals[name]),
                   "accepted_pairs": accepted[name]} for name in methods}


def tree_node_count(model):
    if isinstance(model, ModelBundle):
        model = model.model
    if model is None:
        return 0
    if hasattr(model, "estimators_"):
        return sum(tree_node_count(tree) for tree in model.estimators_)
    if hasattr(model, "tree_"):
        return int(model.tree_.node_count)
    count = 0
    stack = [model.root_]
    while stack:
        node = stack.pop()
        count += 1
        if node.left is not None:
            stack.extend((node.left, node.right))
    return count


def benchmark_run(processed_dir, raw_dir, manifest_path, run_dir, queries=20, *, progress=None):
    if type(queries) is not int or not 1 <= queries <= 100:
        raise ValueError("Benchmark query count must be between 1 and 100")
    run_dir = Path(run_dir)
    if (run_dir / "benchmark.json").exists():
        raise ValueError("Benchmark results already exist; preserve measured samples")
    selection, models = load_frozen_run(run_dir)
    started = perf_counter()
    dataset, partitions, audit = verify_original(raw_dir, manifest_path)
    config = load_selected_config(run_dir / "retrieval.json", processed_dir, dataset, audit)
    provenance = input_provenance(processed_dir, dataset, audit, run_dir / "retrieval.json")
    if provenance != selection["provenance"]:
        raise ValueError("Benchmark inputs differ from frozen training provenance")
    verification_seconds = perf_counter() - started
    training = prepare_partition(processed_dir, dataset, partitions, audit, "train", config,
                                  run_dir / "features", provenance)
    bundle = models["custom_forest"]
    # A separate uncached pass measures feature computation even when preparation
    # above reused the verified disk cache. All classifiers share this same work.
    started = perf_counter()
    cold_X = feature_matrix(training.pairs, training.left, training.right)
    uncached_feature_seconds = perf_counter() - started
    if not np.array_equal(cold_X, training.X):
        raise ValueError("Uncached features disagree with saved training matrix")
    del cold_X
    started = perf_counter()
    indexes = build_catalog_indexes(training.right, config)
    index_seconds = perf_counter() - started
    catalog = PreparedCatalog(training.right, indexes, config)
    requested = min(queries, len(training.left))
    if requested == 0:
        raise ValueError("Benchmark requires training queries")
    raw_queries = [record.raw for record in training.left[:requested]]
    pilot_queries = raw_queries[:min(10, requested)]
    if progress:
        progress(f"Measuring exhaustive pilot: {len(pilot_queries)} queries x {len(catalog.records)} targets")
    started = perf_counter()
    for query in pilot_queries:
        exhaustive_match(query, catalog, bundle)
    pilot_seconds = perf_counter() - started
    # Only grow beyond twenty after a comfortably short measured pilot.
    if requested > 20 and pilot_seconds > 2.0:
        raw_queries = raw_queries[:20]
    if progress:
        progress(f"Benchmarking {len(raw_queries)} queries: one warmup and five timing passes per method")
    timing = _query_timings(raw_queries, catalog, bundle)
    subset = {i for i in range(len(raw_queries))}
    subset_pairs = {tuple(pair) for pair in training.pairs if pair[0] in subset}
    subset_gold = {pair for pair in training.gold if pair[0] in subset}
    candidate_recall = len(subset_gold & subset_pairs) / len(subset_gold) if subset_gold else 0.0
    # Separate allocation observation: no tracemalloc instrumentation in timings.
    tracemalloc.start()
    allocated_bundle = load_bundle(run_dir / "custom_forest.json")
    allocated_right = [prepare_record(record.raw) for record in training.right]
    allocated_catalog = PreparedCatalog(allocated_right, build_catalog_indexes(allocated_right, config), config)
    match_record(raw_queries[0], allocated_catalog, allocated_bundle)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory = {
        "training_feature_nbytes": training.X.nbytes,
        "training_candidate_array_nbytes": training.pairs.nbytes,
        "training_label_nbytes": training.y.nbytes,
        "tree_nodes": {name: tree_node_count(model) for name, model in models.items()},
        "serialized_model_bytes": {name: (run_dir / result["model_file"]).stat().st_size
                                   for name, result in selection["selected"].items()},
        "tracemalloc_current_bytes": current, "tracemalloc_peak_bytes": peak,
        "tracemalloc_scope": "separate custom bundle load + target preparation/index + one query; Python allocations, NOT process RSS",
    }
    report = {
        "split": "train", "quality_scope": "computational comparison on training data, not held-out quality",
        "environment": machine_info(), "retrieval_config": asdict(config),
        "custom_forest_parameters": selection["selected"]["custom_forest"]["hyperparameters"],
        "threshold": bundle.metadata["threshold"], "requested_queries": queries,
        "query_count": len(raw_queries), "query_ids": [record.raw_id for record in raw_queries],
        "target_catalog_size": len(catalog.records), "pilot_query_count": len(pilot_queries),
        "pilot_exhaustive_seconds": pilot_seconds, "verification_seconds": verification_seconds,
        "preparation": training.report, "index_build_seconds": index_seconds,
        "uncached_training_feature_seconds": uncached_feature_seconds,
        "indexed_pair_count": len(subset_pairs), "exhaustive_pair_count": len(raw_queries) * len(catalog.records),
        "candidate_recall": candidate_recall, "gold_pairs": len(subset_gold),
        "retrieval_missed_gold": len(subset_gold - subset_pairs),
        "query_timing": timing,
        "measured_median_pass_speedup": timing["exhaustive"]["pass_totals"]["median_seconds"] / timing["indexed"]["pass_totals"]["median_seconds"],
        "query_methodology": "first raw-ID-ordered training queries; one full warmup and five passes; method order alternates; query normalization, retrieval (indexed only), uncached features, scoring, threshold, output sorting; excludes catalog preparation/index/model load",
        "batch_inference": {name: time_inference(model, training.X) for name, model in models.items()},
        "selected_training_seconds": {name: result["training_seconds"] for name, result in selection["selected"].items()},
        "memory": memory, "frozen_sha256": sha256_file(run_dir / "frozen.json"),
    }
    write_json(run_dir / "benchmark.json", report)
    return report
