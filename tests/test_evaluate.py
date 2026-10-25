"""Independent threshold oracle and synthetic end-to-end freeze/evaluation tests."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from catalogmatch import cli, evaluate, pipeline
from catalogmatch.data import RawRecord, prepare_record
from catalogmatch.features import FEATURE_NAMES
from catalogmatch.forest import RandomForestClassifier
from catalogmatch.tree import DecisionTreeClassifier
from test_pipeline import prepared_fixture, provenance, bundle


def slow_reference(y, scores, gold):
    """Independent direct counts, including predict-none and every distinct score."""
    outcomes = []
    for threshold in [float(np.nextafter(1.0, np.inf))] + sorted(set(scores)):
        tp = sum(int(label == 1 and score >= threshold) for label, score in zip(y, scores))
        fp = sum(int(label == 0 and score >= threshold) for label, score in zip(y, scores))
        fn = gold - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        outcomes.append(((f1, precision, threshold), (tp, fp, fn)))
    key, counts = max(outcomes)
    return key[2], counts


@pytest.mark.parametrize("seed", range(30))
def test_threshold_sweep_matches_independent_slow_reference(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=60)
    scores = rng.integers(0, 11, size=60) / 10
    gold = int(y.sum()) + 3
    expected_threshold, expected_counts = slow_reference(y, scores, gold)
    threshold, metrics = evaluate.select_threshold(y, scores, gold)
    assert threshold == expected_threshold
    assert tuple(metrics[k] for k in ("true_positives", "false_positives", "false_negatives")) == expected_counts


@pytest.mark.parametrize("y,scores,gold,expected_threshold", [
    ([1, 0], [0.5, 0.5], 1, 0.5),
    ([1, 1], [0.3, 0.8], 2, 0.3),
    ([0, 0], [0.4, 1], 2, evaluate.PREDICT_NONE),
    ([], [], 3, evaluate.PREDICT_NONE),
    ([], [], 0, evaluate.PREDICT_NONE),
    ([0, 0], [0, 0], 0, evaluate.PREDICT_NONE),
    # At .9: TP=1,FP=0,FN=1,F1=2/3. At .1: TP=2,FP=2,FN=0,F1=2/3.
    ([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.1], 2, 0.9),
    # One unretrieved gold changes the optimal F1 threshold to .1.
    ([1, 0, 0, 1], [0.9, 0.8, 0.7, 0.1], 3, 0.1),
])
def test_threshold_edge_cases(y, scores, gold, expected_threshold):
    threshold, metrics = evaluate.select_threshold(y, scores, gold)
    assert threshold == expected_threshold
    assert np.isfinite(threshold)
    assert metrics["recall"] <= metrics["candidate_recall"]
    assert metrics["recall"] == pytest.approx(metrics["candidate_recall"] * metrics["conditional_classifier_recall"])


def test_hand_calculated_metrics_retrieval_miss_stays_false_negative():
    result = evaluate.classification_metrics([1, 0, 1], [0.9, 0.8, 0.1], 0.5, 3)
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 2
    assert result["retrieval_missed_gold"] == 1
    assert result["classifier_rejected_gold"] == 1
    assert result["precision"] == 0.5
    assert result["recall"] == pytest.approx(1 / 3)
    assert result["f1"] == 0.4
    assert result["candidate_recall"] == pytest.approx(2 / 3)
    none = evaluate.classification_metrics([1, 0, 1], [0.9, 0.8, 0.1], 2.0, 3)
    assert none["false_negatives"] == 3 and none["precision"] == 0


@pytest.mark.parametrize("y,scores,gold", [([1], [1.1], 1), ([1], [float("nan")], 1),
    ([1], [0.5], 0), ([2], [0.5], 2), ([0, 1], [0.2], 1), ([1], [0.5], -1)])
def test_bad_threshold_inputs(y, scores, gold):
    with pytest.raises(ValueError):
        evaluate.select_threshold(y, scores, gold)


def test_configuration_grid_and_tie_break():
    for name in evaluate.MODEL_NAMES:
        grid = evaluate.parameter_grid(name)
        assert len(grid) == (1 if name == "lexical" else 4)
    rows = [{"validation": {"f1": 0.8}, "hyperparameters": params}
            for params in evaluate.parameter_grid("custom_forest")]
    chosen = max(rows, key=evaluate.configuration_key)
    assert chosen["hyperparameters"]["max_depth"] == 8
    assert chosen["hyperparameters"]["min_samples_leaf"] == 5
    assert evaluate.parameter_grid("custom_cart")[0]["max_features"] is None
    assert evaluate.parameter_grid("sklearn_forest")[0]["n_jobs"] == 1


def candidate_data(split, keys_prefix):
    rng = np.random.default_rng(14 if split == "train" else 15)
    left = [prepare_record(RawRecord("abt", keys_prefix + str(i), "camera", "", "", "")) for i in range(8)]
    right = [prepare_record(RawRecord("buy", keys_prefix + str(i), "camera", "", "", "")) for i in range(4)]
    pairs = np.asarray([(i, j) for i in range(8) for j in range(4)], dtype=np.int64)
    X = rng.random((32, 12))
    y = (X[:, 0] > 0.65).astype(np.int64)
    gold = {tuple(pair) for pair, label in zip(pairs, y) if label}
    return pipeline.CandidateData(split, left, right, pairs, X, y, gold,
                                  {"feature_names": list(FEATURE_NAMES), "X_sha256": split + "-X", "y_sha256": split + "-y"})


def test_comparison_uses_custom_learners_shared_matrices_and_validation_only(provenance, monkeypatch):
    from sklearn.ensemble import RandomForestClassifier as SklearnForest
    train = candidate_data("train", "train")
    validation = candidate_data("validation", "val")
    fits = []
    for cls in (DecisionTreeClassifier, RandomForestClassifier, SklearnForest):
        original = cls.fit
        def spy(self, X, y, _original=original, **kwargs):
            assert X is train.X and y is train.y
            fits.append(type(self))
            return _original(self, X, y, **kwargs)
        monkeypatch.setattr(cls, "fit", spy)
    report, models, scores = evaluate.compare_models(train, validation, pipeline.RETRIEVAL_CONFIGS[0], provenance)
    assert fits.count(DecisionTreeClassifier) == 4
    assert fits.count(RandomForestClassifier) == 4
    assert fits.count(SklearnForest) == 4
    assert len(report["tried"]) == 13
    assert type(models["custom_cart"].model) is DecisionTreeClassifier
    assert type(models["custom_forest"].model) is RandomForestClassifier
    for name, row in report["selected"].items():
        expected, _ = slow_reference(validation.y, scores[name], len(validation.gold))
        assert row["threshold"] == expected
        assert row["training_X_sha256"] == train.report["X_sha256"]
    with pytest.raises(ValueError, match="validation only"):
        evaluate.compare_models(train, replace(validation, split="test"), pipeline.RETRIEVAL_CONFIGS[0], provenance)
    with pytest.raises(ValueError, match="leakage"):
        evaluate.compare_models(train, replace(validation, left=train.left), pipeline.RETRIEVAL_CONFIGS[0], provenance)


def test_frozen_end_to_end_on_synthetic_dataset(prepared_fixture, tmp_path, monkeypatch):
    processed, manifest, selected, dataset, partitions, audit, config, context = prepared_fixture
    monkeypatch.setattr(evaluate, "verify_original", lambda *args: (dataset, partitions, dict(audit)))
    calls = []
    original = pipeline.load_prepared_partition
    def checked(*args, **kwargs):
        calls.append(args[4])
        if args[4] == "test":
            assert (tmp_path / "run" / "frozen.json").exists()
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, "load_prepared_partition", checked)
    run = tmp_path / "run"
    report = evaluate.tune_run(processed, Path("unused"), manifest, selected, run)
    assert calls == ["train", "validation"]
    assert (run / "frozen.json").exists()
    assert not (run / "features" / "test.npz").exists()
    for name in evaluate.MODEL_NAMES:
        assert report["selected"][name]["threshold"] <= evaluate.PREDICT_NONE
    selection_before = (run / "selection.json").read_bytes()
    monkeypatch.setattr(evaluate, "select_threshold", lambda *a: pytest.fail("test retuned threshold"))
    monkeypatch.setattr(RandomForestClassifier, "fit", lambda *a: pytest.fail("test retrained"))
    result = evaluate.evaluate_run(processed, Path("unused"), manifest, run)
    assert calls[-1] == "test"
    assert not result["retuned"] and not result["refit"]
    assert (run / "selection.json").read_bytes() == selection_before
    assert set(result["models"]) == set(evaluate.MODEL_NAMES)
    with pytest.raises(ValueError, match="already exist"):
        evaluate.evaluate_run(processed, Path("unused"), manifest, run)
    with pytest.raises(ValueError, match="empty output"):
        evaluate.tune_run(processed, Path("unused"), manifest, selected, run)
    frozen = run / "frozen.json"
    saved_frozen = frozen.read_text()
    frozen.write_text('{"schema": "invalid", "frozen": false}')
    with pytest.raises(ValueError, match="frozen"):
        evaluate.load_frozen_run(run)
    frozen.write_text(saved_frozen)
    (run / "custom_forest.json").write_text('{}')
    with pytest.raises(ValueError, match="artifact changed"):
        evaluate.load_frozen_run(run)


def test_unfrozen_evaluation_cannot_prepare_test(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "prepare_partition", lambda *a, **kw: pytest.fail("prepared test before freeze"))
    with pytest.raises(FileNotFoundError):
        evaluate.evaluate_run(tmp_path, tmp_path, tmp_path, tmp_path)


def test_error_examples_include_retrieval_misses_and_rejections(bundle):
    data = candidate_data("validation", "val")
    # Remove one positive from candidates, preserving the gold set.
    positive = int(np.flatnonzero(data.y)[0])
    keep = np.arange(len(data.y)) != positive
    data = replace(data, pairs=data.pairs[keep], X=data.X[keep], y=data.y[keep])
    scores = np.where(data.y == 1, 0.1, 0.9)
    report = evaluate.error_examples(data, scores, 0.5, bundle, limit=2)
    categories = {row["error_category"] for row in report["examples"]}
    assert categories == {"false_positive", "retrieved_rejected", "retrieval_miss"}
    assert len(report["examples"]) <= 6
    for row in report["examples"]:
        assert len(row["features"]) == 12
        assert row["score_is_diagnostic_only"] == (row["error_category"] == "retrieval_miss")


def test_query_benchmark_actual_sample_counts_and_same_decisions(bundle):
    right = [RawRecord("buy", str(i), "Sony A100 camera", "", "", "") for i in range(3)]
    catalog = pipeline.prepare_catalog(right, pipeline.RETRIEVAL_CONFIGS[0])
    queries = [RawRecord("abt", "q", "Sony A100 camera", "", "", "")]
    assert evaluate.exhaustive_match(queries[0], catalog, bundle) == pipeline.match_record(queries[0], catalog, bundle)
    measured = evaluate._query_timings(queries, catalog, bundle)
    for name in ("indexed", "exhaustive"):
        assert len(measured[name]["query_latency"]["samples_seconds"]) == 5
        assert len(measured[name]["pass_totals"]["samples_seconds"]) == 5
        assert measured[name]["accepted_pairs"] == 3
    assert evaluate.tree_node_count(bundle) > 0


def test_new_cli_commands_and_path_guards(tmp_path, capsys):
    for command in ("tune", "evaluate", "benchmark"):
        with pytest.raises(SystemExit) as error:
            cli.main([command, "--help"])
        assert error.value.code == 0
    assert cli.main(["tune", "--out", str(tmp_path), "--data", str(tmp_path)]) == 1
    assert "separate" in capsys.readouterr().err
