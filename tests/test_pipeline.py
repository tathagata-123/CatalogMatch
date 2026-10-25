"""Synthetic pipeline, schema, persistence, and gold-free runtime checks."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from catalogmatch import cli, pipeline
from catalogmatch.data import (
    RawRecord, SPLIT_VERSION, audit_dataset, make_manifest, prepare_record,
    sha256_file, validate_partitions, Dataset, build_positive_graph, connected_components, split_components,
)
from catalogmatch.features import FEATURE_NAMES, pair_features
from catalogmatch.forest import RandomForestClassifier
from catalogmatch.retrieval import RETRIEVAL_CONFIGS


@pytest.fixture
def provenance():
    return {
        "raw_sha256": {name: "a" * 64 for name in ("Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv")},
        "manifest_sha256": "b" * 64, "selected_sha256": "c" * 64,
        "prepared_sha256": {name: "d" * 64 for name in ("records.jsonl", "split_memberships.json", "audit.json")},
        "split_version": SPLIT_VERSION, "split_seed": 42,
    }


@pytest.fixture
def bundle(provenance):
    X = np.array([[0] * 12, [1] * 12] * 3, dtype=np.float64)
    model = RandomForestClassifier(n_estimators=3, max_depth=2, min_samples_leaf=1).fit(X, [0, 1] * 3)
    return pipeline.build_bundle(model, "custom_forest", 0.0, RETRIEVAL_CONFIGS[0], provenance)


@pytest.fixture
def prepared_fixture(tmp_path, monkeypatch):
    abt = tuple(RawRecord("abt", f"a{i:02}", f"Camera ZX{i:03}", "photo", "", "100") for i in range(30))
    buy = tuple(RawRecord("buy", f"b{i:02}", f"Camera ZX{i:03}", "photo", "", "100") for i in range(30))
    dataset = Dataset(abt, buy, tuple((a.raw_id, b.raw_id) for a, b in zip(abt, buy)),
                      {name: "a" * 64 for name in ("Abt.csv", "Buy.csv", "abt_buy_perfectMapping.csv")})
    components = connected_components(build_positive_graph(dataset.records, dataset.gold_pairs))
    partitions = split_components(components)
    processed = tmp_path / "processed"
    manifest = tmp_path / "manifest.json"
    pipeline.write_json(manifest, make_manifest(dataset, components, partitions))
    audit = audit_dataset(dataset, original=False)
    audit.update(validate_partitions(dataset.records, dataset.gold_pairs, components, partitions))
    audit["manifest_sha256"] = sha256_file(manifest)
    monkeypatch.setattr(cli, "verify_original", lambda *args: (dataset, partitions, dict(audit)))
    cli.prepare_data(Path("unused"), manifest, processed)
    selected_dir = tmp_path / "retrieval"
    cli.run_retrieval_eval(processed, Path("unused"), manifest, selected_dir)
    selected = selected_dir / "selected.json"
    config = pipeline.load_selected_config(selected, processed, dataset, audit)
    context = pipeline.input_provenance(processed, dataset, audit, selected)
    return processed, manifest, selected, dataset, partitions, audit, config, context


def test_candidate_order_deduplication_and_labels():
    left = [prepare_record(RawRecord("abt", "a", "Sony A100 camera", "", "", ""))]
    right = [prepare_record(RawRecord("buy", str(i), name, "", "", "")) for i, name in enumerate(
        ("Sony A100 camera", "Sony B200 camera", "unrelated item"))]
    pairs = pipeline.candidate_pairs(left, pipeline.build_catalog_indexes(right, RETRIEVAL_CONFIGS[5]), RETRIEVAL_CONFIGS[5])
    assert pairs.dtype == np.int64
    assert len(pairs) == len(set(map(tuple, pairs)))
    assert tuple(pairs[0]) == (0, 0)
    assert pipeline.label_candidates(pairs, {(0, 0)}).tolist() == [int(tuple(p) == (0, 0)) for p in pairs]
    with pytest.raises(ValueError, match="Duplicate"):
        pipeline.label_candidates([(0, 0), (0, 0)], {(0, 0)})


def test_prepare_partition_schema_labels_cache_and_test_gate(prepared_fixture, tmp_path, monkeypatch):
    processed, manifest, selected, dataset, partitions, audit, config, context = prepared_fixture
    args = (processed, dataset, partitions, audit, "train", config, tmp_path / "features", context)
    first = pipeline.prepare_partition(*args)
    assert not first.report["cache_used"]
    assert first.X.shape[1] == 12 and first.X.dtype == np.float64
    assert first.report["feature_names"] == list(FEATURE_NAMES)
    assert first.report["retrieval_missed_gold"] == len(first.gold - set(map(tuple, first.pairs)))
    assert first.report["positive_candidates"] + first.report["negative_candidates"] == len(first.pairs)
    assert first.report["prevalence"] == first.y.mean()
    for i, (a, b) in enumerate(first.pairs):
        np.testing.assert_array_equal(first.X[i], pair_features(first.left[a], first.right[b]))
    monkeypatch.setattr("catalogmatch.features.feature_matrix", lambda *a: pytest.fail("cache recomputed"))
    second = pipeline.prepare_partition(*args)
    np.testing.assert_array_equal(first.X, second.X)
    assert second.report["cache_used"]
    assert not second.X.flags.writeable and not second.y.flags.writeable
    with pytest.raises(ValueError, match="only train and validation"):
        pipeline.prepare_partition(processed, dataset, partitions, audit, "test", config,
                                   tmp_path / "features", context)


def test_stale_cache_rejected_and_rebuilt(prepared_fixture, tmp_path):
    processed, manifest, selected, dataset, partitions, audit, config, context = prepared_fixture
    args = (processed, dataset, partitions, audit, "validation", config, tmp_path / "features", context)
    first = pipeline.prepare_partition(*args)
    metadata_file = tmp_path / "features" / "validation.json"
    metadata = json.loads(metadata_file.read_text())
    metadata["identity"]["feature_names"].reverse()
    pipeline.write_json(metadata_file, metadata)
    second = pipeline.prepare_partition(*args)
    assert not second.report["cache_used"]
    assert "stale" in second.report["cache_reason"]
    np.testing.assert_array_equal(first.X, second.X)


def test_bundle_round_trip_and_multiple_sorted_matches(bundle, tmp_path, monkeypatch):
    path = tmp_path / "forest.json"
    pipeline.save_bundle(bundle, path)
    loaded = pipeline.load_bundle(path)
    X = np.random.default_rng(4).random((20, 12))
    np.testing.assert_array_equal(loaded.score(X), bundle.score(X))
    query = RawRecord("abt", "new", "Sony A100 camera", "", "", "")
    right = [RawRecord("buy", key, query.name, "", "", "") for key in ("z", "a")]
    for name in ("load_dataset", "load_gold", "verify_original"):
        monkeypatch.setattr("catalogmatch.data." + name, lambda *a: pytest.fail("gold consulted"))
    matches = pipeline.match_record(query, right, path)
    assert [row["target_id"] for row in matches] == ["a", "z"]
    assert list(matches[0]["features"]) == list(FEATURE_NAMES)
    assert matches == pipeline.match_record(query, list(reversed(right)), loaded)
    assert pipeline.match_record(query, [], loaded) == []
    assert pipeline.match_record(RawRecord("abt", "empty", "", "", "", ""), right, loaded) == []
    loaded.metadata["threshold"] = float(np.nextafter(1.0, np.inf))
    assert pipeline.match_record(query, right, loaded) == []


@pytest.mark.parametrize("field,value", [
    ("bundle_version", "old"), ("preprocessing_version", "other"),
    ("feature_schema_version", "bad"), ("feature_names", list(reversed(FEATURE_NAMES))),
    ("n_features", 11), ("threshold", float("nan")), ("threshold", float("inf")),
    ("threshold", -0.1), ("threshold", True), ("random_seed", 99),
    ("retrieval_config", {"name": "word10", "word_k": 25, "gram_k": None}),
    ("dependencies", {}), ("provenance", {}), ("frozen", False), ("training_split", "test"),
])
def test_reject_incompatible_bundles(bundle, tmp_path, field, value):
    path = tmp_path / "model.json"
    pipeline.save_bundle(bundle, path)
    saved = json.loads(path.read_text())
    saved["metadata"][field] = value
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        pipeline.load_bundle(path)


def test_missing_metadata_malformed_tree_and_wrong_parameters(bundle, tmp_path):
    path = tmp_path / "model.json"
    pipeline.save_bundle(bundle, path)
    saved = json.loads(path.read_text())
    for damage in ("missing", "tree", "parameters", "features"):
        changed = deepcopy(saved)
        if damage == "missing":
            del changed["metadata"]["provenance"]
        elif damage == "tree":
            changed["model"]["trees"][0]["root"]["positive_probability"] = 2.0
        elif damage == "parameters":
            changed["metadata"]["hyperparameters"]["max_depth"] += 1
        else:
            changed["model"]["n_features"] = 11
        pipeline.write_json(path, changed)
        with pytest.raises(ValueError):
            pipeline.load_bundle(path)


def test_runtime_process_forbids_sklearn_scipy_and_gold(bundle, tmp_path):
    path = tmp_path / "bundle.json"
    pipeline.save_bundle(bundle, path)
    script = '''
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in ('sklearn', 'scipy'):
        raise AssertionError('ML library imported during runtime')
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from catalogmatch.pipeline import load_bundle, match_record
from catalogmatch.data import RawRecord
bundle = load_bundle(__import__('sys').argv[1])
query = RawRecord('abt', 'q', 'Sony A100 camera', '', '', '')
right = [RawRecord('buy', 'b', 'Sony A100 camera', '', '', '')]
assert len(match_record(query, right, bundle)) == 1
'''
    result = subprocess.run([sys.executable, "-c", script, str(path)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_match_cli_uses_no_dataset(bundle, tmp_path, capsys):
    path = tmp_path / "model.json"
    pipeline.save_bundle(bundle, path)
    right = tmp_path / "Buy.csv"
    right.write_text('id,name,description,manufacturer,price\nb,Sony A100 camera,,,\n')
    assert cli.main(["match", "--bundle", str(path), "--right", str(right), "--name", "Sony A100 camera"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["target_id"] == "b"


def test_catalog_validation(bundle):
    raw = RawRecord("buy", "a", "camera", "", "", "")
    with pytest.raises(ValueError, match="unique"):
        pipeline.prepare_catalog([raw, raw], RETRIEVAL_CONFIGS[0])
    catalog = pipeline.prepare_catalog([raw], RETRIEVAL_CONFIGS[1])
    with pytest.raises(ValueError, match="configuration"):
        pipeline.match_record(RawRecord("abt", "q", "camera", "", "", ""), catalog, bundle)
