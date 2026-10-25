"""Independent numerical expectations and cache/CLI boundary checks, all offline."""

from dataclasses import replace
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest

from catalogmatch import cli, features
from catalogmatch.data import RawRecord, prepare_record
from catalogmatch.features import FEATURE_NAMES, cached_feature_matrix, feature_matrix, pair_features


EXPECTED_NAMES = (
    "title_token_jaccard", "title_token_containment", "title_trigram_jaccard",
    "title_edit_similarity", "model_token_jaccard", "model_token_conflict",
    "model_tokens_both_present", "description_token_jaccard", "descriptions_both_present",
    "relative_price_difference", "prices_both_present", "numeric_token_jaccard",
)


def prepared(title="", description="", price="", raw_id="1", source="abt"):
    return prepare_record(RawRecord(source, raw_id, title, description, "", price))


def named(left, right):
    return dict(zip(FEATURE_NAMES, pair_features(left, right)))


def test_exact_contract_and_identical_attributes():
    assert isinstance(FEATURE_NAMES, tuple)
    assert FEATURE_NAMES == EXPECTED_NAMES
    record = prepared("Canon AB123 12 Camera", "A camera", "$100")
    values = pair_features(record, record)
    assert values == (1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.)
    assert all(type(value) is float for value in values)
    assert pair_features(prepared(), prepared()) == (0.,) * 12


@pytest.mark.parametrize("left,right,jaccard,containment", [
    ("red blue", "blue green", 1 / 3, 1 / 2),
    ("red blue", "blue", 1 / 2, 1),
    ("red blue", "blue red", 1, 1),
    ("red", "xyz", 0, 0),
    ("RED, blue!", "red blue", 1, 1),
    ("", "red", 0, 0),
])
def test_title_word_features(left, right, jaccard, containment):
    values = named(prepared(left), prepared(right))
    assert values["title_token_jaccard"] == pytest.approx(jaccard)
    assert values["title_token_containment"] == pytest.approx(containment)


def test_trigrams_and_edit_have_independent_expected_values():
    # abc1234 -> abc1235: shared abc,bc1,c12,123; union also has 234,235.
    values = named(prepared("abc1234"), prepared("abc1235"))
    assert values["title_trigram_jaccard"] == pytest.approx(4 / 6)
    assert values["title_edit_similarity"] == pytest.approx(6 / 7)
    reordered = named(prepared("red blue"), prepared("blue red"))
    assert reordered["title_token_jaccard"] == 1
    assert reordered["title_edit_similarity"] < 1
    assert reordered["title_trigram_jaccard"] < 1
    assert named(prepared("RED, blue!"), prepared("red blue"))["title_edit_similarity"] == 1


@pytest.mark.parametrize("left,right,expected", [
    ("AB-123", "ab123", (1., 0., 1.)),
    ("ab123", "ab124", (0., 1., 1.)),
    ("ab123 cd456", "ab123 ef789", (1 / 3, 0., 1.)),
    ("ab123", "camera", (0., 0., 0.)),
    ("camera", "lens", (0., 0., 0.)),
])
def test_model_evidence(left, right, expected):
    assert pair_features(prepared(left), prepared(right))[4:7] == expected
    # Even disjoint models yield a row, not a rejection.
    assert feature_matrix([(0, 0)], [prepared(left)], [prepared(right)]).shape == (1, 12)


@pytest.mark.parametrize("left,right,expected", [
    ("12 24 camera", "24 12 lens", 1), ("12", "24", 0),
    ("12 24", "24 36", 1 / 3), ("12", "camera", 0), ("", "", 0),
])
def test_numeric_title_tokens(left, right, expected):
    assert pair_features(prepared(left), prepared(right))[11] == expected


@pytest.mark.parametrize("left,right,expected", [
    ("", "", (0., 0.)), ("camera", "", (0., 0.)),
    ("<p>Red CAMERA!</p>", "red camera", (1., 1.)),
    ("red camera", "blue camera", (1 / 3, 1.)),
    ("<p>!!!</p>", "camera", (0., 0.)),
])
def test_description_missingness(left, right, expected):
    assert pair_features(prepared(description=left), prepared(description=right))[7:9] == expected


@pytest.mark.parametrize("left,right,expected", [
    ("", "", (0., 0.)), ("", "100", (0., 0.)), ("100", "100", (0., 1.)),
    ("$100.00", "80", (.2, 1.)), ("25", "100", (.75, 1.)),
    ("0", "100", (0., 0.)), ("nan", "100", (0., 0.)), ("-10", "20", (0., 0.)),
])
def test_prices(left, right, expected):
    assert pair_features(prepared(price=left), prepared(price=right))[9:11] == expected


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1., 0.])
def test_invalid_prepared_price_is_reported(invalid):
    record = replace(prepared(), price=invalid)
    with pytest.raises(ValueError, match="price"):
        pair_features(record, prepared())


def test_nonfinite_computed_feature_is_reported(monkeypatch):
    monkeypatch.setattr(features, "edit_similarity", lambda a, b: float("nan"))
    with pytest.raises(ValueError, match="title_edit_similarity"):
        pair_features(prepared("a"), prepared("b"))


def test_features_are_symmetric_finite_and_bounded():
    records = [prepared(), prepared("ab123 24", "red", "100"),
               prepared("ab124 12", "blue", "25"), prepared("red blue", "red blue")]
    for left in records:
        for right in records:
            values = pair_features(left, right)
            assert len(values) == 12
            assert np.all(np.isfinite(values))
            assert np.all(np.asarray(values) >= 0) and np.all(np.asarray(values) <= 1)
            assert values == pair_features(right, left)


def test_only_prepared_attributes_determine_features(monkeypatch):
    left = prepared("ab123 12", "camera", "100")
    right = prepared("ab124 12", "red camera", "80")
    expected = pair_features(left, right)
    changed_id = replace(left, raw=replace(left.raw, raw_id="different", source="buy", manufacturer="ignored"))
    assert pair_features(changed_id, right) == expected
    # An inaccessible raw record proves IDs/metadata cannot enter the function.
    class ForbiddenMetadata:
        def __getattr__(self, name):
            raise AssertionError(f"Feature extraction read raw metadata {name}")
    assert pair_features(replace(left, raw=ForbiddenMetadata()), right) == expected
    # Prepared fields are reused: no per-pair preparation is allowed.
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected normalization")
    monkeypatch.setattr("catalogmatch.data.prepare_record", forbidden)
    monkeypatch.setattr("catalogmatch.normalize.normalize_text", forbidden)
    assert feature_matrix([(0, 0)], [left], [right])[0] == pytest.approx(expected)


def test_matrix_shape_dtype_order_duplicates_and_single_pair():
    left = [prepared("red"), prepared("blue")]
    right = [prepared("blue"), prepared("red"), prepared("green")]
    pairs = [(1, 0), (0, 2), (0, 1), (1, 0)]
    matrix = feature_matrix(pairs, left, right)
    assert matrix.shape == (4, 12)
    assert matrix.dtype == np.float64
    for row, (a, b) in enumerate(pairs):
        np.testing.assert_array_equal(matrix[row], pair_features(left[a], right[b]))
    np.testing.assert_array_equal(feature_matrix([(1, 0)], left, right)[0], matrix[0])
    assert feature_matrix([], left, right).shape == (0, 12)
    assert feature_matrix([], [], []).dtype == np.float64
    np.testing.assert_array_equal(feature_matrix(np.array(pairs, dtype=np.int64), left, right), matrix)


@pytest.mark.parametrize("pair", [(-1, 0), (0, -1), (1, 0), (0, 1), (0.0, 0),
                                  (True, 0), (0, False), ("0", 0), (0,), (0, 0, 0), None])
def test_invalid_pairs_not_dropped(pair):
    with pytest.raises(ValueError, match="Pair row 1"):
        feature_matrix([(0, 0), pair], [prepared()], [prepared()])


@pytest.fixture
def cache_inputs(tmp_path):
    return {
        "path": tmp_path / "train.npz", "pairs": [(0, 1), (0, 0)],
        "left_records": [prepared("camera ab123", raw_id="a")],
        "right_records": [prepared("camera ab123", raw_id="b"), prepared("red", raw_id="c")],
        "provenance": {"raw_sha256": {"Abt.csv": "raw-a", "Buy.csv": "raw-b"},
                       "manifest_sha256": "manifest", "split": "train",
                       "retrieval_version": "retrieval-v1",
                       "retrieval_config": {"name": "word10", "word_k": 10, "gram_k": None}},
    }


def test_cache_reuse_skips_feature_computation(cache_inputs, monkeypatch):
    matrix, first = cached_feature_matrix(**cache_inputs)
    assert not first["cache_used"]
    assert first["feature_seconds"] >= 0
    def forbidden(*args):
        raise AssertionError("Cache hit recomputed features")
    monkeypatch.setattr(features, "feature_matrix", forbidden)
    loaded, second = cached_feature_matrix(**cache_inputs)
    assert second["cache_used"] and second["feature_seconds"] == 0
    np.testing.assert_array_equal(matrix, loaded)
    with np.load(cache_inputs["path"], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["pairs"], cache_inputs["pairs"])
        np.testing.assert_array_equal(saved["X"], matrix)


@pytest.mark.parametrize("change", ["feature_order", "schema", "normalizer", "raw", "manifest", "split",
                                    "pairs", "config", "retrieval_version", "record_id", "attributes"])
def test_cache_invalidation(cache_inputs, monkeypatch, change):
    cached_feature_matrix(**cache_inputs)
    if change == "feature_order":
        monkeypatch.setattr(features, "FEATURE_NAMES", tuple(reversed(FEATURE_NAMES)))
    elif change == "schema":
        monkeypatch.setattr(features, "FEATURE_SCHEMA_VERSION", "v2")
    elif change == "normalizer":
        monkeypatch.setattr(features, "NORMALIZER_VERSION", "v2")
    elif change == "raw":
        cache_inputs["provenance"]["raw_sha256"]["Abt.csv"] = "changed"
    elif change == "manifest":
        cache_inputs["provenance"]["manifest_sha256"] = "changed"
    elif change == "split":
        cache_inputs["provenance"]["split"] = "validation"
    elif change == "pairs":
        cache_inputs["pairs"].reverse()
    elif change == "config":
        cache_inputs["provenance"]["retrieval_config"]["word_k"] = 25
    elif change == "retrieval_version":
        cache_inputs["provenance"]["retrieval_version"] = "v2"
    elif change == "record_id":
        record = cache_inputs["left_records"][0]
        cache_inputs["left_records"][0] = replace(record, raw=replace(record.raw, raw_id="changed"))
    else:
        cache_inputs["left_records"][0] = prepared("new attributes")
    matrix, report = cached_feature_matrix(**cache_inputs)
    assert not report["cache_used"]
    assert "identity" in report["cache_reason"]
    np.testing.assert_array_equal(matrix, feature_matrix(cache_inputs["pairs"],
                                  cache_inputs["left_records"], cache_inputs["right_records"]))
    assert cached_feature_matrix(**cache_inputs)[1]["cache_used"]


@pytest.mark.parametrize("damage", ["json", "json_list", "missing_json", "missing_npz", "checksum", "zip",
                                    "shape", "dtype", "nan", "range", "pairs", "pair_dtype", "object", "keys", "npy",
                                    "non_array_X", "non_array_pairs", "duplicate_keys"])
def test_corrupt_cache_regenerates_safely(cache_inputs, damage):
    expected, _ = cached_feature_matrix(**cache_inputs)
    path = cache_inputs["path"]
    metadata_path = path.with_suffix(".json")
    if damage == "json":
        metadata_path.write_text("{")
    elif damage == "json_list":
        metadata_path.write_text("[]")
    elif damage == "missing_json":
        metadata_path.unlink()
    elif damage == "missing_npz":
        path.unlink()
    elif damage == "checksum":
        path.write_bytes(b"broken")
    else:
        matrix = expected.copy()
        pairs = np.array(cache_inputs["pairs"], dtype=np.int64)
        if damage == "shape":
            matrix = matrix[:, :-1]
        elif damage == "dtype":
            matrix = matrix.astype(np.float32)
        elif damage == "nan":
            matrix[0, 0] = np.nan
        elif damage == "range":
            matrix[0, 0] = 2
        elif damage == "pairs":
            pairs = pairs[::-1]
        elif damage == "pair_dtype":
            pairs = pairs.astype(np.float64)
        elif damage == "object":
            matrix = matrix.astype(object)
        if damage == "zip":
            path.write_bytes(b"not an npz")
        elif damage == "npy":
            with path.open("wb") as stream:
                np.save(stream, matrix)
        elif damage in ("non_array_X", "non_array_pairs", "duplicate_keys"):
            with zipfile.ZipFile(path, "w") as archive:
                for key, array in (("X", matrix), ("pairs", pairs)):
                    contents = io.BytesIO()
                    np.save(contents, array)
                    if damage == f"non_array_{key}":
                        archive.writestr(key, b"not a numpy array")
                    else:
                        archive.writestr(key + ".npy", contents.getvalue())
                if damage == "duplicate_keys":
                    # X and X.npy both expose the logical key X to NumPy.
                    archive.writestr("X", b"duplicate logical key")
        elif damage == "keys":
            np.savez(path, other=matrix)
        else:
            np.savez(path, X=matrix, pairs=pairs)
        # A matching checksum must not bypass shape/dtype/value/pair validation.
        metadata = json.loads(metadata_path.read_text())
        metadata["npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata_path.write_text(json.dumps(metadata))
    matrix, report = cached_feature_matrix(**cache_inputs)
    assert not report["cache_used"]
    np.testing.assert_array_equal(matrix, expected)
    assert cached_feature_matrix(**cache_inputs)[1]["cache_used"]


def test_empty_cache_and_input_validation(cache_inputs):
    cache_inputs["pairs"] = []
    assert cached_feature_matrix(**cache_inputs)[0].shape == (0, 12)
    assert cached_feature_matrix(**cache_inputs)[1]["cache_used"]
    cache_inputs["pairs"] = [(0, -1)]
    with pytest.raises(ValueError, match="Pair row 0"):
        cached_feature_matrix(**cache_inputs)
    cache_inputs["pairs"] = []
    del cache_inputs["provenance"]["manifest_sha256"]
    with pytest.raises(ValueError, match="manifest_sha256"):
        cached_feature_matrix(**cache_inputs)


@pytest.fixture
def feature_pipeline(tmp_path, monkeypatch):
    from catalogmatch.data import (
        Dataset, audit_dataset, build_positive_graph, connected_components, make_manifest,
        sha256_file, split_components, validate_partitions,
    )
    left = tuple(RawRecord("abt", str(i), f"camera ab{i:03d}", "camera", "", "100") for i in range(10))
    right = tuple(replace(record, source="buy", raw_price="80") for record in left)
    dataset = Dataset(left, right, tuple((str(i), str(i)) for i in range(10)), {"synthetic": "fixture"})
    components = connected_components(build_positive_graph(dataset.records, dataset.gold_pairs))
    partitions = split_components(components)
    manifest = tmp_path / "manifest.json"
    cli.write_json(manifest, make_manifest(dataset, components, partitions))
    audit = audit_dataset(dataset, original=False)
    audit.update(validate_partitions(dataset.records, dataset.gold_pairs, components, partitions))
    audit["manifest_sha256"] = sha256_file(manifest)
    # Only official-data auditing is replaced; checked prepared loading remains real.
    monkeypatch.setattr(cli, "verify_original", lambda *args: (dataset, partitions, audit))
    processed = tmp_path / "processed"
    cli.prepare_data(tmp_path / "raw", manifest, processed)
    selection = tmp_path / "retrieval"
    cli.run_retrieval_eval(processed, tmp_path / "raw", manifest, selection)
    return dataset, partitions, audit, processed, manifest, selection / "selected.json"


def test_selected_feature_cli_partition_isolation_and_reuse(feature_pipeline, tmp_path, monkeypatch, capsys):
    dataset, partitions, audit, processed, manifest, selected = feature_pipeline
    selected_bytes = selected.read_bytes()
    output = tmp_path / "features"
    original_load = cli.load_prepared_partition
    loaded = []
    def checked_load(*args):
        assert args[-1] in ("train", "validation")
        result = original_load(*args)
        loaded.append((args[-1], result))
        return result
    def forbidden(*args):
        raise AssertionError("Feature generation must not evaluate matching quality")
    monkeypatch.setattr(cli, "load_prepared_partition", checked_load)
    monkeypatch.setattr(cli, "evaluate_retrieval", forbidden)
    args = ["features", "--data", str(processed), "--manifest", str(manifest),
            "--selected", str(selected), "--out", str(output)]
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert [name for name, _ in loaded] == ["train", "validation"]
    assert report["retrieval_config"]["name"] == "word10"
    assert report["feature_names"] == list(EXPECTED_NAMES)
    for row, (name, (left, right)) in zip(report["partitions"], loaded):
        assert not row["cache_used"]
        assert row["shape"] == [row["candidate_pairs"], 12]
        with np.load(output / f"{name}.npz", allow_pickle=False) as saved:
            pairs = saved["pairs"]
            np.testing.assert_array_equal(saved["X"], feature_matrix(pairs, left, right))
            words, grams = cli.build_indexes(right)
            expected_pairs = []
            config = cli.RetrievalConfig("word10", word_k=10)
            for position, record in enumerate(left):
                for candidate in cli.retrieve_candidates(record, words, grams, config):
                    expected_pairs.append((position, candidate.target_position))
            np.testing.assert_array_equal(pairs, expected_pairs)
            for a, b in pairs:
                assert left[a].raw.key in partitions[name]
                assert right[b].raw.key in partitions[name]
    monkeypatch.setattr(features, "pair_features", forbidden)
    assert cli.main(args) == 0
    assert all(row["cache_used"] for row in json.loads(capsys.readouterr().out)["partitions"])
    assert selected.read_bytes() == selected_bytes


@pytest.mark.parametrize("key", ["raw_sha256", "manifest_sha256", "normalizer_version", "retrieval_version",
                                 "prepared_sha256", "split_version", "split", "selected_config"])
def test_reject_stale_selected_configuration(feature_pipeline, key):
    dataset, _, audit, processed, _, selected = feature_pipeline
    report = json.loads(selected.read_text())
    report[key] = {} if key == "selected_config" else "changed"
    selected.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="configuration"):
        cli.load_selected_config(selected, processed, dataset, audit)


def test_reject_changed_selection_and_malformed_json(feature_pipeline):
    dataset, _, audit, processed, _, selected = feature_pipeline
    report = json.loads(selected.read_text())
    report["selected_config"] = {"name": "word25", "word_k": 25, "gram_k": None}
    selected.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="saved validation selection"):
        cli.load_selected_config(selected, processed, dataset, audit)
    selected.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        cli.load_selected_config(selected, processed, dataset, audit)


def test_inspection_cli_displays_raw_attributes_and_features(feature_pipeline, capsys):
    _, partitions, _, processed, manifest, _ = feature_pipeline
    abt_id = next(raw_id for source, raw_id in partitions["train"] if source == "abt")
    buy_id = next(raw_id for source, raw_id in partitions["train"] if source == "buy")
    args = ["inspect-features", "--data", str(processed), "--manifest", str(manifest),
            "--abt-id", abt_id, "--buy-id", buy_id]
    assert cli.main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["abt"]["name"].startswith("camera ab")
    assert report["abt"]["description"] == "camera"
    assert report["abt"]["raw_price"] == "100"
    assert report["buy"]["raw_price"] == "80"
    assert tuple(report["features"]) == EXPECTED_NAMES
    assert cli.main(args + ["--abt-id", "unknown"]) == 1
    assert "Unknown abt ID" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["features", "inspect-features"])
def test_feature_cli_rejects_test_split(command):
    args = [command, "--split", "test"]
    if command == "inspect-features":
        args += ["--abt-id", "1", "--buy-id", "1"]
    with pytest.raises(SystemExit):
        cli.main(args)


def test_feature_output_cannot_overwrite_inputs(tmp_path, capsys):
    assert cli.main(["features", "--data", str(tmp_path), "--out", str(tmp_path)]) == 1
    assert "separate from prepared" in capsys.readouterr().err
    assert cli.main(["features", "--out", "runs/retrieval"]) == 1
    assert "separate from the selected" in capsys.readouterr().err


def test_gold_labels_and_split_metadata_do_not_enter_features(dataset):
    left = [prepare_record(record) for record in dataset.abt]
    right = [prepare_record(record) for record in dataset.buy]
    pairs = [(0, 0), (0, 1)]
    expected = feature_matrix(pairs, left, right)
    changed_gold = replace(dataset, gold_pairs=())
    # Neither matrix nor pair extraction accepts any gold/split argument.
    actual = feature_matrix(pairs, [prepare_record(record) for record in changed_gold.abt],
                            [prepare_record(record) for record in changed_gold.buy])
    np.testing.assert_array_equal(actual, expected)


def test_worked_noisy_title_example():
    left = prepared("Canon AB-123 Camera, 12 MP!", "<p>Digital camera</p>", "$100.00")
    right = prepared("CANON AB123 camera 12MP", "Digital camera kit", "80")
    assert left.normalized_title == "canon ab 123 camera 12 mp"
    assert right.normalized_title == "canon ab123 camera 12mp"
    assert left.model_tokens == {"ab123"}
    assert right.model_tokens == {"ab123", "12mp"}
    assert left.numeric_title_tokens == {"123", "12"}
    assert right.numeric_title_tokens == set()
    assert pair_features(left, right) == pytest.approx(
        (.25, .5, 17 / 26, .92, .5, 0., 1., 2 / 3, 1., .2, 1., 0.))
