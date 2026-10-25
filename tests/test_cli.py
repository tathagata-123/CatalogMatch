import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from catalogmatch import cli
from catalogmatch.data import audit_dataset, sha256_file, validate_partitions


@pytest.fixture
def checked_fixture(monkeypatch, dataset, components, partitions):
    report = audit_dataset(dataset, original=False)
    report.update(validate_partitions(dataset.records, dataset.gold_pairs, components, partitions))
    report["manifest_sha256"] = "fixture-manifest"
    # Exercise output orchestration on tiny validated fixtures. Original-size and
    # original-hash enforcement is separately exercised by test_real_data.py.
    monkeypatch.setattr(cli, "verify_original", lambda raw, manifest: (dataset, partitions, dict(report)))
    return dataset, partitions


def test_prepare_outputs_preserve_raw_attributes_and_metadata(tmp_path, checked_fixture):
    dataset, partitions = checked_fixture
    report = cli.prepare_data(Path("unused"), Path("unused.json"), tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert report["prepared_records"] == len(rows) == len(dataset.records)
    assert {(row["raw"]["source"], row["raw"]["raw_id"]) for row in rows} == {r.key for r in dataset.records}
    originals = {record.key: record for record in dataset.records}
    for row in rows:
        key = (row["raw"]["source"], row["raw"]["raw_id"])
        assert row["raw"]["name"] == originals[key].name
        assert row["raw"]["raw_price"] == originals[key].raw_price
        assert key in partitions[row["metadata"]["split"]]
        assert not {"raw_id", "source", "split", "position", "gold"} & row["attributes"].keys()
    for split in partitions:
        for source in ("abt", "buy"):
            group = [row for row in rows if row["metadata"]["split"] == split and row["raw"]["source"] == source]
            assert [row["metadata"]["position"] for row in group] == list(range(len(group)))
            assert [row["raw"]["raw_id"] for row in group] == sorted(row["raw"]["raw_id"] for row in group)
    for filename, digest in report["output_sha256"].items():
        assert sha256_file(tmp_path / filename) == digest
    assert json.loads((tmp_path / "audit.json").read_text()) == report


def test_baseline_outputs_only_training_results(tmp_path, checked_fixture):
    dataset, partitions = checked_fixture
    report = cli.run_baseline(Path("unused"), Path("unused.json"), tmp_path, 0.5)
    with (tmp_path / "matches.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert report["split"] == "train"
    assert report["threshold"] == 0.5
    assert report["threshold_tuned"] is False
    assert report["predicted_pairs"] == len(rows)
    for row in rows:
        assert ("abt", row["idAbt"]) in partitions["train"]
        assert ("buy", row["idBuy"]) in partitions["train"]
        assert float(row["lexical_score"]) >= 0.5
    train_gold = [(a, b) for a, b in dataset.gold_pairs if ("abt", a) in partitions["train"]]
    predicted = {(row["idAbt"], row["idBuy"]) for row in rows}
    assert report["gold_pairs"] == len(train_gold)
    assert report["true_positives"] == len(predicted & set(train_gold))
    assert report["false_negatives"] == len(set(train_gold) - predicted)
    assert report["matches_sha256"] == sha256_file(tmp_path / "matches.csv")
    assert json.loads((tmp_path / "baseline.json").read_text()) == report


def test_module_entrypoint_works_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, "-m", "catalogmatch", "--help"], cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 0
    assert "prepare" in result.stdout
    assert "baseline" in result.stdout


def test_prepare_main(tmp_path, checked_fixture, capsys):
    output = tmp_path / "processed"
    assert cli.main(["prepare", "--output-dir", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["prepared_records"] == 10


def test_missing_dataset_cli_error(tmp_path, capsys):
    assert cli.main(["prepare", "--raw-dir", str(tmp_path / "missing")]) == 1
    assert "Abt.csv" in capsys.readouterr().err


def test_synthetic_dataset_cannot_masquerade_as_original(fixture_dir, tmp_path, capsys):
    output = tmp_path / "processed"
    assert cli.main(["prepare", "--raw-dir", str(fixture_dir), "--output-dir", str(output)]) == 1
    assert "Unexpected original row counts" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("threshold", ["-1", "2", "nan", "inf", "bad"])
def test_cli_rejects_invalid_threshold(threshold):
    with pytest.raises(SystemExit) as error:
        cli.main(["baseline", "--threshold", threshold])
    assert error.value.code == 2


@pytest.mark.parametrize("location", ["raw", "raw/subdir", "manifest_parent"])
def test_cli_cannot_overwrite_inputs(tmp_path, location, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    manifest = tmp_path / "manifest_parent" / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text("preserve this", encoding="utf-8")
    result = cli.main(["prepare", "--raw-dir", str(raw), "--manifest", str(manifest),
                       "--output-dir", str(tmp_path / location)])
    assert result == 1
    assert "separate from raw inputs" in capsys.readouterr().err
    assert manifest.read_text() == "preserve this"
