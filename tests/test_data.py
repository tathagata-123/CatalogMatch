import csv
import hashlib

import pytest

from catalogmatch.data import (
    DataValidationError, RawRecord, SCHEMAS, audit_dataset, load_gold,
    load_records, read_csv, sha256_file,
)


def write_csv(path, rows):
    with path.open("w", encoding="latin-1", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_load_schemas_missing_values_and_namespaced_ids(dataset):
    assert len(dataset.abt) == len(dataset.buy) == 5
    assert len(dataset.gold_pairs) == 5
    assert dataset.abt[0].raw_id == dataset.buy[0].raw_id == "shared"
    assert dataset.abt[0].key != dataset.buy[0].key
    assert all(record.manufacturer == "" for record in dataset.abt)
    assert dataset.abt[1].description == ""
    assert dataset.abt[1].raw_price == ""
    assert dataset.buy[2].manufacturer == ""
    report = audit_dataset(dataset, original=False)
    assert report["rows"] == {"abt": 5, "buy": 5, "gold": 5}
    assert report["missing"]["abt"] == {"id": 0, "name": 0, "description": 2, "price": 1}
    assert report["missing"]["buy"] == {"id": 0, "name": 0, "description": 2, "manufacturer": 1, "price": 1}
    assert report["malformed_prices"] == {"abt": 2, "buy": 0}


@pytest.mark.parametrize("columns", [
    ["id", "name", "price"],
    ["id", "name", "description", "price", "manufacturer"],
    ["id", "name", "description", "Price"],
    ["id", "name", "name", "price"],
    ["name", "id", "description", "price"],
])
def test_exact_schema_rejected(tmp_path, columns):
    path = tmp_path / "Abt.csv"
    write_csv(path, [columns])
    with pytest.raises(DataValidationError, match="expected columns"):
        load_records(path, "abt")


@pytest.mark.parametrize("row", [["1", "camera", "description"], ["1", "camera", "", "10", "extra"]])
def test_absent_and_extra_fields_are_errors(tmp_path, row):
    path = tmp_path / "Abt.csv"
    write_csv(path, [SCHEMAS["Abt.csv"], row])
    with pytest.raises(DataValidationError, match="missing fields or extra fields"):
        load_records(path, "abt")


def test_latin1_original_bytes_and_string_id(tmp_path):
    path = tmp_path / "Abt.csv"
    write_csv(path, [SCHEMAS["Abt.csv"], ["001", "Café £", "Über", ""]])
    record, = load_records(path, "abt")
    assert record.raw_id == "001"
    assert record.name == "Café £"
    assert record.description == "Über"
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_duplicate_ids(tmp_path):
    path = tmp_path / "Abt.csv"
    write_csv(path, [SCHEMAS["Abt.csv"], ["1", "A", "", ""], ["1", "B", "", ""]])
    with pytest.raises(DataValidationError, match="duplicate ID"):
        load_records(path, "abt")


@pytest.mark.parametrize("raw_id", ["", "   "])
def test_blank_id(tmp_path, raw_id):
    path = tmp_path / "Abt.csv"
    write_csv(path, [SCHEMAS["Abt.csv"], [raw_id, "camera", "", ""]])
    with pytest.raises(DataValidationError, match="nonblank strings"):
        load_records(path, "abt")


def test_unknown_source(fixture_dir):
    with pytest.raises(DataValidationError, match="Unknown source"):
        load_records(fixture_dir / "Abt.csv", "other")
    with pytest.raises(DataValidationError, match="Unknown source"):
        RawRecord("other", "1", "", "", "", "")


def test_duplicate_gold_pairs(tmp_path, dataset):
    path = tmp_path / "gold.csv"
    write_csv(path, [["idAbt", "idBuy"], ["shared", "shared"], ["shared", "shared"]])
    with pytest.raises(DataValidationError, match="duplicate gold pair"):
        load_gold(path, dataset.abt, dataset.buy)


@pytest.mark.parametrize("pair", [("missing", "shared"), ("shared", "missing"), ("", "shared")])
def test_gold_foreign_keys(tmp_path, dataset, pair):
    path = tmp_path / "gold.csv"
    write_csv(path, [["idAbt", "idBuy"], pair])
    with pytest.raises(DataValidationError, match="nonexistent gold reference"):
        load_gold(path, dataset.abt, dataset.buy)


@pytest.mark.parametrize("text", ['"unterminated', 'id,name,description,price\n1,"unterminated'])
def test_malformed_csv_has_informative_error(tmp_path, text):
    path = tmp_path / "Abt.csv"
    path.write_text(text, encoding="latin-1")
    with pytest.raises(DataValidationError, match="Abt.csv, line"):
        read_csv(path, SCHEMAS["Abt.csv"])


def test_original_audit_rejects_synthetic_counts(dataset):
    with pytest.raises(DataValidationError, match="Unexpected original row counts"):
        audit_dataset(dataset)
