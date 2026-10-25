"""Optional offline integration test against the unmodified official CSV bytes."""

from pathlib import Path

import pytest

from catalogmatch.data import (
    EXPECTED_MISSING, EXPECTED_ROWS, EXPECTED_SHAPES, EXPECTED_SPLITS, SCHEMAS,
    connected_components, build_positive_graph, prepare_record, sha256_file,
    split_components, verify_original,
)


@pytest.mark.real_data
def test_original_dataset_audit_and_supplied_manifest():
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    available = [filename for filename in SCHEMAS if (raw_dir / filename).is_file()]
    if not available:
        pytest.skip("Original Abt–Buy CSVs absent; see README for the official download")
    assert len(available) == 3, f"Incomplete real dataset: found {available}"
    manifest = root / "CatalogMatch_Split_Manifest.json"
    original_manifest_hash = sha256_file(manifest)
    dataset, partitions, report = verify_original(raw_dir, manifest)
    assert report["rows"] == EXPECTED_ROWS
    assert report["missing"] == EXPECTED_MISSING
    assert report["malformed_prices"] == {"abt": 0, "buy": 0}
    assert report["component_count"] == 1076
    assert report["component_shapes"] == EXPECTED_SHAPES
    assert report["partitions"] == EXPECTED_SPLITS
    assert report["gold_pairs_by_partition"] == {"train": 658, "validation": 220, "test": 219}
    assert all(report["checks"].values())
    assert len(dataset.records) == sum(len(vertices) for vertices in partitions.values()) == 2173
    assert len(dataset.gold_pairs) == sum(report["gold_pairs_by_partition"].values()) == 1097
    components = connected_components(build_positive_graph(dataset.records, dataset.gold_pairs))
    assert split_components(components) == partitions
    # Every official row is preparable, including all missing prices/descriptions.
    for raw in dataset.records:
        assert prepare_record(raw).raw is raw
    assert sha256_file(manifest) == original_manifest_hash
