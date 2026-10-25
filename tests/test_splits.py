from copy import deepcopy
from dataclasses import replace
import random

import pytest

from catalogmatch.data import (
    DataValidationError, RawRecord, SPLIT_NAMES, build_positive_graph,
    connected_components, split_components, validate_partitions, verify_manifest,
)


def test_graph_contains_only_positive_edges_and_all_vertices(dataset):
    graph = build_positive_graph(dataset.records, dataset.gold_pairs)
    assert set(graph) == {record.key for record in dataset.records}
    assert graph[("abt", "shared")] == {("buy", "shared"), ("buy", "b2")}
    assert graph[("buy", "shared")] == {("abt", "shared")}
    assert graph[("buy", "b3")] == {("abt", "a2"), ("abt", "a3")}
    assert graph[("abt", "a5")] == graph[("buy", "b5")] == set()
    actual_edges = set()
    for vertex, neighbors in graph.items():
        for neighbor in neighbors:
            assert vertex in graph[neighbor]
            if vertex[0] == "abt":
                actual_edges.add((vertex[1], neighbor[1]))
    # Every non-gold (negative) pair stays absent, including singleton candidates.
    assert actual_edges == set(dataset.gold_pairs)
    assert sum(len(neighbors) for neighbors in graph.values()) == 2 * len(dataset.gold_pairs)


def test_labeled_negative_candidate_is_not_a_graph_input(dataset):
    with pytest.raises(DataValidationError, match="positive pair of string IDs"):
        build_positive_graph(dataset.records, (("a5", "b5", 0),))


def test_expected_components(components):
    assert components == [
        (("abt", "a2"), ("abt", "a3"), ("buy", "b3")),
        (("abt", "a4"), ("buy", "b4")),
        (("abt", "a5"),),
        (("abt", "shared"), ("buy", "b2"), ("buy", "shared")),
        (("buy", "b5"),),
    ]


def test_graph_rejects_duplicate_vertices_edges_and_unknown_endpoints(dataset):
    with pytest.raises(DataValidationError, match="Duplicate record key"):
        build_positive_graph(dataset.records + dataset.abt[:1], dataset.gold_pairs)
    with pytest.raises(DataValidationError, match="Duplicate gold pair"):
        build_positive_graph(dataset.records, dataset.gold_pairs * 2)
    with pytest.raises(DataValidationError, match="Nonexistent gold reference"):
        build_positive_graph(dataset.records, (("missing", "shared"),))


def test_iterative_dfs_handles_long_positive_chain():
    # Longer than Python's usual recursion limit, with cycles/branching at vertices.
    records = []
    pairs = []
    for index in range(1500):
        raw_id = str(index)
        records.append(RawRecord("abt", raw_id, "", "", "", ""))
        records.append(RawRecord("buy", raw_id, "", "", "", ""))
        pairs.append((raw_id, raw_id))
        if index:
            pairs.append((raw_id, str(index - 1)))
    graph = build_positive_graph(tuple(records), tuple(pairs))
    components = connected_components(graph)
    assert len(components) == 1
    assert len(components[0]) == 3000


def test_dfs_order_is_independent_of_record_and_gold_order(dataset, components):
    graph = build_positive_graph(dataset.records[::-1], dataset.gold_pairs[::-1])
    assert connected_components(graph) == components


def test_deterministic_split_bucket_order_and_one_rng():
    # Five components per bucket gives exactly 3/1/1 from each; hard-coded
    # allocations detect bucket-order changes and resetting the RNG per bucket.
    components = []
    for index in range(5):
        components.append((("abt", f"m{index}"), ("buy", f"m{index}")))
        components.append((("abt", f"a{index}"),))
        components.append((("buy", f"b{index}"),))
    partitions = split_components(components)
    assert partitions == split_components(components[::-1])
    assert partitions["train"] == tuple(sorted([
        ("abt", "m3"), ("buy", "m3"), ("abt", "m1"), ("buy", "m1"),
        ("abt", "m2"), ("buy", "m2"),
        ("abt", "a3"), ("abt", "a2"), ("abt", "a0"),
        ("buy", "b3"), ("buy", "b1"), ("buy", "b2"),
    ]))
    assert partitions["validation"] == (("abt", "a4"), ("abt", "m4"), ("buy", "b0"), ("buy", "m4"))
    assert partitions["test"] == (("abt", "a1"), ("abt", "m0"), ("buy", "b4"), ("buy", "m0"))
    random.seed(987)
    assert split_components(components) == partitions  # No global random state.


def test_floor_allocation_for_small_buckets(components):
    result = split_components(components)
    assert len(result["train"]) in (2, 3)  # One of three matched components.
    assert result["validation"] == ()
    assert ("abt", "a5") in result["test"]
    assert ("buy", "b5") in result["test"]
    assert split_components([]) == {name: () for name in SPLIT_NAMES}


@pytest.mark.parametrize("components, message", [
    ([()], "Empty connected component"),
    ([(('other', '1'),)], "Unknown component sources"),
    ([(('abt', '1'),), (('abt', '1'),)], "Repeated component vertex"),
])
def test_invalid_split_components(components, message):
    with pytest.raises(DataValidationError, match=message):
        split_components(components)


def test_exact_coverage_components_edges_and_gold_accounting(dataset, components, partitions):
    report = validate_partitions(dataset.records, dataset.gold_pairs, components, partitions)
    all_vertices = [vertex for vertices in partitions.values() for vertex in vertices]
    assert len(all_vertices) == len(set(all_vertices)) == len(dataset.records)
    assert set(all_vertices) == {record.key for record in dataset.records}
    owner = {vertex: name for name, vertices in partitions.items() for vertex in vertices}
    for component in components:
        assert len({owner[vertex] for vertex in component}) == 1
    for left, right in dataset.gold_pairs:
        assert owner[("abt", left)] == owner[("buy", right)]
    assert sum(report["gold_pairs_by_partition"].values()) == len(dataset.gold_pairs)
    assert sum(sizes["components"] for sizes in report["partitions"].values()) == len(components)


def test_partition_rejects_duplicates_missing_and_unknown_records(dataset, components, partitions):
    for alteration, message in [("duplicate", "multiple times"), ("missing", "Missing partition"), ("unknown", "Unknown partition")]:
        changed = dict(partitions)
        vertex = changed["train"][0]
        if alteration == "duplicate":
            changed["test"] += (vertex,)
        elif alteration == "missing":
            changed["train"] = changed["train"][1:]
        else:
            changed["train"] += (("abt", "unknown"),)
        with pytest.raises(DataValidationError, match=message):
            validate_partitions(dataset.records, dataset.gold_pairs, components, changed)


def test_split_endpoint_and_gold_edge_leakage_are_errors(dataset, components, partitions):
    changed = dict(partitions)
    vertex = next(vertex for vertex in changed["train"] if vertex[0] == "abt")
    changed["train"] = tuple(item for item in changed["train"] if item != vertex)
    changed["test"] += (vertex,)
    with pytest.raises(DataValidationError, match="component crosses partitions"):
        validate_partitions(dataset.records, dataset.gold_pairs, components, changed)
    # Even a caller supplying singleton 'components' cannot hide a crossing edge.
    singletons = [(record.key,) for record in dataset.records]
    with pytest.raises(DataValidationError, match="Gold edge crosses partitions"):
        validate_partitions(dataset.records, dataset.gold_pairs, singletons, changed)


@pytest.mark.parametrize("alteration, message", [
    ("missing", "do not cover all records"), ("duplicate", "Repeated component vertex"),
    ("empty", "Empty connected component"), ("unknown", "Unknown component record"),
])
def test_invalid_component_coverage(dataset, components, partitions, alteration, message):
    changed = list(components)
    if alteration == "missing":
        changed.pop()
    elif alteration == "duplicate":
        changed.append(components[0])
    elif alteration == "empty":
        changed.append(())
    else:
        changed.append((("abt", "unknown"),))
    with pytest.raises(DataValidationError, match=message):
        validate_partitions(dataset.records, dataset.gold_pairs, changed, partitions)


def test_exact_manifest_consistency(dataset, components, partitions, manifest):
    saved = deepcopy(manifest)
    assert verify_manifest(dataset, manifest, components) == partitions
    assert manifest == saved


@pytest.mark.parametrize("alteration, message", [
    ("hash", "SHA-256 mismatch"), ("seed", "seed"), ("version", "split_version"),
    ("count", "component_count mismatch"), ("duplicate", "multiple times"),
    ("missing", "Missing partition records"), ("id_type", "list of strings"),
    ("split_name", "exactly"), ("count_type", "component_count"),
])
def test_manifest_mismatches(dataset, components, manifest, alteration, message):
    changed = deepcopy(manifest)
    if alteration == "hash":
        changed["raw_sha256"]["Abt.csv"] = "0" * 64
    elif alteration == "seed":
        changed["seed"] = 43
    elif alteration == "version":
        changed["split_version"] = "unknown"
    elif alteration == "count":
        changed["splits"]["train"]["component_count"] += 1
    elif alteration == "duplicate":
        changed["splits"]["test"]["abt_ids"] += changed["splits"]["train"]["abt_ids"]
    elif alteration == "missing":
        changed["splits"]["train"]["abt_ids"].pop()
    elif alteration == "id_type":
        changed["splits"]["train"]["abt_ids"] = [123]
    elif alteration == "split_name":
        changed["splits"]["dev"] = changed["splits"].pop("validation")
    else:
        changed["splits"]["train"]["component_count"] = True
    with pytest.raises(DataValidationError, match=message):
        verify_manifest(dataset, changed, components)


def test_changed_input_bytes_fail_hash_verification(dataset, components, manifest):
    changed_hashes = dict(dataset.raw_sha256)
    changed_hashes["Buy.csv"] = "f" * 64
    with pytest.raises(DataValidationError, match="SHA-256 mismatch"):
        verify_manifest(replace(dataset, raw_sha256=changed_hashes), manifest, components)


def test_leakage_free_but_different_manifest_is_rejected(dataset, components, manifest):
    changed = deepcopy(manifest)
    # Swapping whole partitions retains coverage, edges, and component counts,
    # but must fail deterministic reproduction rather than being accepted/repaired.
    changed["splits"]["train"], changed["splits"]["test"] = changed["splits"]["test"], changed["splits"]["train"]
    with pytest.raises(DataValidationError, match="seed-42 reproduction"):
        verify_manifest(dataset, changed, components)
