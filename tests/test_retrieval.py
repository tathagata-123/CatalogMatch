"""Independent full-scan/full-sort oracles, heap invariants, and split-safe evaluation."""

import builtins
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest

from catalogmatch import cli, retrieval
from catalogmatch.data import (
    DataValidationError, RawRecord, audit_dataset, build_positive_graph,
    connected_components, load_dataset, load_prepared_partition, make_manifest,
    partition_records, prepare_record, sha256_file, split_components, validate_partitions,
)
from catalogmatch.normalize import NORMALIZER_VERSION
from catalogmatch.retrieval import (
    BoundedTopK, InvertedIndex, RETRIEVAL_CONFIGS, RetrievalConfig, RetrievalStats,
    bounded_top_k, build_indexes, evaluate_retrieval, retrieve_candidates, select_configuration,
)


def exhaustive_cosine(documents, query, k):
    """Independent oracle: scan every document and fully sort eligible results."""
    documents = [set(document) for document in documents]
    query = set(query)
    vocabulary = set(query)
    for document in documents:
        vocabulary.update(document)
    weights = {}
    for term in vocabulary:
        frequency = sum(term in document for document in documents)
        weights[term] = 1 + math.log((len(documents) + 1) / (frequency + 1))
    query_norm = math.sqrt(math.fsum(weights[term] ** 2 for term in query))
    scores = []
    for position, document in enumerate(documents):
        if not document.intersection(query):
            continue
        document_norm = math.sqrt(math.fsum(weights[term] ** 2 for term in document))
        dot = sum(weights[term] ** 2 for term in sorted(query) if term in document)
        scores.append((position, dot / (query_norm * document_norm)))
    scores.sort(key=lambda item: (-item[1], item[0]))
    return scores[:k]


def assert_scores_equal(actual, expected):
    assert [position for position, _ in actual] == [position for position, _ in expected]
    assert [score for _, score in actual] == pytest.approx([score for _, score in expected], abs=1e-14)


def prepared(title, source="abt", raw_id="query"):
    return prepare_record(RawRecord(source, raw_id, title, "", "", ""))


def full_sort(scores, k):
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


def assert_heap_invariant(heap):
    assert len(heap.entries) <= heap.k
    for child in range(1, len(heap.entries)):
        assert heap.entries[(child - 1) // 2] <= heap.entries[child]
    if heap.entries:
        assert heap.entries[0] == min(heap.entries)


@pytest.mark.parametrize("scores,k", [
    ({}, 1), ({4: 0.6}, 1), ({4: 0.6}, 10),
    ({9: 0.9, 2: 0.9, 1: 0.2}, 1),
    ({9: 0.9, 2: 0.9, 1: 0.2}, 2),
    ({7: 1.0, 3: 1.0, 8: 1.0, 0: 1.0}, 3),
    ({4: 0.1, 6: 0.9, 8: 0.5}, 20),
])
def test_heap_agrees_with_full_sort(scores, k):
    assert bounded_top_k(scores, k) == full_sort(scores, k)


@pytest.mark.parametrize("k", [1, 2, 7, 50, 150])
def test_heap_invariants_and_shuffled_order(k):
    rng = random.Random(904)
    scores = {position: rng.choice([0.0, 0.2, 0.5, 0.9, 1.0]) for position in range(100)}
    for _ in range(6):
        items = list(scores.items())
        rng.shuffle(items)
        heap = BoundedTopK(k)
        seen = {}
        for position, score in items:
            heap.add(position, score)
            seen[position] = score
            assert_heap_invariant(heap)
            assert heap.best_first() == full_sort(seen, k)
        assert heap.best_first() == full_sort(scores, k)


def test_sift_up_and_sift_down():
    heap = BoundedTopK(8)
    heap.entries = [(0.2, -8), (0.5, -7), (0.3, -6), (0.7, -5), (0.6, -4)]
    heap.entries.append((0.1, -3))
    heap.sift_up(5)
    assert_heap_invariant(heap)
    assert heap.entries[0] == (0.1, -3)
    heap.entries[0] = (0.9, -2)
    heap.sift_down(0)
    assert_heap_invariant(heap)
    assert heap.entries[0] == (0.2, -8)


def test_root_replacement_requires_strictly_better_key():
    heap = BoundedTopK(2)
    heap.add(3, 0.5)
    heap.add(1, 0.5)
    assert heap.entries[0] == (0.5, -3)
    before = list(heap.entries)
    heap.add(9, 0.5)
    heap.add(3, 0.5)  # Exact same key must not replace the root.
    assert heap.entries == before
    heap.add(2, 0.5)
    assert heap.best_first() == [(1, 0.5), (2, 0.5)]
    heap.add(8, 0.8)
    assert_heap_invariant(heap)
    assert heap.best_first() == [(8, 0.8), (1, 0.5)]


def test_top_k_sorts_only_retained_entries(monkeypatch):
    sizes = []

    def checked_sorted(items, **kwargs):
        sizes.append(len(items))
        return builtins.sorted(items, **kwargs)

    monkeypatch.setattr(retrieval, "sorted", checked_sorted, raising=False)
    assert len(bounded_top_k({i: i / 100 for i in range(100)}, 3)) == 3
    assert sizes == [3]


@pytest.mark.parametrize("k", [0, -1, -10, 1.5, True])
def test_invalid_k_rejected_even_for_empty_inputs(k):
    with pytest.raises(ValueError, match="positive integer"):
        bounded_top_k({}, k)
    with pytest.raises(ValueError, match="positive integer"):
        InvertedIndex.build([]).query([], k)
    with pytest.raises(ValueError, match="positive integer"):
        RetrievalConfig("bad", word_k=k)


def test_configuration_requires_a_channel():
    with pytest.raises(ValueError, match="at least one channel"):
        RetrievalConfig("bad")


def test_hand_calculated_cosine_including_unseen_query_term():
    index = InvertedIndex.build([["a", "b", "b"], ["a"], ["c"]])
    # N=3: df(a)=2, df(b)=df(c)=1, df(unseen)=0.
    a = 1 + math.log(4 / 3)
    b = 1 + math.log(2)
    unseen = 1 + math.log(4)
    d0_norm = math.sqrt(a * a + b * b)
    q_norm = math.sqrt(a * a + b * b + unseen * unseen)
    dot0 = a * a + b * b
    dot1 = a * a
    assert index.n_records == 3
    assert index.df == {"a": 2, "b": 1, "c": 1}
    assert index.idf == pytest.approx({"a": a, "b": b, "c": b})
    assert index.norms == pytest.approx([d0_norm, a, b])
    assert index.postings == {"a": [0, 1], "b": [0], "c": [2]}
    assert_scores_equal(index.query(["a", "b", "unseen"], 10), [
        (0, dot0 / (q_norm * d0_norm)), (1, dot1 / (q_norm * a)),
    ])
    assert index.query(["a", "b"], 10)[0][1] == pytest.approx(1.0)
    assert index.query(["a", "b", "unseen"], 10)[0][1] < 1.0


@pytest.mark.parametrize("documents,query", [
    ([], []), ([], ["unknown"]), ([[], []], ["x"]),
    ([["a"], ["b"]], []), ([["a"]], ["unknown"]),
    ([["a", "a", "b"], ["b"], []], ["a", "a", "b"]),
    ([["common", str(i)] for i in range(30)], ["common"]),
    ([["a"], ["a"], ["b"]], ["a", "unknown"]),
])
def test_index_matches_tiny_exhaustive_oracle(documents, query):
    index = InvertedIndex.build(documents)
    for k in (1, 2, 50):
        assert_scores_equal(index.query(query, k), exhaustive_cosine(documents, query, k))


@pytest.mark.parametrize("seed", [7, 42, 151])
def test_randomized_cosine_against_exhaustive_oracle(seed):
    rng = random.Random(seed)
    vocabulary = [f"term{i}" for i in range(20)]
    for _ in range(25):
        documents = []
        for _ in range(rng.randrange(40)):
            documents.append(rng.choices(vocabulary, k=rng.randrange(15)))
        index = InvertedIndex.build(documents)
        for _ in range(8):
            query = rng.choices(vocabulary + ["unseen1", "unseen2"], k=rng.randrange(15))
            k = rng.choice([1, 2, 5, 10, 100])
            assert_scores_equal(index.query(query, k), exhaustive_cosine(documents, query, k))


def test_postings_unique_sorted_and_query_repeats_do_not_change_scores():
    documents = [["z", "a", "z"], ["z", "b"], ["a", "z"], [], ["z"]]
    index = InvertedIndex.build(iter(documents))
    assert index.postings == {"z": [0, 1, 2, 4], "a": [0, 2], "b": [1]}
    assert index.query(["z", "a", "z", "a"], 9) == index.query(["a", "z"], 9)
    assert index.norms[3] == 0.0


def test_posting_counters_and_no_full_target_scan():
    index = InvertedIndex.build([["a", "b"], ["a"], ["b"], ["irrelevant"]] + [[]] * 1000)
    accesses = []

    class IndexedNorms(list):
        def __iter__(self):
            raise AssertionError("Query must not scan target norms")

        def __getitem__(self, position):
            accesses.append(position)
            return super().__getitem__(position)

    index.norms = IndexedNorms(index.norms)
    stats = RetrievalStats()
    assert len(index.query(["b", "a", "unknown", "a"], 1, stats)) == 1
    assert stats.postings_visited == 4
    assert stats.distinct_candidates == 3  # The heap has only ONE retained entry.
    assert sorted(accesses) == [0, 1, 2]
    index.query(["unknown"], 1, stats)
    assert (stats.postings_visited, stats.distinct_candidates) == (4, 3)
    index.query(["b"], 1, stats)
    assert (stats.postings_visited, stats.distinct_candidates) == (6, 5)


@pytest.mark.parametrize("title", ["", "a", "ab", "abc"])
def test_short_titles_use_existing_normalization(title):
    query = prepared(title)
    targets = [prepared(title, "buy", "target")]
    words, grams = build_indexes(targets)
    for index, terms in ((words, query.title_word_tokens), (grams, query.title_trigrams)):
        result = index.query(terms, 10)
        assert [position for position, _ in result] == ([0] if title else [])
        if result:
            assert result[0][1] == pytest.approx(1.0)


def test_trigrams_recover_match_with_no_shared_words_and_preserve_ranking():
    query = prepared("alpha")
    targets = [prepared("alphabeta", "buy", "0"), prepared("alpha", "buy", "1")]
    words, grams = build_indexes(targets)
    word_candidates = retrieve_candidates(query, words, grams, RetrievalConfig("word", word_k=10))
    gram_candidates = retrieve_candidates(query, words, grams, RetrievalConfig("char", gram_k=10))
    assert [c.target_position for c in word_candidates] == [1]
    assert [c.target_position for c in gram_candidates] == [1, 0]
    assert gram_candidates[0].gram_score == pytest.approx(1.0)
    assert all(c.channels == ("char",) and c.word_score is None for c in gram_candidates)
    assert not query.title_word_tokens & targets[0].title_word_tokens
    assert query.title_trigrams & targets[0].title_trigrams


@pytest.mark.parametrize("word_k,gram_k", [(1, 1), (2, 2), (1, 3), (10, 10)])
def test_union_scores_sources_deduplication_order_and_bound(word_k, gram_k):
    # Shared position 2; top-1 sets are disjoint at positions 0 and 1.
    words = InvertedIndex.build([["red"], [], ["red", "extra"]])
    grams = InvertedIndex.build([[], ["red"], ["red", "xyz"]])
    query = prepared("red")
    word_scores = dict(words.query(query.title_word_tokens, word_k))
    gram_scores = dict(grams.query(query.title_trigrams, gram_k))
    candidates = retrieve_candidates(query, words, grams, RetrievalConfig("union", word_k, gram_k))
    positions = [c.target_position for c in candidates]
    assert positions == sorted(word_scores.keys() | gram_scores.keys())
    assert len(positions) == len(set(positions)) <= word_k + gram_k
    if word_k == gram_k == 1:
        assert positions == [0, 1]  # No second top-1 truncation.
    for candidate in candidates:
        position = candidate.target_position
        assert candidate.word_score == word_scores.get(position)
        assert candidate.gram_score == gram_scores.get(position)
        expected_channels = []
        if position in word_scores:
            expected_channels.append("word")
        if position in gram_scores:
            expected_channels.append("char")
        assert candidate.channels == tuple(expected_channels)


def test_union_empty_and_word_ranking():
    words = InvertedIndex.build([["red", "extra"], ["red"]])
    grams = InvertedIndex.build([[], []])
    candidates = retrieve_candidates(prepared("red"), words, grams, RetrievalConfig("word", word_k=2))
    assert [c.target_position for c in candidates] == [1, 0]
    assert all(c.channels == ("word",) and c.gram_score is None for c in candidates)
    assert retrieve_candidates(prepared("unknown"), words, grams, RetrievalConfig("union", 2, 2)) == []


def test_results_identical_across_python_hash_seeds():
    program = """
import json
from catalogmatch.data import RawRecord, prepare_record
from catalogmatch.retrieval import build_indexes, RETRIEVAL_CONFIGS, retrieve_candidates
from dataclasses import asdict
titles = ['alpha beta 100', 'beta alpha 100', 'alphabet 101', '', 'camera beta']
targets = [prepare_record(RawRecord('buy', str(i), t, '', '', '')) for i, t in enumerate(titles)]
query = prepare_record(RawRecord('abt', 'q', 'alpha beta 100 unseen', '', '', ''))
words, grams = build_indexes(targets)
print(json.dumps({'norms': [words.norms, grams.norms], 'candidates': [
    [asdict(c) for c in retrieve_candidates(query, words, grams, config)]
    for config in RETRIEVAL_CONFIGS]}, sort_keys=True))
"""
    outputs = []
    for seed in ("0", "1", "83"):
        result = subprocess.run([sys.executable, "-c", program], check=True, capture_output=True,
                                text=True, env=dict(os.environ, PYTHONHASHSEED=seed))
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1] == outputs[2]


def selection_rows():
    return [{"name": config.name, "candidate_recall": 0.5, "candidate_pairs": 100}
            for config in RETRIEVAL_CONFIGS]


def test_selection_target_boundary_fewest_actual_pairs_and_fixed_order():
    rows = selection_rows()
    rows[0].update(candidate_recall=0.979999, candidate_pairs=1)
    rows[1].update(candidate_recall=1.0, candidate_pairs=90)
    rows[3].update(candidate_recall=0.98, candidate_pairs=80)
    rows[6].update(candidate_recall=1.0, candidate_pairs=80)
    assert select_configuration(list(reversed(rows))).name == "char10"


def test_selection_fallback_recall_then_cost_then_fixed_order():
    rows = selection_rows()
    rows[0].update(candidate_pairs=1)
    rows[1].update(candidate_recall=0.97, candidate_pairs=95)
    rows[4].update(candidate_recall=0.97, candidate_pairs=90)
    rows[6].update(candidate_recall=0.97, candidate_pairs=90)
    assert select_configuration(list(reversed(rows))).name == "char25"
    with pytest.raises(ValueError, match="exactly the seven"):
        select_configuration(rows[:-1])


def test_evaluation_metrics_gold_used_only_after_retrieval(monkeypatch):
    left = [prepared("alpha", raw_id="left1"), prepared("zzzz", raw_id="left2")]
    right = [prepared("alpha", "buy", "right1"), prepared("alphabeta", "buy", "right2")]
    seen = []
    original = retrieval.retrieve_candidates

    def capture(record, words, grams, config, stats=None):
        candidates = original(record, words, grams, config, stats)
        seen.append((config.name, record.raw.raw_id, candidates))
        return candidates

    monkeypatch.setattr(retrieval, "retrieve_candidates", capture)
    report = evaluate_retrieval(left, right, {(0, 1), (1, 0)})
    assert [row["name"] for row in report["comparison"]] == [c.name for c in RETRIEVAL_CONFIGS]
    assert report["cartesian_pairs"] == 4
    word = report["comparison"][0]
    assert word["candidate_pairs"] == 1
    assert word["candidate_recall"] == 0.0
    assert word["reduction_ratio"] == 0.75
    assert word["average_candidates"] == 0.5
    assert word["maximum_candidates"] == 1
    assert word["postings_visited"] == word["distinct_candidates_encountered"] == 1
    union = report["comparison"][5]
    assert union["gold_hits"] == 1
    assert union["candidate_recall"] == 0.5  # The missing gold pair is never injected.
    assert union["candidate_pairs"] == 2
    assert union["reduction_ratio"] == 0.5
    assert union["average_candidates"] == 1.0
    assert union["maximum_candidates"] == 2
    assert union["postings_visited"] == 7  # One word plus three grams in two targets.
    assert union["distinct_candidates_encountered"] == 3  # Word + char, before union.
    assert report["selected_config"]["name"] == "char10"
    assert report["recall_target_met"] is False
    for row in report["comparison"]:
        assert 0 <= row["average_latency_ms"] <= row["maximum_latency_ms"]
        assert row["retrieval_seconds"] >= 0
    first_run = list(seen)
    seen.clear()
    changed_labels = evaluate_retrieval(left, right, {(0, 0)})
    assert seen == first_run  # Different labels change recall, never candidates or scores.
    assert changed_labels["comparison"][0]["candidate_recall"] == 1.0


def test_empty_or_out_of_partition_gold_cannot_select_configuration():
    with pytest.raises(ValueError, match="at least one gold"):
        evaluate_retrieval([], [], set())
    with pytest.raises(ValueError, match="outside"):
        evaluate_retrieval([prepared("a")], [prepared("a", "buy")], {(0, 1)})


@pytest.fixture
def prepared_dataset(tmp_path, monkeypatch):
    """Twenty synthetic entities give nonempty train/validation/test partitions."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for filename, source in (("Abt.csv", "abt"), ("Buy.csv", "buy")):
        with (raw_dir / filename).open("w", encoding="latin-1", newline="") as stream:
            writer = csv.writer(stream)
            header = ["id", "name", "description"]
            if source == "buy":
                header.append("manufacturer")
            writer.writerow(header + ["price"])
            for i in range(20):
                row = [str(i), f"shared camera model{i}", ""]
                if source == "buy":
                    row.append("")
                writer.writerow(row + [""])
    with (raw_dir / "abt_buy_perfectMapping.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["idAbt", "idBuy"])
        writer.writerows((str(i), str(i)) for i in range(20))
    dataset = load_dataset(raw_dir)
    components = connected_components(build_positive_graph(dataset.records, dataset.gold_pairs))
    partitions = split_components(components)
    manifest_path = tmp_path / "manifest.json"
    cli.write_json(manifest_path, make_manifest(dataset, components, partitions))
    report = audit_dataset(dataset, original=False)
    report.update(validate_partitions(dataset.records, dataset.gold_pairs, components, partitions))
    report["manifest_sha256"] = sha256_file(manifest_path)
    # Only official-size enforcement is replaced; real file hashes, component
    # splits, preparation checks and the complete retrieval CLI still execute.
    monkeypatch.setattr(cli, "verify_original", lambda raw, manifest: (dataset, partitions, dict(report)))
    processed_dir = tmp_path / "processed"
    cli.prepare_data(raw_dir, manifest_path, processed_dir)
    return dataset, partitions, report, raw_dir, manifest_path, processed_dir


def test_partition_loading_and_indexes_never_mix_partitions(prepared_dataset):
    dataset, partitions, audit, _, _, processed_dir = prepared_dataset
    loaded = {}
    for split in ("train", "validation"):
        left, right = load_prepared_partition(processed_dir, dataset, partitions, audit, split)
        loaded[split] = (left, right)
        assert [r.raw for r in left] == list(partition_records(dataset, partitions, split, "abt"))
        assert [r.raw for r in right] == list(partition_records(dataset, partitions, split, "buy"))
        words, grams = build_indexes(right)
        assert words.n_records == grams.n_records == len(right)
        assert set(words.postings) == set().union(*(r.title_word_tokens for r in right))
        for other in ("train", "validation", "test"):
            if other == split:
                continue
            for record in partition_records(dataset, partitions, other, "buy"):
                assert f"model{record.raw_id}" not in words.postings
        for record in left:
            candidates = retrieve_candidates(record, words, grams, RetrievalConfig("all", 100, 100))
            assert {right[c.target_position].raw.key for c in candidates} <= set(partitions[split])
    assert {r.raw.key for r in loaded["train"][1]}.isdisjoint(r.raw.key for r in loaded["validation"][1])
    with pytest.raises(DataValidationError, match="only train and validation"):
        load_prepared_partition(processed_dir, dataset, partitions, audit, "test")


def test_cli_validation_output_and_provenance(prepared_dataset, tmp_path, capsys, monkeypatch):
    dataset, partitions, _, raw_dir, manifest_path, processed_dir = prepared_dataset
    original_build = retrieval.build_indexes
    built = []

    def checked_build(targets):
        built.append([r.raw for r in targets])
        return original_build(targets)

    monkeypatch.setattr(retrieval, "build_indexes", checked_build)
    output = tmp_path / "results"
    result = cli.main(["retrieval-eval", "--data", str(processed_dir), "--raw-dir", str(raw_dir),
                       "--manifest", str(manifest_path), "--split", "val", "--out", str(output)])
    assert result == 0
    assert built == [list(partition_records(dataset, partitions, "validation", "buy"))]
    report = json.loads((output / "selected.json").read_text())
    assert report["split"] == "validation"
    assert report["query_records"] == report["target_records"] == 4
    assert [r["name"] for r in report["comparison"]] == [c.name for c in RETRIEVAL_CONFIGS]
    assert all(row["gold_pairs"] == 4 for row in report["comparison"])
    assert report["raw_sha256"] == dataset.raw_sha256
    assert report["manifest_sha256"] == sha256_file(manifest_path)
    assert report["normalizer_version"] == NORMALIZER_VERSION
    for filename, digest in report["prepared_sha256"].items():
        assert digest == sha256_file(processed_dir / filename)
    assert report["selected_config"]["name"] == "word10"
    text = capsys.readouterr().out
    assert "Selected: word10" in text and "Postings" in text


@pytest.mark.parametrize("split", ["train", "test"])
def test_cli_rejects_non_validation_evaluation(split):
    with pytest.raises(SystemExit) as error:
        cli.main(["retrieval-eval", "--split", split])
    assert error.value.code == 2


@pytest.mark.parametrize("change", ["normalizer", "raw_hash", "manifest_identity", "records_hash", "memberships_hash"])
def test_prepared_provenance_rejects_stale_inputs(prepared_dataset, change):
    dataset, partitions, audit, _, _, directory = prepared_dataset
    report = json.loads((directory / "audit.json").read_text())
    if change == "normalizer":
        report["normalizer_version"] = "old"
    elif change == "raw_hash":
        report["raw_sha256"]["Buy.csv"] = "wrong"
    elif change == "manifest_identity":
        report["manifest_sha256"] = "wrong"
    else:
        filename = "records.jsonl" if change == "records_hash" else "split_memberships.json"
        report["output_sha256"][filename] = "wrong"
    cli.write_json(directory / "audit.json", report)
    with pytest.raises(DataValidationError):
        load_prepared_partition(directory, dataset, partitions, audit, "validation")


@pytest.mark.parametrize("change", ["duplicate", "missing", "split", "position", "raw", "attributes"])
def test_prepared_rows_checked_even_if_output_hash_updated(prepared_dataset, change):
    dataset, partitions, audit, _, _, directory = prepared_dataset
    path = directory / "records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in rows if row["metadata"]["split"] == "validation")
    if change == "duplicate":
        rows.append(row)
    elif change == "missing":
        rows.remove(row)
    elif change == "split":
        row["metadata"]["split"] = "test"
    elif change == "position":
        row["metadata"]["position"] = 99
    elif change == "raw":
        row["raw"]["name"] = "changed"
    else:
        row["attributes"]["title_word_tokens"] = ["injected"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = json.loads((directory / "audit.json").read_text())
    report["output_sha256"]["records.jsonl"] = sha256_file(path)
    cli.write_json(directory / "audit.json", report)
    with pytest.raises(DataValidationError):
        load_prepared_partition(directory, dataset, partitions, audit, "validation")


def test_cli_retrieval_cannot_overwrite_prepared_inputs(tmp_path, capsys):
    assert cli.main(["retrieval-eval", "--data", str(tmp_path), "--out", str(tmp_path)]) == 1
    assert "separate from prepared inputs" in capsys.readouterr().err


@pytest.mark.real_data
def test_real_validation_retrieval_matches_exhaustive_oracle(tmp_path):
    root = Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    if not (raw / "Abt.csv").exists():
        pytest.skip("Official data unavailable; synthetic retrieval coverage still runs")
    manifest = root / "CatalogMatch_Split_Manifest.json"
    processed = tmp_path / "processed"
    cli.prepare_data(raw, manifest, processed)
    dataset, partitions, audit = cli.verify_original(raw, manifest)
    left, right = load_prepared_partition(processed, dataset, partitions, audit, "validation")
    words, grams = build_indexes(right)
    for attribute, index in (("title_word_tokens", words), ("title_trigrams", grams)):
        documents = [getattr(record, attribute) for record in right]
        for record in left:
            query = getattr(record, attribute)
            assert_scores_equal(index.query(query, 50), exhaustive_cosine(documents, query, 50))
    report = cli.run_retrieval_eval(processed, raw, manifest, tmp_path / "results")
    assert report["query_records"] == 217 and report["target_records"] == 218
    assert all(row["gold_pairs"] == 220 for row in report["comparison"])
    assert len(report["comparison"]) == 7
    assert report["selected_config"]["name"] == select_configuration(report["comparison"]).name
