"""Prepare verified data, retrieve candidates, and inspect/cache numerical features."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

from catalogmatch.data import (
    DataValidationError, make_manifest, connected_components, build_positive_graph,
    load_prepared_partition, partition_records, prepared_attributes, prepare_record,
    sha256_file, SPLIT_VERSION, verify_original, load_records, RawRecord,
)
from catalogmatch.normalize import NORMALIZER_VERSION
from catalogmatch.features import FEATURE_NAMES, cached_feature_matrix, pair_features
from catalogmatch.retrieval import (
    build_indexes, evaluate_retrieval, retrieve_candidates, select_configuration,
    RetrievalConfig, RETRIEVAL_CONFIGS, RETRIEVAL_VERSION,
)
from catalogmatch.similarity import training_baseline
from catalogmatch.pipeline import load_selected_config, match_record


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_data(raw_dir: Path, manifest_path: Path, output_dir: Path) -> dict:
    dataset, partitions, report = verify_original(raw_dir, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    components = connected_components(build_positive_graph(dataset.records, dataset.gold_pairs))
    # Write a checked copy, never modify the user's supplied root manifest.
    write_json(output_dir / "split_memberships.json", make_manifest(dataset, components, partitions))
    record_count = 0
    with (output_dir / "records.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for split in partitions:
            for source in ("abt", "buy"):
                for position, record in enumerate(partition_records(dataset, partitions, split, source)):
                    prepared = prepare_record(record)
                    row = {"metadata": {"split": split, "position": position},
                           "raw": asdict(record), "attributes": prepared_attributes(prepared)}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    record_count += 1
    report["prepared_records"] = record_count
    report["normalizer_version"] = NORMALIZER_VERSION
    report["output_sha256"] = {
        "records.jsonl": sha256_file(output_dir / "records.jsonl"),
        "split_memberships.json": sha256_file(output_dir / "split_memberships.json"),
    }
    write_json(output_dir / "audit.json", report)
    return report


def run_baseline(raw_dir: Path, manifest_path: Path, output_dir: Path,
                 threshold: float) -> dict:
    dataset, partitions, audit = verify_original(raw_dir, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Evaluation consumes training labels only, after scoring product attributes.
    train_abt = {vertex[1] for vertex in partitions["train"] if vertex[0] == "abt"}
    train_gold = {(left, right) for left, right in dataset.gold_pairs if left in train_abt}
    prediction_count = 0
    true_positive_count = 0
    matches_path = output_dir / "matches.csv"
    with matches_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["idAbt", "idBuy", "lexical_score"])
        for left, right, score in training_baseline(dataset, partitions, threshold):
            writer.writerow([left, right, score])
            prediction_count += 1
            if (left, right) in train_gold:
                true_positive_count += 1
    sizes = audit["partitions"]["train"]
    report = {
        "split": "train", "threshold": threshold, "threshold_tuned": False,
        "comparisons": sizes["abt"] * sizes["buy"],
        "predicted_pairs": prediction_count, "gold_pairs": len(train_gold),
        "true_positives": true_positive_count,
        "false_positives": prediction_count - true_positive_count,
        "false_negatives": len(train_gold) - true_positive_count,
        "precision": true_positive_count / prediction_count if prediction_count else 0.0,
        "recall": true_positive_count / len(train_gold) if train_gold else 0.0,
        "raw_sha256": dataset.raw_sha256,
        "manifest_sha256": audit["manifest_sha256"],
        "matches_sha256": sha256_file(matches_path),
    }
    write_json(output_dir / "baseline.json", report)
    return report


def run_retrieval_eval(processed_dir: Path, raw_dir: Path, manifest_path: Path,
                       output_dir: Path) -> dict:
    dataset, partitions, audit = verify_original(raw_dir, manifest_path)
    left, right = load_prepared_partition(processed_dir, dataset, partitions, audit, "validation")
    left_positions = {record.raw.raw_id: position for position, record in enumerate(left)}
    right_positions = {record.raw.raw_id: position for position, record in enumerate(right)}
    # Labels enter evaluation only. Retrieval sees product terms and local offsets.
    gold_positions = set()
    for left_id, right_id in dataset.gold_pairs:
        if left_id in left_positions and right_id in right_positions:
            gold_positions.add((left_positions[left_id], right_positions[right_id]))
    report = evaluate_retrieval(left, right, gold_positions)
    report.update({
        "split": "validation", "query_source": "abt", "target_source": "buy",
        "raw_sha256": dataset.raw_sha256, "manifest_sha256": audit["manifest_sha256"],
        "split_version": SPLIT_VERSION, "split_seed": 42,
        "normalizer_version": NORMALIZER_VERSION, "retrieval_version": RETRIEVAL_VERSION,
        "prepared_sha256": {
            filename: sha256_file(processed_dir / filename)
            for filename in ("records.jsonl", "split_memberships.json", "audit.json")
        },
        "parameters": {
            "term_frequency": "binary", "idf": "1 + ln((N + 1) / (df + 1))",
            "unknown_query_df": 0, "query_term_order": "sorted distinct terms",
            "eligibility": "positive term overlap", "tie_break": "smaller target position",
            "word_terms": "normalized title words", "char_terms": "normalized title trigrams",
            "union": "independent top-k lists; deduplicate; position order; no second truncation",
            "configuration_order": [row["name"] for row in report["comparison"]],
        },
        "measurement": {
            "timer": "perf_counter", "passes_per_configuration": 1,
            "latency_scope": "retrieval calls including counters, heap and union; excludes index build and evaluation",
            "distinct_candidates_scope": "sum per query and channel, before top-k; repeats across channels count twice",
            "python": platform.python_version(), "platform": platform.platform(),
        },
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "selected.json", report)
    return report


def print_retrieval_table(report: dict) -> None:
    print("Configuration  Recall   Pairs  Reduction  Avg/query  Max   Time(ms)  Postings")
    for row in report["comparison"]:
        print(f"{row['name']:<13} {row['candidate_recall']:7.4f} {row['candidate_pairs']:7d} "
              f"{row['reduction_ratio']:9.4f} {row['average_candidates']:10.2f} "
              f"{row['maximum_candidates']:4d} {1000 * row['retrieval_seconds']:10.2f} "
              f"{row['postings_visited']:9d}")
    status = "met" if report["recall_target_met"] else "not met; highest-recall fallback"
    print(f"Selected: {report['selected_config']['name']} (recall target {status})")



def run_features(processed_dir: Path, raw_dir: Path, manifest_path: Path,
                 selected_path: Path, output_dir: Path, split: str = "both") -> dict:
    if split not in ("train", "validation", "both"):
        raise DataValidationError("Features support only train and validation partitions")
    dataset, partitions, audit = verify_original(raw_dir, manifest_path)
    config = load_selected_config(selected_path, processed_dir, dataset, audit)
    splits = ("train", "validation") if split == "both" else (split,)
    reports = []
    for name in splits:
        left, right = load_prepared_partition(processed_dir, dataset, partitions, audit, name)
        started = perf_counter()
        word_index, gram_index = build_indexes(right)
        pairs = []
        for position, record in enumerate(left):
            for candidate in retrieve_candidates(record, word_index, gram_index, config):
                pairs.append((position, candidate.target_position))
        retrieval_seconds = perf_counter() - started
        provenance = {
            "raw_sha256": dataset.raw_sha256, "manifest_sha256": audit["manifest_sha256"],
            "split": name, "split_version": SPLIT_VERSION, "split_seed": 42,
            "retrieval_config": asdict(config), "retrieval_version": RETRIEVAL_VERSION,
            "selected_sha256": sha256_file(selected_path),
        }
        cache_path = output_dir / f"{name}.npz"
        matrix, timing = cached_feature_matrix(cache_path, pairs, left, right, provenance=provenance)
        reports.append({"split": name, "candidate_pairs": len(pairs),
                        "shape": list(matrix.shape), "dtype": str(matrix.dtype),
                        "retrieval_seconds": retrieval_seconds, "cache_path": str(cache_path), **timing})
    report = {"retrieval_config": asdict(config), "feature_names": list(FEATURE_NAMES),
              "partitions": reports,
              "measurement": "perf_counter; feature_seconds excludes loading, retrieval, and cache I/O"}
    write_json(output_dir / "report.json", report)
    return report


def inspect_features(processed_dir: Path, raw_dir: Path, manifest_path: Path,
                     split: str, abt_id: str, buy_id: str) -> dict:
    dataset, partitions, audit = verify_original(raw_dir, manifest_path)
    left, right = load_prepared_partition(processed_dir, dataset, partitions, audit, split)
    selected = []
    for records, raw_id, source in ((left, abt_id, "abt"), (right, buy_id, "buy")):
        record = next((record for record in records if record.raw.raw_id == raw_id), None)
        if record is None:
            raise DataValidationError(f"Unknown {source} ID {raw_id!r} in {split}")
        selected.append(record)
    values = pair_features(selected[0], selected[1])
    return {"split": split, "abt": asdict(selected[0].raw), "buy": asdict(selected[1].raw),
            "features": dict(zip(FEATURE_NAMES, values))}


def threshold_value(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("threshold must be finite and between 0 and 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare", help="audit, verify the manifest, and prepare every record")
    baseline = subcommands.add_parser("baseline", help="run exhaustive lexical scoring on training records only")
    retrieval = subcommands.add_parser("retrieval-eval", help="compare seven retrieval configurations on validation only")
    features = subcommands.add_parser("features", help="cache selected training/validation candidate features")
    inspect = subcommands.add_parser("inspect-features", help="display raw attributes and twelve pair features")
    tune = subcommands.add_parser("tune", help="select models and thresholds using validation, then freeze")
    evaluate = subcommands.add_parser("evaluate", help="evaluate frozen models on the held-out test partition")
    benchmark = subcommands.add_parser("benchmark", help="measure indexed/exhaustive queries on training products")
    match = subcommands.add_parser("match", help="match a product using a custom forest; no gold or sklearn")
    for command in (prepare, baseline, retrieval, features, inspect, tune, evaluate, benchmark):
        command.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
        command.add_argument("--manifest", type=Path, default=Path("CatalogMatch_Split_Manifest.json"))
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    baseline.add_argument("--output-dir", type=Path, default=Path("runs/train_baseline"))
    baseline.add_argument("--threshold", type=threshold_value, default=0.5)
    retrieval.add_argument("--data", type=Path, default=Path("data/processed"))
    retrieval.add_argument("--split", choices=("val", "validation"), default="val")
    retrieval.add_argument("--out", dest="output_dir", type=Path, default=Path("runs/retrieval"))
    for command in (features, inspect):
        command.add_argument("--data", type=Path, default=Path("data/processed"))
    features.add_argument("--split", choices=("train", "validation", "both"), default="both")
    features.add_argument("--selected", type=Path, default=Path("runs/retrieval/selected.json"))
    features.add_argument("--out", dest="output_dir", type=Path, default=Path("runs/features"))
    inspect.add_argument("--split", choices=("train", "validation"), default="train")
    inspect.add_argument("--abt-id", required=True)
    inspect.add_argument("--buy-id", required=True)
    for command in (tune, evaluate, benchmark):
        command.add_argument("--data", type=Path, default=Path("data/processed"))
    tune.add_argument("--retrieval", type=Path, default=Path("runs/retrieval/selected.json"))
    tune.add_argument("--out", type=Path, default=Path("runs/main"))
    for command in (evaluate, benchmark):
        command.add_argument("--run", type=Path, default=Path("runs/main"))
    evaluate.add_argument("--split", choices=("test",), default="test")
    benchmark.add_argument("--queries", type=int, default=20)
    match.add_argument("--bundle", type=Path, required=True)
    match.add_argument("--right", type=Path, required=True)
    match.add_argument("--name", required=True)
    match.add_argument("--description", default="")
    match.add_argument("--price", default="")
    args = parser.parse_args(argv)
    try:
        if args.command == "match":
            query = RawRecord("abt", "query", args.name, args.description, "", args.price)
            print(json.dumps(match_record(query, load_records(args.right, "buy"), args.bundle), indent=2))
            return 0
        if args.command in ("tune", "evaluate", "benchmark"):
            from catalogmatch.evaluate import tune_run, evaluate_run, benchmark_run
            output = (args.out if args.command == "tune" else args.run).resolve()
            for source in (args.raw_dir.resolve(), args.data.resolve()):
                if output.is_relative_to(source) or source.is_relative_to(output):
                    raise DataValidationError("Run output must be separate from data inputs")
            if args.manifest.resolve().is_relative_to(output):
                raise DataValidationError("Run output must be separate from the supplied manifest")
            progress = lambda message: print(message, file=sys.stderr, flush=True)
            if args.command == "tune":
                if args.retrieval.resolve().is_relative_to(output):
                    raise DataValidationError("Run output must be separate from selected retrieval input")
                report = tune_run(args.data, args.raw_dir, args.manifest, args.retrieval, args.out, progress=progress)
            elif args.command == "evaluate":
                report = evaluate_run(args.data, args.raw_dir, args.manifest, args.run, args.split)
            else:
                report = benchmark_run(args.data, args.raw_dir, args.manifest, args.run, args.queries, progress=progress)
            print(json.dumps(report, indent=2))
            return 0
        if args.command == "inspect-features":
            report = inspect_features(args.data, args.raw_dir, args.manifest,
                                      args.split, args.abt_id, args.buy_id)
            print(json.dumps(report, indent=2))
            return 0
        # Output directories must not overwrite inputs, including the supplied manifest.
        output = args.output_dir.resolve()
        raw = args.raw_dir.resolve()
        manifest = args.manifest.resolve()
        if output == raw or output.is_relative_to(raw) or manifest.is_relative_to(output):
            raise DataValidationError("Output directory must be separate from raw inputs and the supplied manifest")
        if args.command in ("retrieval-eval", "features"):
            processed = args.data.resolve()
            if output.is_relative_to(processed) or processed.is_relative_to(output):
                raise DataValidationError("Retrieval output directory must be separate from prepared inputs")
        if args.command == "features" and args.selected.resolve().is_relative_to(output):
            raise DataValidationError("Feature output directory must be separate from the selected configuration")
        if args.command == "prepare":
            report = prepare_data(args.raw_dir, args.manifest, args.output_dir)
        elif args.command == "baseline":
            report = run_baseline(args.raw_dir, args.manifest, args.output_dir, args.threshold)
        elif args.command == "features":
            report = run_features(args.data, args.raw_dir, args.manifest,
                                  args.selected, args.output_dir, args.split)
        else:
            report = run_retrieval_eval(args.data, args.raw_dir, args.manifest, args.output_dir)
    except (OSError, ValueError) as error:
        print(f"catalogmatch: {error}", file=sys.stderr)
        return 1
    if args.command == "retrieval-eval":
        print_retrieval_table(report)
        print(f"Saved: {args.output_dir / 'selected.json'}")
    else:
        print(json.dumps(report, indent=2))
    return 0
