"""Profile the frozen train/validation workload using standard Python tools.

Run from the repository root with the installed project. This adds observations,
not model selection; all caches and profiles go into a new output directory.
"""

import argparse
import cProfile
from dataclasses import asdict
import io
from pathlib import Path
import pstats
import sys
from time import perf_counter
import tracemalloc

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnForest

from catalogmatch.data import sha256_file, verify_original
from catalogmatch.evaluate import load_frozen_run, machine_info, timing_summary
from catalogmatch.features import feature_matrix
from catalogmatch.forest import RandomForestClassifier
from catalogmatch.pipeline import (
    build_catalog_indexes, candidate_pairs, input_provenance,
    load_selected_config, prepare_partition, write_json,
)
from catalogmatch.retrieval import RetrievalStats, retrieve_candidates
from catalogmatch.tree import DecisionTreeClassifier


def timed(operation, passes=5):
    operation()  # Warm imports/allocator; feature calls still recompute every row.
    samples = []
    for _ in range(passes):
        started = perf_counter()
        operation()
        samples.append(perf_counter() - started)
    return {**timing_summary(samples), "warmups": 1, "passes": passes}


def profiled(operation, path):
    profiler = cProfile.Profile()
    profiler.runcall(operation)
    profiler.dump_stats(str(path.with_suffix(".prof")))
    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative")
    stats.print_stats(30)
    path.with_suffix(".txt").write_text(output.getvalue())
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in stats.stats.items():
        rows.append({"file": filename, "line": line, "function": function,
                     "primitive_calls": primitive, "calls": calls,
                     "self_seconds": own, "cumulative_seconds": cumulative})
    return {"total_seconds": stats.total_tt,
            "functions": sorted(rows, key=lambda row: -row["cumulative_seconds"])}


def allocations(operation):
    tracemalloc.start()
    try:
        result = operation()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del result
    return {"current_bytes": current, "peak_bytes": peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/main"))
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("CatalogMatch_Split_Manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # New directory prevents modifying any existing measurements or frozen cache.
    if args.out.exists():
        parser.error("--out must be a new directory")
    for source in (args.run, args.data, args.raw_dir, args.manifest):
        if args.out.resolve().is_relative_to(source.resolve()):
            parser.error("--out must be separate from input artifacts")
    selection, models = load_frozen_run(args.run)
    dataset, partitions, audit = verify_original(args.raw_dir, args.manifest)
    config = load_selected_config(args.run / "retrieval.json", args.data, dataset, audit)
    provenance = input_provenance(args.data, dataset, audit, args.run / "retrieval.json")
    if provenance != selection["provenance"]:
        raise ValueError("Inputs differ from frozen selection")
    args.out.mkdir(parents=True)
    report = {"command": [sys.executable, *sys.argv], "environment": machine_info(),
              "provenance": provenance, "retrieval_config": asdict(config),
              "source_sha256": {str(p): sha256_file(p) for p in sorted(Path("src/catalogmatch").glob("*.py"))},
              "method": "one warmup + five uninstrumented passes; separate cProfile and tracemalloc passes; no test workload",
              "partitions": {}, "profiles": {}, "timings": {}, "allocations": {}}
    data_by_split = {}
    for split in ("train", "validation"):
        print(f"Measuring {split} features and retrieval", flush=True)
        data = prepare_partition(args.data, dataset, partitions, audit, split, config,
                                 args.out / "features", provenance)
        data_by_split[split] = data
        report["partitions"][split] = data.report
        indexes = build_catalog_indexes(data.right, config)
        extract = lambda: feature_matrix(data.pairs, data.left, data.right)
        retrieve = lambda: candidate_pairs(data.left, indexes, config)
        np.testing.assert_array_equal(extract(), data.X)
        with np.load(args.run / "features" / f"{split}.npz", allow_pickle=False) as frozen:
            np.testing.assert_array_equal(data.pairs, frozen["pairs"])
            np.testing.assert_array_equal(data.X, frozen["X"])
        for name, operation in ((f"{split}_features", extract), (f"{split}_retrieval", retrieve)):
            report["timings"][name] = timed(operation)
            report["profiles"][name] = profiled(operation, args.out / name)
        report["timings"][f"{split}_index"] = timed(lambda: build_catalog_indexes(data.right, config))
        counters = RetrievalStats()
        for record in data.left:
            retrieve_candidates(record, *indexes, config, counters)
        report["partitions"][split]["retrieval_counters"] = asdict(counters)
        if split == "validation":
            report["allocations"]["validation_features"] = allocations(extract)
            report["allocations"]["validation_retrieval"] = allocations(retrieve)
    train, validation = data_by_split["train"], data_by_split["validation"]
    for name, constructor in (("custom_cart", DecisionTreeClassifier),
                              ("custom_forest", RandomForestClassifier),
                              ("sklearn_forest", SklearnForest)):
        print(f"Measuring {name} fit and validation scoring", flush=True)
        parameters = selection["selected"][name]["hyperparameters"]
        fit = lambda: constructor(**parameters).fit(train.X, train.y)
        model = models[name]
        score = lambda: model.score(validation.X) if name != "sklearn_forest" else model.predict_proba(validation.X)[:, 1]
        fitted = fit()
        np.testing.assert_array_equal(fitted.predict_proba(validation.X)[:, 1], score())
        if name != "sklearn_forest":
            assert fitted.to_dict() == model.model.to_dict()
        report["timings"][f"{name}_fit"] = timed(fit)
        report["timings"][f"{name}_validation_score"] = timed(score)
        report["profiles"][f"{name}_fit"] = profiled(fit, args.out / f"{name}_fit")
        report["profiles"][f"{name}_score"] = profiled(score, args.out / f"{name}_score")
        report["allocations"][f"{name}_fit"] = allocations(fit)
        report["allocations"][f"{name}_score"] = allocations(score)
    report["equivalence"] = "Exact frozen train/validation candidates/features, refit validation scores, and custom tree structures"
    report["allocation_scope"] = "Separate operation after inputs loaded; current includes returned object; tracemalloc is not total process RSS"
    write_json(args.out / "profile.json", report)
    print(f"Saved {args.out / 'profile.json'}", flush=True)


if __name__ == "__main__":
    main()
