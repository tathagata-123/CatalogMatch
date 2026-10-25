# Saved experimental evidence

These reports substantiate the documented measurements without bundling raw data,
model bundles or feature caches. [files.json](files.json) records each public file's
SHA-256 and original local path. Original experiments remain under ignored `runs/`.
A clean checkout can inspect these reports and reproduce a new run using the
[README instructions](../../README.md#reproduce-from-a-clean-checkout).

## Evaluation reports

The `evaluation/` files are byte-identical copies of the original reports:

- `selection.json`: all thirteen validation trials, selected settings, shared-matrix
  identities, timings and environment.
- `retrieval.json`: seven-configuration validation retrieval selection.
- `test_metrics.json`: first frozen test result.
- `frozen.json`: original selection-artifact checksums. Referenced models and score
  arrays are local outputs, not files bundled in this evidence directory.
- `benchmark.json`: original uncached indexed/exhaustive comparison and memory.
- `product_trace.json`, `retrieval_miss_analysis.json`, `hardware.json` and
  `training_baseline.json`: explanatory product trace, post-evaluation ranks,
  machine identification and preliminary training-only lexical baseline.

## Performance reports

- `performance/before/profile.json` and `after/profile.json`: public extracts of
  the profiling reports, retaining every profile row, wall sample, allocation
  observation, source hash and environment version.
- `performance/paired_dp.json`: byte-identical paired feature measurements and
  equality checks on full training/validation candidate inputs.
- `performance/benchmark.json`: byte-identical benchmark replay using the frozen
  bundles with optimized edit-distance computation.
- `performance/verification.json`: a public extract of the recorded equivalence
  checks, including exact candidates, features, scores, decisions and metrics.
- `performance/trace_details.json`: byte-identical postings, IDF, ranking and
  edit-distance details for the training example.
- `performance/pytest-final.txt`: unmodified test output from the optimization check.

## Extracts and historical inputs

The three public extracts explicitly identify their original reports by path and
SHA-256 in `_publication` and in `files.json`. The profile extracts omit only the
`command` field, which contains a local absolute interpreter path. The verification
extract omits `gold_free_cli/cli_arguments` (absolute local paths) and
`changed_preexisting_files` (an editing-session inventory). No measurements,
model settings, source hashes, comparisons or result values were changed. The
original reports remain unmodified locally and are excluded from publication.

Redundant command logs, editing-session checksum inventories, duplicate test
summaries and the one-off workspace verification driver remain local. They are
not prerequisites for the documented reproduction commands.

`performance/compare_dp.py` and `performance/baseline_source/similarity.py` are
byte-identical **historical experiment inputs**, not runtime modules. The driver
retains the original relative `runs/m6/` paths so its recorded provenance is not
rewritten. To repeat that paired experiment, use an isolated workspace with the
original `runs/main/` artifacts and copy the baseline source to the path the driver
expects. Never run it over existing measurements. The reusable profiler is
`python tools/profile_workload.py --run RUN_DIRECTORY --out NEW_DIRECTORY`.
It requires a generated frozen run; report copies alone are not model bundles.

## Replay an available frozen run

To replay an available original freeze without replacing its first reports, copy
only the freeze-listed artifacts to a new directory:

```sh
python - <<'PY'
import json
from pathlib import Path
import shutil
source = Path('runs/main')
target = Path('runs/frozen-replay-new')
target.mkdir()  # Fails if already present.
freeze = json.loads((source / 'frozen.json').read_text())
for name in ['frozen.json', *freeze['files_sha256']]:
    shutil.copyfile(source / name, target / name)
PY
python -m catalogmatch evaluate --data data/processed --run runs/frozen-replay-new --split test
python -m catalogmatch benchmark --data data/processed --run runs/frozen-replay-new --queries 20
```

This loads the same model structures and thresholds. It does not fit or select;
test results are repeated observations of the already evaluated partition. Keep
the original prepared data and retrieval snapshot because their hashes are part
of the freeze. Running retrieval selection again produces different report bytes
because timing fields change, even if the selected configuration is identical.
