# CatalogMatch

CatalogMatch links product listings across catalogs using classical algorithms and
machine learning. A custom inverted index retrieves plausible pairs, twelve
numerical similarity features describe each pair, and a Random Forest implemented
from scratch decides which pairs match. Multiple matches per product are allowed.

The project demonstrates algorithmic engineering: connected-component splitting,
binary TF-IDF cosine accumulation, a bounded binary heap, exact two-row Levenshtein
DP, sorting-and-prefix-count CART splits, and sequential bootstrapped forests.
NumPy supplies arrays, sorting and random sampling; scikit-learn is an evaluation
reference. Custom training and matching do not delegate to scikit-learn.

## Results

First frozen Abt–Buy test evaluation, with entity-disjoint partitions and
validation-selected settings:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Lexical baseline | 0.6797 | 0.7169 | 0.6978 |
| Custom CART | 0.9198 | 0.7854 | 0.8473 |
| Custom Random Forest | 0.8700 | 0.8858 | 0.8778 |
| scikit-learn Random Forest | 0.8733 | 0.8813 | 0.8773 |

Frozen `word10` retrieval finds 217/219 test gold pairs using 2,053 candidates,
a 95.66% reduction from exhaustive comparison. The original measured 20-query
benchmark was 61.84× faster with indexing, using uncached features on both paths.
An exact feature-extraction optimization measured about 1.74× faster in paired
training/validation measurements without changing outputs.

See [experimental results](docs/results.md) for exact evidence, timing scopes,
error accounting and limitations; [design](docs/design.md) for implementation
contracts; and [technical study notes](docs/interview.md) for algorithms and exercises.

## Install

Python **3.11+** is required. From the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The only runtime dependency is NumPy (`pip install -e .` for matching alone).
The `dev` extra adds pytest and scikit-learn, which brings its own dependencies.
For the recorded experiment environment, use **CPython 3.12.15** and:

```sh
python -m pip install -r requirements-reproduce.txt
python -m pip install --no-deps -e .
```

Exact reference-pickle replay checks Python, NumPy and sklearn versions.
Other compatible versions can run a new experiment, but numerical/RNG behavior
and timings may differ. Custom inference loads portable versioned JSON structures.

## Dataset

Download the **original** [Abt–Buy benchmark from Leipzig's Database Group](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution).
Credit that source and its cited VLDB 2010 evaluation paper when reusing the data;
the source page supplies the dataset's license and attribution information.

```sh
mkdir -p data/raw
curl -fL https://dbs.uni-leipzig.de/files/datasets/Abt-Buy.zip -o /tmp/CatalogMatch-Abt-Buy.zip
unzip -j -n /tmp/CatalogMatch-Abt-Buy.zip '*.csv' -d data/raw
```

Alternatively place the unmodified `Abt.csv`, `Buy.csv`, and
`abt_buy_perfectMapping.csv` directly in `data/raw/`. Keep the supplied
`CatalogMatch_Split_Manifest.json` at the repository root. Do not regenerate it or
substitute cleaned CSVs. Preparation checks their SHA-256 hashes, preserves all
2,173 records, and reproduces the seed-42 component split. CSVs are Latin-1;
prices are interpreted as USD, with missing/invalid values kept missing.

## Reproduce from a clean checkout

Run these commands in order. Output paths below deliberately name a new run.
Generated data, models and `runs/` are Git-ignored; they are not prerequisites in
a clean checkout. The manifest and small [saved evidence](docs/evidence/) accompany
the documentation.

```sh
python -m catalogmatch --help
python -m catalogmatch prepare --raw-dir data/raw --manifest CatalogMatch_Split_Manifest.json --output-dir data/processed
python -m catalogmatch retrieval-eval --data data/processed --split val --out runs/retrieval-reproduction
python -m catalogmatch tune --data data/processed --retrieval runs/retrieval-reproduction/selected.json --out runs/reproduction
python -m catalogmatch evaluate --data data/processed --run runs/reproduction --split test
python -m catalogmatch benchmark --data data/processed --run runs/reproduction --queries 20
python -m catalogmatch match --bundle runs/reproduction/custom_forest.json --right data/raw/Buy.csv --name 'Omnimount Wall Speaker Mount - 20WLWH' --price '40.00'
python -m pytest -q
```

`prepare` writes `records.jsonl`, `split_memberships.json` and `audit.json` into
`data/processed/`. `retrieval-eval` compares the seven prescribed configurations
on validation and selects the fewest pairs reaching 98% candidate recall:
`word10` on these inputs. `tune` generates/caches the shared feature matrices,
replays the fixed thirteen-trial protocol on train/validation, selects thresholds,
saves models and writes `frozen.json` last. `evaluate` verifies the freeze before
using test attributes; it never fits or selects. A reproduction is a repeat of an
already observed test set, not a new held-out experiment.

`tune` requires an empty output directory. `evaluate` and `benchmark` refuse to
overwrite their first reports. Use fresh output paths for repeats, including
retrieval selection; its JSON identity includes timing fields, so a new retrieval
report changes provenance hashes even when its candidates are identical. Preserve
original raw/prepared inputs and selection files when replaying `runs/main/`.

The benchmark uses the same queries, catalog, feature functions, model and
threshold for indexed and exhaustive matching. One warmup and five alternating
passes exclude target preparation/model loading and recompute features each time.
It reports these setup costs and memory separately; it does not extrapolate scale.

Optional inspection and profiling commands:

```sh
python -m catalogmatch features --selected runs/retrieval-reproduction/selected.json --out runs/features-reproduction
python -m catalogmatch inspect-features --split train --abt-id 10102 --buy-id 90106210
python tools/profile_workload.py --run runs/reproduction --out runs/profile-reproduction
```

The profiler requires a new output directory and uses training/validation only.
It saves `cProfile` statistics, five-pass wall timings, allocation observations,
and exact checks against saved feature matrices and fixed-parameter refits.
See [equivalence verification](docs/results.md#equivalence-verification) for
the recorded checks. No additional profiling packages are needed.

## Gold-free application

`match` reads only a supplied target CSV and custom forest bundle. Optional
`--description` and `--price` add query attributes. Its JSON result contains every
accepted target's ID, raw title, score and twelve features, sorted by descending
score then ID. No accepted pair gives `[]`. The score is an average of tree leaf
fractions, not a calibrated probability of product identity.

For repeated queries, reuse `pipeline.prepare_catalog` and call
`pipeline.match_record` with that catalog and `pipeline.load_bundle`'s result.
Catalog-dependent IDF and candidate membership can change when the target catalog
changes. Matching requires neither the mapping CSV nor split metadata, sklearn,
or scipy. A fresh-process test forbids the ML-library imports explicitly.

## Repository

| Path | Responsibility |
|---|---|
| `src/catalogmatch/data.py`, `normalize.py` | Original-byte validation, component splits, prepared attributes |
| `retrieval.py`, `similarity.py`, `features.py` under `src/catalogmatch/` | Index, heap, DP, twelve-feature contract and cache |
| `tree.py`, `forest.py` under `src/catalogmatch/` | From-scratch learners and JSON persistence |
| `pipeline.py`, `evaluate.py`, `cli.py` under `src/catalogmatch/` | Gold-free application, frozen experiments and commands |
| `tests/` | Independent cosine, full-sort, full-table DP and split-search oracles; integration checks |
| `tools/profile_workload.py` | Bounded standard-library profiling of existing workloads |
| `docs/` | Design, results, technical study notes and compact experimental evidence |
| `data/raw/`, `data/processed/`, `runs/` | Local inputs and generated artifacts (ignored by Git) |

## Limitations

Abt–Buy is small and old, and every record has a known partner in the other source;
it does not establish performance on modern catalogs with many unmatched products.
Descriptions/prices are often missing. Model-number extraction is heuristic;
closely related color/fuel variants can be confused. Lexical top-K retrieval can
miss gold pairs that the classifier could otherwise accept. Python DP and tree
traversal limit throughput; compiled sklearn scoring is faster here. The near-equal
forest F1 results do not establish general superiority. This project demonstrates
DSA and classical ML integration, without neural language understanding or a
production-scale claim.
