# Experimental results

Quality numbers below come from the first frozen evaluation. The later performance
optimization preserved the model choices and predictions. Tables round for
readability; linked JSON retains every recorded digit and timing sample. All times
are local observations, not throughput guarantees.

## Experimental setup

The original [Abt–Buy benchmark](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution)
was supplied by Leipzig's Database Group and used in its VLDB 2010 entity-resolution
evaluation. It contains 1,081 Abt records, 1,092 Buy records and 1,097 positive edges.
The original Latin-1 CSV bytes are retained. No cleaned derivative or synthetic
product record replaces these inputs.

SHA-256 identities checked by preparation:

| Input | SHA-256 |
|---|---|
| `Abt.csv` | `ed6dbe5ac508e2e27ec7f78948875d4f754b6753674c65ccda3bf903ddaa13ea` |
| `Buy.csv` | `37d78c33f85b27e92be1d8ee1b863ba7107dfe369c6ca1dc7961c89b460dd47a` |
| `abt_buy_perfectMapping.csv` | `20af553a0846712de11703e2b9569988468d5c639650348dd5a059f4bb861b6a` |
| Root split manifest | `38807dded7aa6c47f4fea09ac0c159538a6ed3d6f205d4edc16be44df8e7fa2a` |

Gold-positive connected components define entities. Canonical components are
shuffled with Python `random.Random(42)` and assigned 60/20/20 by component count
(with floor allocation and a remainder). Entire components stay together; every
record appears once and no endpoint crosses partitions. Negative candidates never
build the graph. Product features exclude IDs, gold, component IDs and split metadata.

| Partition | Components | Abt queries | Buy targets | Gold pairs |
|---|---:|---:|---:|---:|
| Train | 645 | 648 | 655 | 658 |
| Validation | 215 | 217 | 218 | 220 |
| Test | 216 | 216 | 219 | 219 |

Recorded environment: CPython 3.12.15, NumPy 2.5.4, scikit-learn 1.9.1,
pytest 9.1.1, macOS 26.5.2 ARM64, Apple M4, ten logical CPUs. See
[hardware evidence](evidence/evaluation/hardware.json) and the complete experiment
[version pins](../requirements-reproduce.txt), including SciPy 1.18.1.
The frozen report records the `arm` architecture fallback; the separate hardware
query identifies the M4. Custom trees train sequentially; sklearn uses `n_jobs=1`.

Retrieval was selected on validation before classifier selection. `word10` uses
binary title-word TF, IDF `1 + ln((N+1)/(df+1))`, cosine scores, positive-overlap
eligibility, and at most ten hits per query. Each partition has its own target
index. Unknown query terms contribute df=0 weights to the query norm.

All learned models see the same 6,426 training rows: 630 positives and 5,796 negatives
(9.8039215686% positive). Missing gold pairs are not injected. No negative sampling,
class weighting, validation refit, pruning or new feature search is performed.
The twelve ordered features and normalizer version are specified in
[the design contract](design.md#ordered-twelve-feature-contract).

The fixed comparison has thirteen trials: lexical, plus depth {8,12} × minimum
leaf {2,5} for each of CART/custom forest/sklearn forest. All use minimum split 2
and seed 42; CART inspects all twelve columns. Both forests use 32 bootstrap trees
and square-root feature sampling. Sklearn uses Gini. Threshold selection sweeps
complete equal-score groups and maximizes validation **end-to-end F1**, then
precision, then higher threshold. Model-configuration ties prefer shallower depth,
then larger leaf size. Selected models remain trained on train only.

The models, validation results and their checksums were written before
[`frozen.json`](evidence/evaluation/frozen.json), dated 2026-10-24 UTC (2026-10-25 locally).
Test evaluation only scores those frozen models and thresholds. A later replay
checks equivalence; it does not make this test set newly unseen.

## Retrieval results

The early exhaustive training lexical baseline at untuned threshold 0.5 made
424,440 comparisons, accepting 535 pairs with 289 TP (precision
0.5401869158878505; recall 0.439209726443769).
This [preliminary training observation](evidence/evaluation/training_baseline.json) is
separate from the validation-selected lexical classifier below. No saved
seven-configuration training retrieval selection is claimed.

The seven-row **validation selection** is preserved in
[`retrieval.json`](evidence/evaluation/retrieval.json). The denominator is 47,306 possible
pairs, with 220 gold pairs; timings are one pass and exclude index construction.

| Configuration | Gold retrieved | Candidates | Recall | Reduction | Postings visited | Retrieval ms |
|---|---:|---:|---:|---:|---:|---:|
| word10 | 217/220 | 2,078 | 0.986364 | 0.956073 | 10,176 | 3.082047 |
| word25 | 218/220 | 4,640 | 0.990909 | 0.901915 | 10,176 | 4.263003 |
| word50 | 219/220 | 6,909 | 0.995455 | 0.853951 | 10,176 | 5.006505 |
| char10 | 220/220 | 2,170 | 1.000000 | 0.954128 | 108,251 | 10.747842 |
| char25 | 220/220 | 5,425 | 1.000000 | 0.885321 | 108,251 | 12.900248 |
| union10+10 | 220/220 | 2,718 | 1.000000 | 0.942544 | 118,427 | 13.956753 |
| union25+25 | 220/220 | 6,693 | 1.000000 | 0.858517 | 118,427 | 17.915328 |

Both validation indexes together took 0.0024191669654101133 s to build.
Words had 911 terms/1,853 term-record associations; trigrams had 3,110/10,058.
`word10` encountered 7,408 distinct targets summed over queries before retaining
2,078 pairs; `char10` encountered 32,061. Postings and encounter counts in union
rows include work in both channels, while final pairs are deduplicated.

**Selection rule:** minimize actual candidates among recall ≥0.98; otherwise choose
highest recall, then lowest cost, with fixed configuration order breaking ties.
`word10` qualified with 2,078 pairs versus `char10`'s 2,170. Trigrams recovered all
validation positives, at greater postings work. This is an explicit cost/recall
trade-off; it is not evidence that word retrieval is universally better.

Once frozen, the selected retrieval produced:

| Partition | Gold retrieved | Candidates | Recall | Reduction | Original pipeline retrieval s |
|---|---:|---:|---:|---:|---:|
| Train (descriptive) | 630/658 | 6,426 | 0.957447 | 0.984860 | 0.038190 |
| Validation (selection) | 217/220 | 2,078 | 0.986364 | 0.956073 | 0.006858 |
| Test (held out) | 217/219 | 2,053 | 0.990868 | 0.956600 | 0.007158 |

The pipeline times include ordered candidate-array construction; they are separate
runs from the seven-row retrieval-only comparison. Original test postings counters
were not recorded. Profiling measured 85,750 train postings/64,080 encounters
and reproduced validation's 10,176/7,408. No additional test configuration search
was performed.

## Classification results

Selected settings from [`selection.json`](evidence/evaluation/selection.json):

| Model | Depth | Min leaf | Exact threshold |
|---|---:|---:|---:|
| Lexical | — | — | 0.3484848484848485 |
| Custom CART | 8 | 2 | 0.43478260869565216 |
| Custom Random Forest | 12 | 5 | 0.2347583241835786 |
| sklearn Random Forest | 12 | 2 | 0.2294045757232059 |

Lexical score is the maximum of title-word and title-trigram Jaccard (columns
0 and 2). Matching accepts `score >= threshold`; the learners' separate built-in
`predict` method uses `score > 0.5` and is not used for these decisions.

**Validation**, 2,078 candidates, 220 total gold pairs:

| Model | Precision | Recall | F1 | Candidate-only AP | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lexical | 0.700441 | 0.722727 | 0.711409 | 0.794018 | 159 | 68 | 61 |
| Custom CART | 0.966851 | 0.795455 | 0.872818 | 0.903180 | 175 | 6 | 45 |
| Custom Random Forest | 0.913876 | 0.868182 | 0.890443 | 0.947879 | 191 | 18 | 29 |
| sklearn Random Forest | 0.899083 | 0.890909 | 0.894977 | 0.951990 | 196 | 22 | 24 |

**First frozen held-out test**, 2,053 candidates, 219 total gold pairs, from
[`test_metrics.json`](evidence/evaluation/test_metrics.json):

| Model | Precision | Recall | F1 | Candidate-only AP | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Lexical | 0.679654 | 0.716895 | 0.697778 | 0.712905 | 157 | 74 | 62 |
| Custom CART | 0.919786 | 0.785388 | 0.847291 | 0.852507 | 172 | 15 | 47 |
| Custom Random Forest | 0.869955 | 0.885845 | 0.877828 | 0.938996 | 194 | 29 | 25 |
| sklearn Random Forest | 0.873303 | 0.881279 | 0.877273 | 0.939274 | 193 | 28 | 26 |

AP is computed only over retrieved candidates, so it excludes retrieval misses
from its ranking population. It is not full-universe average precision. In contrast,
recall/F1 use all gold pairs. No accuracy headline is used on the imbalanced pool.

The custom forest's exact held-out F1 is **0.8778280542986425**, versus
**0.8772727272727273** for sklearn. This small difference on one fixed split does
not establish superiority. These are independently validation-selected settings
(custom leaf 5, sklearn leaf 2). The full report also compares equal depth/leaf
configurations on validation: no sklearn-minus-custom gap exceeds 0.03.
RNG sequences, duplicate-occurrence versus weighted-bootstrap constraints, exact
feature-subset search, tie-breaking and float64/float32 arithmetic differ.

Original fit times are single fits on 6,426 rows; batch inference times are medians
of five passes after one warmup, excluding retrieval, features and model loading:

| Model | Fit s | Validation inference s (2,078 rows) | Test inference s (2,053 rows) |
|---|---:|---:|---:|
| Lexical | 0.000000000 | 0.000033416 | 0.000033708 |
| Custom CART | 0.044069208 | 0.002752875 | 0.002446375 |
| Custom Random Forest | 0.561180125 | 0.085774458 | 0.084149416 |
| sklearn Random Forest | 0.140510500 | 0.003992500 | 0.004260875 |

## End-to-end error accounting

For the custom forest: 194 TP, 29 FP and **25 FN = 2 retrieval misses + 23 retrieved
positives rejected**. The retrieval ceiling is 217/219 = 0.9908675799086758.
Conditional classifier recall is 194/217 = 0.8940092165898618, so end-to-end recall
is `(217/219) × (194/217) = 194/219 = 0.8858447488584474`. An absent candidate
cannot be recovered by a later classifier.

Observed examples in the saved test traces include the black/plum Nikon S550 color
variant and propane/natural-gas Weber E-310 variant accepted as matches. Gold LG
washer pair 27729/203379250 ranked 11th, below `word10`'s cutoff; the other miss
ranked 27th ([saved rank analysis](evidence/evaluation/retrieval_miss_analysis.json)).
These are observations, not proven causal explanations of forest decisions.
Diagnostic scores computed for missed pairs do not enter evaluation. No error
example was used to change features, retrieval, thresholds or model choice.

## Performance results

The original [benchmark](evidence/evaluation/benchmark.json) used the first twenty raw-ID-ordered
training queries against the same 655 training targets. A ten-query exhaustive
pilot took 2.623138332972303 s. Each method then had one complete warmup and five
measured passes, alternating execution order. Both compute the same twelve features
**uncached**, score the same frozen custom forest, and apply its same threshold.
Target preparation/index construction/model load are outside query timing.
Query preparation, retrieval (indexed path), features, prediction, thresholding
and output sorting are inside. Both accepted 23 pairs; subset candidate recall
was 20/20, which is not a guarantee for other queries.

| Run / method | Pairs per pass | Median 20-query pass s | Median query ms | p95 query ms |
|---|---:|---:|---:|---:|
| Original indexed | 200 | 0.088286792 | 3.747146 | 6.486614 |
| Original exhaustive | 13,100 | 5.460002749 | 249.895791 | 376.703769 |
| Optimized replay indexed | 200 | 0.029065874 | 1.255375 | 2.074525 |
| Optimized replay exhaustive | 13,100 | 1.742962623 | 80.863959 | 115.614765 |

Original measured median-pass speedup: **61.84393612475235×**.
The separate [optimized replay](evidence/performance/benchmark.json) measures
**59.965945371042665×**. Each query percentile uses 100 actual
samples with NumPy's linear method. Both methods benefit from the DP improvement;
the indexed/exhaustive ratio need not increase. Original and replay absolute times also differ
on unchanged code, so the paired comparison below is the evidence for optimization
speedup, not a ratio across historical runs.

Original uncached feature times were 2.887775125 s train,
0.952310584 s validation, 0.919787417 s test.
The original benchmark separately measured 2.856790292 s
for all training features and 0.059428792 s for verified cache
reuse. A cache hit performs zero feature recomputation; it must not be compared
with the uncached exhaustive path as an algorithmic speedup.

Replay setup measurements: full `prepare_data` (verified loading, normalization, all
2,173 records and output writes) took 0.086301208 s
in a separate single pass. Training word-index build median was
0.001262958 s and candidate retrieval/array construction
0.018807291 s. These are distinct scopes;
partition loading additionally rechecks prepared provenance.

### Measured bottlenecks and the retained change

[`profile_workload.py`](../tools/profile_workload.py) runs full training and validation
candidate workloads, fixed selected-parameter fits and validation inference. Wall
measurements have one warmup and five passes. `cProfile` and `tracemalloc` run
separately; their overhead is excluded from wall-time comparisons.

Before optimization, the train feature profile spent 2.663298530 of 2.744966999 s
(**97.02% cumulative**) inside `similarity.levenshtein`; validation spent
0.882172627 of 0.907655916 s (**97.19%**). The train profile counted 17,696,657
`min` calls across the whole feature path (17,690,231 DP cells; the remainder is
containment). Call profiling magnifies Python-call overhead; these profile times
are attribution measurements, not uninstrumented latency.

Normalization, token/trigram construction and price parsing have **zero calls** in
these feature profiles: `PreparedRecord` already caches them. Jaccard contributed
0.033708372 s cumulatively on train; position checking 0.013143481 s. Set unions,
small conversions and tuple allocations were minor beside the DP. No extra pair
cache or repeated-feature framework was introduced.

The only algorithmic edit replaces the DP cell's three-argument `min` call with
named deletion/insertion/substitution costs and two integer comparisons. The exact
recurrence, two reused rows, shorter column dimension and public interface remain.
No prefix trimming, approximate distance or feature/schema change was introduced.
The existing full-table oracle and two-row memory checks still pass.

An alternating paired experiment uses the original source snapshot and current
implementation on identical prepared records and ordered pairs, recomputing every
feature matrix. Equality checks and scoring occur outside timers. Each implementation
has one warmup and five passes ([every sample](evidence/performance/paired_dp.json)):

| Workload | Rows | Before median s | After median s | Speedup |
|---|---:|---:|---:|---:|
| train | 6,426 | 1.464323292 | 0.843670458 | 1.735658× |
| validation | 2,078 | 0.492912000 | 0.282651708 | 1.743885× |

Every candidate/order, all twelve features, frozen forest scores, threshold
decisions and resulting accepted pair sets agree exactly. After optimization,
Levenshtein still accounts for 0.933541577/1.008660417 s (92.55%) of the train
feature profile. The remaining quadratic work is a deliberate exactness trade-off.
The separate after-profile validation wall median was 0.322969041 s; the paired
0.282651708 s result above is the controlled comparison, not a hand-picked
replacement. Raw samples document the run-to-run variation.

Other profiles justified retaining the existing code:

- Retrieval: 0.047454 s profiled train total; `InvertedIndex.query` 0.040777009 s,
  including heap selection 0.021682551 s. Dictionary accumulation visits only
  encountered postings. Only query terms and retained K entries are sorted.
  Its uninstrumented 0.018861875 s baseline is small beside features.
- CART: 0.025364715 s profiled fit; `_best_split` 0.024263835 s, including 984
  per-feature sorts (0.010059013 s in the NumPy `argsort` wrapper). Prefix counts,
  legal-boundary arrays and one final partition retain straightforward lifetimes.
  The current fit is too small to justify changing split semantics or architecture.
- Forest inference: 0.045973874 s profiled validation scoring; 32 calls to
  `_positive_probabilities` take 0.045852585 s cumulatively (99.74%). Python traversal,
  not temporary probability-array allocation, dominates. Inference accumulates
  one length-C vector per tree and never retains a C×T array. No vectorized tree
  representation or parallel prediction was added.

Fixed-parameter refit/scoring medians in the optimized profiling run (five passes, identical
shared train matrix and validation matrix; no threshold selection):

| Model | Fit s (6,426 rows) | Validation scoring s (2,078 rows) |
|---|---:|---:|
| Custom CART | 0.022832208 | 0.001243958 |
| Custom Random Forest | 0.276410125 | 0.044895417 |
| sklearn Random Forest | 0.072868958 | 0.002079167 |

Custom refits reproduce entire saved tree structures; all three learned refits
reproduce saved validation scores exactly. Training and inference implementations
were not optimized, so differences from original timings must not be credited to the DP edit.
The [before](evidence/performance/before/profile.json) and [after](evidence/performance/after/profile.json)
reports retain source hashes, versions, profiles, wall samples and allocation scopes.

### Memory

| Item | Bytes / count |
|---|---:|
| Training float64 X, 6,426 × 12 | 616,896 bytes |
| Training int64 pair positions | 102,816 bytes |
| Training int64 labels | 51,408 bytes |
| Validation X / pairs | 199,488 / 33,248 bytes |
| Test X / pairs | 197,088 / 32,848 bytes |
| Custom CART nodes / serialized JSON | 163 / 50,770 bytes |
| Custom forest nodes / serialized JSON | 7,288 / 2,596,803 bytes |
| sklearn forest nodes / serialized pickle | 9,440 / 765,673 bytes |
| Bundle load + 655 prepared targets/index + one query, tracemalloc peak | 6,778,368 bytes |
| Validation feature computation alone, tracemalloc peak | 212,444 bytes |
| Validation retrieval alone, tracemalloc peak | 123,792 bytes |
| Custom CART / forest fit, tracemalloc peak | 582,817 / 1,767,158 bytes |
| Custom forest validation scoring, tracemalloc peak | 67,120 bytes |

The first eight rows are from the original evaluation; the complete benchmark's tracked peak
also matched on replay. Per-operation allocation measurements are separate profiling observations
with inputs already resident, and include the returned object. `tracemalloc` does
not necessarily cover native allocations or total process RSS. Indented JSON versus
binary pickle sizes are not an algorithm-memory comparison. Models retain nodes,
not training arrays. No new dependency was added to runtime.

## Equivalence verification

The recorded optimization checks passed **767 tests before** the change (8.62 s)
and **767 after** (8.17 s), with no skips. The [saved test output](evidence/performance/pytest-final.txt)
records the latter run. Independent full-sort/cosine, full-table edit-distance,
brute-force Gini split and direct threshold-count oracles remain in the test suite.

The [verification extract](evidence/performance/verification.json) records
byte-identical original artifacts and regenerated preparation; exact
train/validation/test candidate arrays and feature matrices; and all four saved
validation/test score arrays and accepted pair sets. Metrics were independently
recounted with sets, and candidate-only AP was recomputed by grouped precision
increments (agreement within 1e-14). All seven validation retrieval configurations
reproduce their non-timing metrics. Retrieval settings, features, splits,
hyperparameters and thresholds stayed fixed.

A fresh subprocess ran the actual `match` CLI from an empty temporary working
directory with sklearn/scipy imports and gold-loading functions forbidden. Against
all 1,092 Buy records, the complete Omnimount query reproduced both original matches,
including features and scores.

Evaluation and benchmarking used byte-for-byte copies of the freeze-listed
artifacts in a fresh output directory. Profiling refit only the selected
configurations; it did not repeat model selection. The full training command is
covered by the synthetic integration test. See the [README](../README.md#reproduce-from-a-clean-checkout)
for current reproduction commands and the [evidence index](evidence/README.md)
for original reports, public extracts and safe frozen-run replay instructions.
Running the profiler on current source measures the optimized implementation;
the paired experiment retains the original source snapshot for comparison.

Integration checks cover whole-component splits, positive-only graph edges,
unique postings and unseen-term norms, stable heap ties and untruncated unions,
attribute-only features and cache rejection, scored/actual CART partitions,
zero-gain termination, bootstrap multiplicities, fresh per-node features,
shared matrices, freeze-before-test and serialization round trips.

## Limitations and remaining work

- Abt–Buy is small, old and limited in category/time coverage. All its records have
  a known partner, so unmatched-query rejection and deployment prevalence are not
  well represented. No cross-dataset generalization or production scale was measured.
- Buy descriptions are missing for 441/1,092 records; prices for 663/1,081 Abt and
  502/1,092 Buy records. Indicators preserve missingness without imputing equality.
- Model tokens are a heuristic; e.g. a token containing letters and digits can be
  a specification rather than a model number. Color, fuel and bundle variants can
  share most tokens. ASCII word normalization also limits language coverage.
- Lexical retrieval has no semantic representation and can exclude a correct pair.
  The frozen test examples did not motivate a new retrieval or model choice.
- Forest scores are uncalibrated averages of empirical leaf fractions. The
  many-to-many output has no global assignment constraint or business-rule layer.
- Exact Python edit distance remains the main feature cost, and Python tree
  traversal is slower than sklearn's compiled kernels. Predictive quality and
  computational speed are distinct dimensions.

The validation-only edit-feature ablation was not run. Broader datasets,
feature ablations or calibration would require a separately defined protocol
and would not replace the frozen evaluation.
