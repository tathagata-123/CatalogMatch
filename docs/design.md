# CatalogMatch design

This document describes the architecture, algorithmic invariants, complexity and
implementation contracts. The frozen experiment is described in
[results.md](results.md); [technical study notes](interview.md) provide worked
examples and reconstruction exercises.

## Records and normalization

`data.RawRecord` retains source, string ID, original title, description,
manufacturer, and raw price. Abt manufacturer is empty because its schema has no
manufacturer column. `data.PreparedRecord` contains its original record, normalized
title/description, immutable word sets, title trigrams, heuristic model tokens,
numeric title tokens, and optional price. IDs remain metadata and are never scored.

`normalize.normalize_text` decodes HTML entities, applies Unicode NFKC and case
folding, extracts `[a-z0-9]+` runs, and joins them with single spaces. Descriptions
first use HTMLParser to extract actual markup text; entity-escaped angle brackets
remain literal text. This deliberately ASCII tokenization does not transliterate
accented letters. Character trigrams include spaces; nonempty strings shorter than
three characters become singleton sets. Model extraction uses NFKC/case-folded raw
titles and `[a-z0-9]+(?:[-./][a-z0-9]+)*`, removes internal separators, and keeps
tokens of length at least four with both letters and digits. It never joins words
across whitespace. Model tokens are only heuristic evidence. Numeric tokens retain
digit-only normalized title words; there is no unit conversion.

`normalize.parse_price` accepts positive finite decimal USD prices with an optional
leading `$` and correctly grouped commas. It rejects signs, exponent notation,
zero, nonfinite values, and malformed grouping. Surrounding whitespace is ignored.
Audit counts distinguish blanks from malformed nonblank prices; no rows are dropped.

## Entity splits and leakage

`data.build_positive_graph` creates one vertex `(source, raw_id)` per record,
including unmatched singletons. Only positive gold pairs add undirected bipartite
edges. Negative pairs are never accepted as graph input. Namespaces prevent raw-ID
collisions between sources. The graph API requires positive string-ID pairs;
labeled candidate triples are rejected rather than interpreted as edges.

`data.connected_components` uses an explicit DFS stack and marks vertices on push.
Graph construction and DFS take O(V + E) time and space. After traversal, vertices
are sorted inside components and components by minimum vertex; sorting adds up to
O(V log V) work. `split_components` puts canonical components into matched,
Abt-only, and Buy-only buckets. One `random.Random(42)` shuffles nonempty buckets
in that order. Each bucket assigns floor(3c/5), floor(c/5), and the remainder to
train/validation/test. Small buckets can have empty train or validation allocations.

Individual pairs can share endpoints, and chains of positives can share an entity.
Keeping each entire component together prevents both forms of cross-partition
leakage. `validate_partitions` checks exact coverage, no duplicate membership,
component coverage without repeated vertices, whole-component assignment, and
every gold edge. `verify_manifest` additionally
requires matching CSV SHA-256 hashes, correct component counts, and exactly the
same assignments as the specified deterministic algorithm. It returns the checked
manifest memberships. `verify_original` checks the original dataset's expected
row counts, missing values, component shapes, and partition sizes.

Split membership, gold labels, IDs, and dense positions are evaluation metadata,
never predictive inputs. Prepared JSONL separates metadata/raw fields from product
attributes. The training baseline selects training records before preparing or
scoring them; only training gold labels are used to summarize accepted pairs.

## Baseline and module responsibilities

`similarity.jaccard` divides set-intersection size by set-union size and returns
zero if either set is empty. `lexical_score` takes the maximum of title-word and
title-trigram Jaccard. `training_baseline` prepares each training record once and
streams accepted pairs from the exhaustive Cartesian product. Complexity is
O(A × B × title-set work), with no score matrix. It permits many-to-many predictions.
The default threshold 0.5 is an initial smoke-test configuration, not tuned.

`data.py` handles records, CSV auditing, provenance, and splits. `normalize.py`
handles text/price preparation. `similarity.py` handles lexical scores and the
training baseline. `cli.py` orchestrates checked preparation and baseline outputs;
`__main__.py` exposes `python -m catalogmatch`.

## Record preparation and splitting: reading order

1. `data.prepare_record`: follow one raw title/description through normalization,
   token extraction, trigrams, and optional price parsing; the raw record survives.
2. `data.build_positive_graph`: see namespaced vertices and symmetric positive edges.
3. `data.connected_components`: trace stack push/pop and the visited invariant.
4. `data.split_components`, then `validate_partitions` and `verify_manifest`: see why
   component assignment prevents shared endpoints or positive chains from leaking.
5. `similarity.jaccard`, then `lexical_score` and `training_baseline`: trace the
   overlap calculation and its use on training product attributes only.

## Retrieval indexes and cosine

`retrieval.InvertedIndex.build` accepts term collections in the prepared target
catalog's position order. It deduplicates each collection and appends its integer
position to `postings[term]`. Each posting list is therefore unique and ascending
without a sorting pass. The same class indexes title words and title character
trigrams; `build_indexes` takes one partition-local target list for both channels.
`df`, `idf`, `n_records`, and `norms` store document frequencies, weights, catalog
size, and precomputed target-vector lengths. Raw IDs and splits never enter scores.

Binary TF counts each term once. With N target records,

```
idf(t) = 1 + ln((N + 1) / (df(t) + 1))
norm(d) = sqrt(sum(idf(t)^2 for distinct t in d))
cosine(q, d) = sum(idf(t)^2 for t shared by q and d) / (norm(q) * norm(d))
```

`InvertedIndex.query` deduplicates and sorts query terms. Every distinct query
term contributes to its norm, including unknown terms with df=0. In sorted term
order, it traverses postings and adds squared weights into `score_by_id[position]`.
It normalizes only encountered candidates and calls `bounded_top_k`. No full
target scan or zero-score padding occurs. Empty queries/catalogs return no hits;
nonpositive or noninteger K is rejected. Norms use `math.fsum` over positive
weights so hash-set order does not change their values; query dot products use
the specified sorted-term accumulation order. Tests also compare subprocesses
with different Python hash seeds.

Character trigrams reuse title preparation, including spaces and singleton
sets for nonempty titles shorter than three characters. For example, `alpha` and
`alphabeta` share no complete word but share `alp`, `lph`, and `pha`. Trigrams can
recover such pairs, though common fragments also increase postings work and the
number of competing candidates.

## Heap and channel union

`BoundedTopK.entries` is an ordinary list of `(score, -position)` keys. Each parent
is less than or equal to its children; the root is the worst retained candidate.
`add` appends and calls `sift_up` until K entries are retained. Afterwards, only a
strictly better key replaces the root, followed by `sift_down`. Equal scores favor
smaller positions. `best_first` sorts only the retained entries by descending key
and converts them back to `(position, score)` pairs. It never sorts all U scores.

`retrieve_candidates` uses the same query/heap algorithm independently for word
and character channels. `Candidate` retains `target_position`, optional
`word_score`/`gram_score`, and a tuple of selected channels (`word`, `char`). An
absent score means that channel did not select the record; it does not assert a
zero cosine. Single-channel lists preserve best-first order. A union deduplicates
by target position, preserves both available scores, and returns ascending target
positions. It has at most K_word + K_gram entries, with no blended ranking or
second truncation.

## Complexity and counters

Let Z be distinct term-record associations, V vocabulary size, N target records,
q distinct query terms, P visited postings entries, and U distinct encountered
targets. With already tokenized sets, construction takes expected O(Z + N) time
(usually written O(Z) when each record is nonempty) and O(Z + V + N) storage.
Temporary deduplicated documents also use O(Z + N) space and are discarded after
norm calculation. If input terms contain repetitions, reading them adds their
input length. Dictionary/set access is expected O(1) under ordinary hash
distribution, not an adversarial worst-case guarantee; these bounds treat term
hashing/comparison as bounded work, or already cached for reused strings.

Sorting query terms costs O(q log q). Norm calculation costs O(q), traversal and
accumulation expected O(P), and normalization O(U). Heap selection costs
O(U log K), followed by O(K log K) retained sorting, giving approximately
O(P + q log q + U log K + K log K) overall. For empty/small cases, use log(max(2,K))
and the actual retained count min(U,K). Query working memory is O(q + U + K):
the score map still stores **all U candidates**, even when the heap holds only K.
`RetrievalStats` counts P and U before truncation. Evaluation sums these counters
over queries and channels, so a target encountered in both channels counts twice
in U, while generated candidate pairs are counted once after the union.

A union adds expected O(K_word + K_gram) merge work and O(C log C) sorting for
C distinct retained positions. Its cost includes both channel queries. Common
terms may make U approach N; indexing does not promise sublinear work for every
query.

## Validation selection and provenance

`data.load_prepared_partition` requires the raw dataset and split already checked
by `verify_original`. It validates prepared-file hashes, the original manifest
identity, copied memberships, full record coverage, raw fields, and dense
positions. For the requested train/validation partition, it checks cached
attributes against the current preparation functions. `NORMALIZER_VERSION`
identifies the normalization rules; older prepared audits must be
regenerated with `prepare`. Training and validation use separately loaded target
lists and separate indexes. The retrieval CLI accepts only `val`/`validation`.
Test membership is checked for integrity; access to test attributes requires
an explicit opt-in from frozen evaluation.

`evaluate_retrieval` evaluates exactly word10, word25, word50, char10, char25,
union10+10, union25+25, in that order. Only evaluation consumes validation gold
positions, after candidate retrieval. Recall is retrieved gold pairs divided by
all validation gold pairs; reduction is 1 - candidates/(left count × right count).
No labels are injected into results. A validation set with no gold pairs cannot
support selection and raises an error.

`select_configuration` chooses the fewest actual pairs among recall >= 0.98,
then the fixed order. If none qualifies, it chooses highest recall, then fewest
pairs, then fixed order. The engineering target is not guaranteed. Top-K can
exclude a true pair because it has no term overlap or because K competitors rank
ahead of it. Even higher K cannot recover zero-overlap pairs in that channel.
Candidate recall measures the ceiling available to later classification, not
classifier precision or final matching accuracy.

`cli.run_retrieval_eval` writes the full comparison and selection to
`runs/retrieval/selected.json`, with raw/prepared hashes, manifest identity, split
version/seed, normalizer/retrieval versions, and parameters. Timings use
`perf_counter`, one pass per configuration, with indexes built once. Per-query
retrieval latency includes counters, heap selection, and union; it excludes
loading, preparation, index construction, gold checks, and output. Index build
time is reported separately. These are local wall-clock observations, not timing
thresholds in tests. Timing fields can vary across runs; candidates and selection
remain deterministic.

## Retrieval reading order and trace

1. `InvertedIndex.build`: binary terms, ascending postings, df/IDF, and target norms.
2. `InvertedIndex.query`: unseen-term query norms, postings traversal, score-map
   accumulation, normalization, and the P/U counters.
3. `BoundedTopK.add`, `sift_up`, `sift_down`, `best_first`, then `bounded_top_k`:
   trace parent/child comparisons and the worst-retained root invariant.
4. `retrieve_candidates` and `normalize.character_trigrams`: explain independent
   channel selection, word-free overlaps, union deduplication, and its bound.
5. `evaluate_retrieval` and `select_configuration`: distinguish missed gold pairs,
   candidate cost, observed latency, and the fixed validation selection policy.
6. `tests/test_retrieval.py::exhaustive_cosine` and the hand-calculated test:
   understand the independent full-scan/full-sort reference before optimizations.

Trace `InvertedIndex.build([{'a'}, {'a', 'b'}, {'c'}])`, then query
`{'a', 'b', 'unseen'}` with K=1. N=3 gives postings `a:[0,1]`, `b:[1]`, `c:[2]`;
df(a)=2, df(b)=df(c)=1. The weights are a=1.287682, b=c=1.693147, and
unseen=2.386294. Target norms are `[1.287682, 2.127175, 1.693147]`; the query norm
is 3.196760, including the unseen term.

| Step in `query` | Score accumulator (dot products) |
|---|---|
| Visit `a` postings | `{0: 1.658125, 1: 1.658125}` |
| Visit `b` postings | `{0: 1.658125, 1: 4.524872}` |
| Visit `unseen` | unchanged; no postings |
| Divide by query × target norms | `{0: 0.402809, 1: 0.665416}` |

There are P=3 visits and U=2 candidates. `bounded_top_k` first inserts
`(0.402809, 0)`. Candidate 1 has the better key `(0.665416, -1)`, so `add` replaces
the root and calls `sift_down` (a one-entry heap needs no swap). `best_first`
returns `[(1, 0.665416)]`. Target 2 is ineligible; it is never scored or padded into
the result. With K=2, both overlapping targets survive in order `[1, 0]`.

## Exact edit distance

`similarity.levenshtein(a, b)` computes unit-cost insertion, deletion, and
substitution distance directly. Define `dp[i][j]` as the fewest edits transforming
`a[:i]` into `b[:j]`. Empty prefixes give `dp[i][0] = i` and `dp[0][j] = j`.
The last operation must delete from `a`, insert from `b`, or align the last two
characters, so:

```
cost = 0 if a[i-1] == b[j-1] else 1
dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
```

The independent `tests/test_similarity.py::full_table_levenshtein` retains the
entire table: O(mn) time and O(mn) space. Production keeps exactly two reusable
rows. Before a cell is computed, `previous[j]` is above, `previous[j-1]` is
diagonal, and `current[j-1]` is the completed cell to its left. After each row,
swap the two buffers. Older rows can never be referenced by the recurrence.
Putting the shorter string in columns gives O(min(m,n)) auxiliary space while
retaining O(mn) time (including row initialization, O((m+1)(n+1)) overall).
Empty-string distance returns the other length directly. There is no truncation,
recursion, approximation, or external edit-distance implementation.

The inner loop spells out deletion, insertion and substitution costs, then uses two
integer comparisons to find their minimum. This removes one Python function call
per DP cell without changing the recurrence or row allocation. The decision to
retain it comes from alternating before/after wall measurements, not from the
call profiler's timings alone.

`edit_similarity` consumes already normalized titles. It returns
`1 - distance / max(m,n)` for two nonempty titles, otherwise zero. In particular,
`levenshtein('', '') == 0` but `edit_similarity('', '') == 0.0`: two missing titles
are not identity evidence. Jaccard divides intersection size by union size;
containment divides by the smaller set size. Thus a subset gets containment 1
without necessarily getting Jaccard 1. Both return zero if either set is empty.

## Ordered twelve-feature contract

`features.FEATURE_NAMES` is the single immutable tuple defining column order.
`pair_features` returns a tuple of twelve floats from `PreparedRecord` attributes.
The schema version is `product_pair_12_v1`; semantic changes must bump this version.

| Index | Name | Definition |
|---|---|---|
| 0 | `title_token_jaccard` | Title word-set intersection / union |
| 1 | `title_token_containment` | Title word-set intersection / smaller set size |
| 2 | `title_trigram_jaccard` | Title character-trigram Jaccard |
| 3 | `title_edit_similarity` | Normalized exact Levenshtein similarity |
| 4 | `model_token_jaccard` | Extracted model-token Jaccard |
| 5 | `model_token_conflict` | 1 iff both model sets are nonempty and disjoint |
| 6 | `model_tokens_both_present` | 1 iff both model sets are nonempty |
| 7 | `description_token_jaccard` | Description word-set Jaccard |
| 8 | `descriptions_both_present` | 1 iff both normalized descriptions are nonempty |
| 9 | `relative_price_difference` | abs(a-b)/max(a,b) for two valid prices; else 0 |
| 10 | `prices_both_present` | 1 iff both parsed prices are present |
| 11 | `numeric_token_jaccard` | Jaccard of all-digit title-token sets |

All values must be finite and in [0,1]; invalid values raise `ValueError` naming
the feature. Prepared prices must be positive finite values or `None` (forged
invalid prepared prices raise an error). Raw malformed, zero, or absent prices
already become `None` during existing preparation. A missing price gives columns
9/10 = `(0,0)`; equal known prices give `(0,1)`. Missing normalized descriptions
produce description similarity/presence `(0,0)`, while identical nonempty
descriptions give `(1,1)`. Missing models produce Jaccard/conflict/presence
`(0,0,0)`, while disjoint nonempty model sets give `(0,1,1)`.

No manufacturer, ID, label, component, partition, or retrieval-score feature is
included. Model disagreement remains a feature, not a rejection: heuristic model
tokens can be incomplete or noisy. Future classification may learn its usefulness
in combination with other attributes. Extraction does not normalize or parse again.

## Matrices, caching, and boundaries

`feature_matrix(pairs, left_records, right_records)` accepts a sized sequence of
integer position pairs (including a NumPy integer array). It allocates one
`float64` array with shape `(len(pairs), 12)`, then fills rows with `pair_features`.
Order and duplicate rows survive unchanged. Empty inputs yield `(0,12)`. Negative,
out-of-range, boolean, noninteger, or malformed positions raise errors identifying
the row and, where applicable, side. No invalid pair is silently removed.

Only candidates from `retrieve_candidates` are processed. Feature storage is
O(12C) for C pairs; edit work is the sum of O(title_left_length × title_right_length)
over those pairs, alongside set comparisons. No Cartesian product or permanent
all-pairs edit-distance cache is constructed. Keeping the returned `X` in memory
lets future model experiments reuse it directly.

`cli.run_features` verifies the existing data and preparation, then
`load_selected_config` validates saved selection provenance, versions, parameters
in the selected config, membership in the seven allowed configurations, and
agreement with the saved validation comparison's selection rule. It does not
rerun matching-quality evaluation or rewrite `runs/retrieval/selected.json`.
Training and validation use separate partition-local indexes. Pair order follows
left-record position, then the existing candidate order. Scores never enter
`feature_matrix`. The loader checks all split memberships for integrity, but
returns only the requested training/validation attributes.

`cached_feature_matrix` stores `<split>.npz` (`X` float64 and `pairs` int64) beside
`<split>.json`. Metadata identity includes:

- Raw CSV SHA-256 hashes, split-manifest hash, partition, split version/seed.
- Normalizer version, feature schema version, exact ordered feature names.
- Retrieval configuration/version and selected-file hash.
- Candidate count and an order-sensitive SHA-256 of positions and namespaced raw IDs.
- Separate SHA-256 hashes of the actual ordered prepared records' attributes/IDs.

The last checks also catch direct Python callers changing prepared contents without
updating provenance. IDs identify cached rows only; they do not affect feature
values. The Python cache caller must supply verified provenance; the CLI does so.

Cache reuse requires identical identity, a matching NPZ byte checksum, exactly the
expected archive keys, exact pair positions/order, float64 shape `(C,12)`, and
finite bounded values. Loading always uses `allow_pickle=False`. Missing, stale,
malformed, or corrupted caches are regenerated with a reported reason. Complete
temporary files are individually atomically replaced, with metadata published
last; the byte checksum detects an interrupted or mixed two-file generation.
This is a local disposable cache, not a multi-writer transaction store.

The CLI prints pair counts, shape/dtype, feature extraction time, retrieval time,
cache time, and reuse status, and saves `report.json`. Feature timing excludes
loading, preparation verification, retrieval, and cache I/O. Hits report zero new
feature-computation time. `inspect-features` displays raw titles, descriptions,
prices, and all twelve named values for requested training/validation IDs.

Boundaries remain: `data`/`normalize` prepare records; `retrieval` chooses candidate
pairs; `similarity` supplies primitives; `features` computes/caches attribute
comparisons; `cli` supplies verified provenance and presentation. CART and
Random Forest consume numerical matrices. Training and inference must use
the same column semantics, or a learned column threshold would test the wrong
attribute. The product-model bundle validates that shared feature order.

## Feature reading order and worked example

1. `tests/test_similarity.py::full_table_levenshtein`: derive the recurrence by
   considering the final edit, and trace the empty-prefix base cases.
2. `similarity.levenshtein`: identify above/left/diagonal and explain the buffer
   swap and shorter-column choice. Study `test_levenshtein_randomized_oracle` and
   `test_levenshtein_memory_is_two_short_rows` for independent checks.
3. `similarity.jaccard`, `containment`, `edit_similarity`: distinguish set overlap,
   subset coverage, character order, and the missing-title convention.
4. `data.prepare_record`, then `features.pair_features`: follow prepared attributes,
   soft model conflict, and explicit missing-value indicators.
5. `features.feature_matrix`, `feature_cache_identity`, `cached_feature_matrix`:
   follow candidate order, fixed schema, stale-cache rejection, and reuse.
6. `cli.load_selected_config`, `run_features`, `inspect_features`: see the separation
   of retrieval, feature computation, provenance, and inspection.

Concrete noisy pair (also asserted in `test_worked_noisy_title_example`):

| Attribute | Abt | Buy |
|---|---|---|
| Raw title | `Canon AB-123 Camera, 12 MP!` | `CANON AB123 camera 12MP` |
| Normalized title | `canon ab 123 camera 12 mp` | `canon ab123 camera 12mp` |
| Title words | `{canon, ab, 123, camera, 12, mp}` | `{canon, ab123, camera, 12mp}` |
| Model tokens | `{ab123}` | `{ab123, 12mp}` |
| Numeric tokens | `{123, 12}` | `{}` |
| Raw description | `<p>Digital camera</p>` | `Digital camera kit` |
| Description words | `{digital, camera}` | `{digital, camera, kit}` |
| Raw / parsed price | `$100.00` / 100 | `80` / 80 |

The heuristic treats `12mp` as a model token; normalization deliberately does not
split it into `12` and `mp`. Title word sets share two tokens with union size eight.
Their trigram sets have 22 and 21 members, with intersection 17 and union 26.
Deleting the space after `ab` and the space before `mp` transforms the normalized
Abt title into the Buy title: distance 2, maximum length 25.

| Index / feature | Calculation | Value |
|---|---|---|
| 0 title token Jaccard | 2/8 | 0.25 |
| 1 title token containment | 2/4 | 0.5 |
| 2 title trigram Jaccard | 17/26 | 0.653846 |
| 3 title edit similarity | 1 - 2/25 | 0.92 |
| 4 model token Jaccard | 1/2 | 0.5 |
| 5 model conflict | shared `ab123` | 0.0 |
| 6 models both present | both nonempty | 1.0 |
| 7 description Jaccard | 2/3 | 0.666667 |
| 8 descriptions both present | both nonempty | 1.0 |
| 9 relative price difference | 20/100 | 0.2 |
| 10 prices both present | both valid | 1.0 |
| 11 numeric token Jaccard | Buy numeric set empty | 0.0 |

## CART nodes, inputs, and inference

`tree.DecisionTreeClassifier` uses a `TreeNode` dataclass with sample count,
positive count, positive probability, optional feature index/threshold, and two
optional children. A leaf has no split or children. An internal node has two
nonempty children whose sample and positive counts sum to its own. Counts include
every occurrence of a repeated bootstrap index. No node or fitted estimator
retains training matrices, labels, indices, sorting arrays, or random generators.
Inspection uses `root_`, `n_features_in_`, and the node fields directly.

Defaults are depth 8, minimum split size 2, minimum leaf size 2, all features, and
seed 42. Depth is a nonnegative integer or `None`; root depth is zero. Minimum
split size is an integer >= 2 and leaf size an integer >= 1. `max_features` is
`None`, `'sqrt'`, or an integer from 1 through the input dimension. Seeds are
nonnegative integers; booleans are rejected as integer parameters. Parameters
are checked at construction and again at fit, including dimension-dependent limits.

Training accepts nonempty, finite, two-dimensional real integer/float arrays with
at least one feature; it converts features to float64. Numerical distinctions are
therefore those representable after float64 conversion (large integers and wider
floating types can lose precision). Labels must be a one-dimensional array of
0/1 values, including boolean labels, with matching row count. Strings, object
arrays, complex values, NaNs, infinities, and malformed dimensions are rejected.
The twelve-feature pipeline needs no changes; synthetic data may have any positive
feature dimension. Optional `sample_indices` must be nonempty valid integer row
positions; their order and duplicates survive. The complete X/y inputs are
validated even if only a subset of rows is selected.

`_positive_probabilities` iteratively traverses each row: `value <= threshold`
goes left, otherwise right. It returns the reached leaf's positive fraction.
`predict_proba` always returns `[1-p, p]`, shape `(q,2)`, classes `[0,1]`, including
one-class training and `(0,2)` for empty valid prediction matrices. Prediction
checks fitted state, finite real inputs, and the original feature count.
`predict` uses strict `p > 0.5`; ties predict 0. Matching instead uses the
validation-selected threshold saved in the model bundle and accepts `score >= threshold`.

## Sorted split search and recursive construction

`_gini(positive_count, sample_count)` computes `2*p*(1-p)`. For left/right counts
`l,r` and impurities `G_l,G_r`, weighted impurity is
`l/(l+r)*G_l + r/(l+r)*G_r`; gain is parent Gini minus this value.

`_best_split` processes the selected feature indices in ascending order. For each
feature it gathers that node's values, uses `np.argsort(..., kind='stable')`, and
computes `np.cumsum(sorted_labels, dtype=np.int64)`. Boundary `k` means k sample
occurrences on the left. Only boundaries with distinct adjacent values and both
child sizes >= `min_samples_leaf` are scored. Prefix count `k-1` gives the left
positive count; subtracting it from the total gives the right count. Vectorized
arithmetic scores all valid boundaries without constructing their partitions.

Tie policy is **strictly lower computed impurity**, then lower feature index on
exact equality, then earliest boundary. There is no approximate impurity tie band;
tiny unequal computed values remain ordered. `np.argmin` supplies the first minimum
within a feature. The `1e-12` tolerance is used only to reject materially negative
gain. This policy is deterministic for the same inputs and numerical environment.

`_threshold(lo, hi)` uses `lo/2 + hi/2` to avoid overflow. If rounding violates
`lo <= threshold < hi`, it uses `lo`. Thus adjacent representable floats and
subnormals still split correctly; identical values never split. No approximate
binning or float32 conversion is used. `_build_tree` partitions indices once for
the chosen split and checks that the actual left count equals the scored count.

`_build_tree` first constructs a provisional leaf, then stops on purity, depth,
`n < min_samples_split`, or `n < 2*min_samples_leaf`. Otherwise it samples features
and searches. No legal split in that subset also means a leaf. A legal zero-gain
split is allowed: four-row XOR needs one before its children can separate the
classes. Gain below `-1e-12` raises an internal correctness error. Every split
strictly reduces both child populations. Ordinary recursion builds left then right
subtrees with row-index arrays; it never copies the full X matrix per child.
`None` removes the configured depth cap, but pathological deep trees remain subject
to Python's recursion limit; the intended depth-8/12 configurations avoid that issue.

## Bootstrap forest and per-node feature sampling

`forest.RandomForestClassifier` defaults to 32 trees, depth 8, split size 2, leaf
size 2, `'sqrt'` features, bootstrap enabled, and seed 42. Its `fit` validates X/y
once and initializes a master `np.random.default_rng`. For each tree it draws a
seed from `[0, int64_max)`, creates a separate generator, and draws n row indices
uniformly with replacement. Duplicate indices remain repeated occurrences, so they
affect impurity, probabilities, and minimum sample constraints. `bootstrap=False`
uses `arange(n)` once per tree, with every row appearing exactly once.

The tree's `_fit_with_indices(X, y, indices, rng)` continues that same generator
after the bootstrap draw. `_select_features` takes a fresh subset at every eligible
node, without replacement, then sorts it. `'sqrt'` means `max(1, floor(sqrt(d)))`,
so d=12 selects three. All-feature selection returns all indices without consuming
RNG draws. Exactly the selected subset is inspected; a constant/unsplittable subset
produces a leaf without trying extra features. Trees train sequentially and share
the same custom CART builder. `estimators_` and `tree_seeds_` permit inspection.

Forest inference accumulates each tree's leaf fraction into one length-q array,
divides by the number of trees, and returns `[1-mean, mean]`. It never votes over
hard classes or allocates a q-by-T probability matrix. It uses the same strict
`> 0.5` decision rule and two-column shape contract as the tree.

## JSON models and validation

Both learners have `to_dict` and classmethod `from_dict`; use standard
`json.dumps(..., allow_nan=False)` and `json.loads`. Schemas are
`catalogmatch.tree.v1` and `catalogmatch.forest.v1`. They store exact parameter
fields, input dimension, scalar node statistics, thresholds, and nested children.
The forest also stores the master seed, per-tree seeds, and all tree dictionaries.
Seed replay follows NumPy's default generator; reproducing training requires
compatible RNG behavior. Loading and inference use the stored tree structures.
There is no pickle, eval, training data, or CatalogMatch pipeline envelope.

Loading rejects unknown versions/fields, missing fields, invalid hyperparameters,
feature indices, nonfinite scalars, noninteger/invalid counts, inconsistent
probabilities, missing children, active leaf splits, shared/cyclic nodes, inconsistent
parent/child counts, stopping-rule violations, and impossible descendant thresholds.
Forests additionally check tree/seed counts, seed derivation, dimensions, parameters,
equal root sample counts, and equal root positives when bootstrap is disabled.
These are structural checks; without training data a loader cannot prove a valid
threshold was originally optimal or recover which rows trained a leaf.

## Learner complexity

For m occurrences at a node and s inspected features, each feature takes
O(m log m) sorting and O(m) prefix counting/scoring. Split search therefore costs
O(s*m*log m), with O(m) working arrays per feature, plus O(d) feature-index storage.
Only the winning partition is materialized. At a fixed depth, node populations
sum to n. A safe depth-capped training bound is O(s*n*D*log n), plus O(nd+n) input
validation/conversion and count work. D=0 performs validation and root counting.
The code sorts again at each node; it is **not** a presorted tree implementation
and does not inherit a more efficient presorting bound.

Training features occupy O(nd), labels/root indices O(n); conversion may allocate
another feature matrix if the input is not already float64. A node's temporary
sort, prefix, score, and partition arrays use O(m). Recursive callers may retain
child-index arrays until a recursive call returns: a safe worst-case bound for
active indices is O(nD), O(n) for balanced partitions with geometrically decreasing
sizes. The call stack uses O(D); K fitted nodes use O(K), with K <= 2n-1 and
K <= 2^(D+1)-1 for finite D. Fitted nodes retain only scalars and child pointers.

T sequential trees multiply training work by T and store O(sum(K_t)) nodes.
They reuse the validated X/y and require only one tree's training workspace at a
time. Prediction traversal costs O(D) per sample per tree, or O(T*D) for a forest
(more precisely O(T*(D+1)) including root leaves). For q rows, add O(qd) validation
and possible float64 conversion; output/accumulation uses O(q) space. Serialization
is linear in stored node count; recursion follows the tree depth. Deserialization
also copies feature path bounds, adding up to O(K*min(d,D)) work.

## Independent verification and scikit-learn differences

`tests/test_tree.py::brute_force_split` independently enumerates feature thresholds,
explicitly partitions row occurrences, and recounts labels. It shares no optimized
split/count helper. Thirty seeds, four leaf constraints, and ordinary/repeated
indices produce 240 randomized comparisons, with duplicate/tied and constant
features. Further tests cover no-split cases, hand Gini calculations, recursive
counts, zero-gain XOR, representable boundaries, prediction validation, and JSON
corruption. Forest tests instrument the bootstrap entry point, replay independent
generators, count eligible node selections, and check exact sampled-subset behavior.

Aligned sklearn comparisons check unique splits, noisy leaf probabilities,
physically expanded bootstrap rows, an eight-tree expanded-row ensemble, and a
nonbootstrap forest. All these comparisons agree. The expanded-row test also shows
why sklearn sample weights are not interchangeable with repeated indices for
minimum sample constraints: five occurrences of two distinct rows can split with
`min_samples_split=5`, while two weighted rows cannot.

Relevant deliberate differences:

- NumPy `default_rng` seed/bootstrap/feature sequences differ from sklearn's RNGs.
- Exact impurity ties prefer lower features/earlier boundaries; sklearn permutes
  features and may select a different tied split.
- Exactly s selected features are inspected; sklearn can inspect additional
  features to find a valid partition.
- Bootstrap multiplicities are explicit indices; sklearn's forest passes bootstrap
  counts as sample weights, with different distinct-row minimum-count semantics.
- Float64 features and representable thresholds retain distinctions that sklearn's
  float32 conversion can collapse; an adjacent-float regression test demonstrates it.
- One-class fits still expose two probability columns in these custom learners.

See the official [DecisionTreeClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
and [RandomForestClassifier reference](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
Identical seeds do not imply identical forests. An additional fixed synthetic
holdout check compares predictive quality with broad tolerances; it supplements
the exact oracles. No sklearn import exists in learner runtime modules.

## Learner reading order and trace

1. `tree._gini`: derive impurity from positive/sample counts.
2. `tree._best_split`, then `_threshold`: sort, prefix count, score legal boundaries,
   break ties, and guarantee that `<=` reproduces the scored partition.
3. `DecisionTreeClassifier._build_tree`: provisional leaf, stopping rules,
   zero-gain acceptance, one partition, and left/right recursion.
4. `DecisionTreeClassifier._positive_probabilities`, `predict_proba`, `predict`:
   iterative routing, leaf fractions, two columns, and class-zero ties.
5. `RandomForestClassifier.fit` and `DecisionTreeClassifier._fit_with_indices`:
   independent seeds, repeated bootstrap indices, and continuing RNG state.
6. `DecisionTreeClassifier._select_features`: fresh per-node subsets without
   replacement, sorted indices, and no additional-feature fallback.
7. `RandomForestClassifier.predict_proba`: accumulate probabilities, then average.

Take unsorted `(feature,label)` rows `[(4,1),(1,0),(3,1),(2,0)]`, leaf size 1.
Sorting gives values `[1,2,3,4]`, labels `[0,0,1,1]`, and cumulative positives
`[0,0,1,2]`. Parent Gini is 0.5.

| Threshold | Left (samples, positives) | Right (samples, positives) | Weighted Gini | Gain |
|---|---|---|---|---|
| 1.5 | (1,0) | (3,2) | 1/3 | 1/6 |
| 2.5 | (2,0) | (2,2) | 0 | 1/2 |
| 3.5 | (3,1) | (1,1) | 1/3 | 1/6 |

Choose 2.5. Left is rows with values `[1,2]`, probability 0; right is `[3,4]`,
probability 1. Both children are pure leaves. A separate three-tree forest returning
positive probabilities `[0.6,0.6,0.0]` averages to 0.4, returns `[0.6,0.4]`, and
predicts class 0, even though two of its three hard predictions would be class 1.

## End-to-end integration

`pipeline.py` connects retrieval, features and learners and defines the inference
bundle. `evaluate.py` owns threshold/configuration selection, reference comparisons,
frozen test evaluation, errors and benchmarks. `cli.py` dispatches commands and
imports `load_selected_config` from `pipeline.py`. `data.load_prepared_partition`
defaults to train/validation access and accepts an explicit `allow_test=True`
from the frozen evaluation stage.

Flow:

```
original CSVs + fixed manifest + prepared audit + saved retrieval selection
    -> verify original hashes, graph partitions, preparation and selection identity
    -> separate training / validation right-side indexes
    -> stable, deduplicated candidate positions
    -> existing ordered float64 feature cache
    -> attach partition gold labels in evaluation code
    -> fit on training X/y; score validation X
    -> grouped threshold sweep; fixed-grid model selection
    -> save selected models, validation results and hashes; write frozen.json last
    -> verify freeze; prepare test candidates/features; report fixed decisions

new raw query + supplied raw target catalog + custom_forest.json
    -> validate bundle and feature schema
    -> prepare attributes + catalog-local index (reusable)
    -> retrieve candidates -> twelve features -> average custom tree scores
    -> score >= saved threshold -> matches sorted by (-score, target ID)
```

`build_catalog_indexes` invokes the existing `InvertedIndex.build` only for enabled
channels. A disabled channel has an empty index object that retrieval never queries.
This avoids building unused trigrams for `word10` without changing the saved retrieval
semantics. `candidate_pairs` preserves query position and retriever order while
removing repeated target positions. `label_candidates` independently rejects duplicate
pairs. `prepare_partition` checks every gold edge's partition containment, attaches
labels after retrieval, records positive/negative/missed counts, and marks X/y/pairs
read-only before comparison. No missing positive is injected, and no arbitrary
negative is created. The run stores SHA-256 identities of all shared matrices.

The cache uses an NPZ/JSON pair; its identity also carries
the prepared-file hashes from `input_provenance`. On incompatibility the old cache
is rejected and regenerated with a reason. Labels are cheap membership checks and
are regenerated from checked partition gold, rather than trusted from a stale label
cache. Score archives store labels only for evaluation; bundles contain neither
candidate labels nor gold pairs. Tree node aggregate class counts remain the existing
learner representation.

## End-to-end metrics and selection

For a retrieved candidate label array y, score array s, all-gold count G, and threshold t:

```
accepted = s >= t
TP = sum(y[accepted])
FP = count(accepted) - TP
FN = G - TP
precision = TP / (TP + FP)
end_to_end_recall = TP / G
F1 = 2*TP / (2*TP + FP + FN)
retrieved_gold = sum(y)
candidate_recall = retrieved_gold / G
conditional_classifier_recall = TP / retrieved_gold
```

Every zero denominator returns zero. `metrics_from_counts` verifies the recall
product and ceiling. FN is decomposed into `G - retrieved_gold` and
`retrieved_gold - TP`. Retrieval reduction is `1 - C/(left_count*right_count)`;
for an empty Cartesian universe it is defined as zero. Candidate-only AP uses
sklearn's `average_precision_score`, with zero for empty/no-positive candidates.
No accuracy headline or full-universe AP claim is made.

`select_threshold` validates finite [0,1] scores and `G >= sum(y)`, sorts once in
O(C log C), and traverses equal-score groups with cumulative TP/FP counts in O(C).
All groups are complete because the decision operator is `>=`. Predict-none uses
finite `np.nextafter(1.0, np.inf)`, including for empty arrays. It compares tuples
`(F1, precision, threshold)`; no repeated scoring or repeated array-wide prediction
is performed during the sweep. The independent slow test enumerates every distinct
threshold and directly recounts labels. Another test shows an unretrieved positive
changing the best threshold, which candidate-only selection would miss.

`parameter_grid` enumerates exactly the prescribed combinations; `compare_models`
accepts only a train/validation pair and rejects shared record identities. Twelve
learned fits see the exact same X/y objects. CART uses all features. Both forests
use 32 trees, sqrt features, bootstrap, seed 42 and min split 2; sklearn adds Gini
and `n_jobs=1`. Configuration key is `(validation_F1, -max_depth, min_samples_leaf)`.
Threshold precision does not override the prescribed configuration tie rule.
The lexical model uses only columns 0 and 2 of that same candidate matrix.

The first full custom forest fit took 0.439 seconds. All four fits completed below
one second, so no negative-sampling code was added. Had fit time exceeded 120 seconds,
tuning would stop before freezing and require profiling and the bounded fallback
decision. Negative subsampling would increase positive prevalence and could change
score distributions: all learners would then need the same frozen sampled list,
while validation/test candidates and their threshold-selection denominators would
remain complete. It is not a way to improve reported F1 by removing evaluation negatives.

`compare_models` saves all thirteen trials and compares forest F1 at equal depth/leaf
settings. A gap over 0.03, or a forest below lexical, leaves a validation diagnostic
review pending. `freeze_run(..., diagnostic_review=...)` can record the investigation
before freezing; it does not retune. No trigger occurred on the real data. Known
custom/sklearn differences are detailed above: `default_rng` versus sklearn RNG,
explicit repeated bootstrap indices versus weighted distinct rows, exact selected
feature subsets versus extra-feature fallback, impurity ties, float64 versus float32.

## Freeze and persistence boundary

`tune_run` writes models trained on training data only and saves all validation
results before `freeze_run` publishes `frozen.json` with timestamp and artifact
checksums. `load_frozen_run` verifies the required artifact list, each hash, protocol
metadata, custom bundle metadata, reference parameters and dependency versions.
`evaluate_run` calls that loader before test preparation, then verifies current input
provenance against the frozen provenance. It has no training/selection call. Existing
first test results are preserved; corrections must be documented in a separate run.
Checksums detect accidental changes, not malicious rewriting of an entire run.

Custom bundles include schema/preprocessing/feature/retrieval versions, exact ordered
names/count, model parameters, structures, threshold, seed, dataset/prepared/manifest
hashes, and dependencies. `load_bundle` invokes existing learner `from_dict` validation
for malformed nodes, impossible thresholds/counts, seed structure and feature dimension,
then checks the surrounding metadata. `save_bundle` validates before writing.
`train_bundle` trains one custom configuration and selects its validation threshold;
`compare_models` applies the split boundary and compares those resulting bundles.
No train+validation refit occurs. Selected JSON round trips are checked against the
exact validation score arrays before freezing.

The sklearn evaluation reference is a local standard-library pickle with a frozen
checksum and exact Python/NumPy/sklearn version check. This is a trusted local
experiment artifact, not a general model interchange format; external pickles are
not accepted by the runtime JSON loader. The custom application has no sklearn/scipy
import. `ModelBundle.score` uses the custom learners' probability averaging. Runtime
`match_record` accepts raw records, an optional reusable `PreparedCatalog`, and a
bundle or JSON path; it requires the custom forest. It validates metadata without
serializing every tree per query, rebuilds catalog-dependent IDF when needed, and
never uses a Dataset, split membership, gold mappings or labels.

## Benchmark and error analysis boundaries

`benchmark_run` measures verified preparation, enabled-index build, candidate retrieval,
uncached feature extraction, saved training fits, and batch scoring separately. The
same custom forest and threshold score indexed and exhaustive queries; both delegate
the final feature/scoring/output work to `pipeline.score_pairs`. The first raw-ID-ordered
training queries form a fixed subset. A measured ten-query exhaustive pilot precedes
the requested twenty-query comparison; growth above twenty is permitted only when the
pilot is at most two seconds. The real pilot took 2.623 seconds, and the requested
20-query size was retained. No 100-query or full-catalog timing is extrapolated.

`_query_timings` performs one complete warmup per method and five timed passes,
alternating which method runs first. It retains every per-query sample and each pass
total; NumPy linear percentiles describe those samples. Warm query time includes query
normalization, retrieval for indexed matching, uncached features, scoring, thresholding,
and sorted match output. It excludes target preparation, index construction, model load,
and labels. Cold feature time and verified cache load time are reported independently.

Memory reports array `nbytes`, node counts, actual serialized file sizes and a separate
tracemalloc observation for model load, catalog preparation/index, and one query.
Tracemalloc instrumentation is absent from timing passes and its peak is not process
RSS. The custom forest JSON is indented and includes aggregate node counts, so its size
is not directly comparable to the binary sklearn pickle as an algorithm-memory measure.

`error_examples` exports bounded false positives, retrieved rejections, and retrieval
misses. Raw titles, IDs, gold label, threshold and twelve numerical features are retained.
A diagnostic score is computed for a bounded missed pair only after evaluation; the
pair remains absent from classifier decisions and remains an FN. Model conflicts and
missing fields are observable flags, not established causal explanations. The actual
color/fuel-variant mistakes and word-rank 11/27 retrieval misses are documented in
[results.md](results.md#end-to-end-error-accounting). No test-driven algorithm or
parameter changes followed those examples.

## Integration reading order and actual product trace

Study these exact functions in order:

1. `data.verify_original`, `verify_manifest`, `load_prepared_partition`: explain the
   fixed hashes, component partitions, no-leakage check and frozen-stage test opt-in.
2. `data.prepare_record`, `normalize.normalize_text`, `model_tokens`, `parse_price`:
   show how raw attributes become normalized strings, sets and optional prices.
3. `pipeline.build_catalog_indexes`, `candidate_pairs`; then
   `retrieval.InvertedIndex.build`, `query`, `BoundedTopK.add`, `retrieve_candidates`:
   trace postings, IDF, top-K, local target positions and stable candidate order.
4. `features.pair_features`, `feature_matrix`, `cached_feature_matrix`; then
   `pipeline.label_candidates`, `prepare_partition`: trace twelve columns, exact
   Levenshtein, cache identity and the separate label boundary.
5. `tree._best_split`, `DecisionTreeClassifier._build_tree`; then
   `forest.RandomForestClassifier.fit`, `predict_proba`: explain sorting/prefix counts,
   recursive splits, bootstrap occurrences, per-node random features and averaging.
6. `evaluate.parameter_grid`, `compare_models`, `configuration_key`: show every learner
   receiving shared training X/y and only validation selecting depth/leaf settings.
7. `evaluate.select_threshold`, `metrics_from_counts`, `classification_metrics`:
   work through equal-score groups, finite predict-none, tie rules and retrieval FNs.
8. `evaluate.freeze_run`, `load_frozen_run`, `evaluate_run`; then
   `pipeline.save_bundle`, `load_bundle`: explain frozen train-only models and schema checks.
9. `pipeline.prepare_catalog`, `match_record`, `score_pairs`: trace a gold-free query
   to many-to-many accepted matches and deterministic output order.
10. `evaluate.benchmark_run`, `_query_timings`, `exhaustive_match`, `tree_node_count`,
    `error_examples`: connect measured cost and memory to honest quality/error reports.

Consequences of these choices:

- A retrieval miss has no feature row in the classifier's input. Even an excellent
  hypothetical forest score cannot recover it; the test LG washer demonstrates this.
- Most candidate rows are negative. An all-negative classifier would look about 90%
  accurate while finding zero matches. Precision measures accepted-pair reliability,
  recall measures recovered gold, and F1 balances them while retaining retrieval misses.
- No negatives were sampled here. Sampling would raise training prevalence and alter
  what a score means; validation/test prevalence must stay intact, and thresholds must
  be selected again on that complete validation pool.
- Lexical and both custom learners and sklearn use identical candidates/features.
  The reference checks practical quality without requiring identical trees or scores.
- Exact pair feature extraction, particularly Python Levenshtein, dominates this small
  workload. The training matrix took 2.888 s to build versus 0.561 s to fit the selected
  forest; test features took 0.920 s versus 0.084 s for custom forest batch scoring.

Actual training pair (saved in `runs/main/product_trace.json`):

- Raw Abt `10102`: **Omnimount Wall Speaker Mount - 20WLWH**, description beginning
  “Omnimount Wall Speaker Mount - 20WLWH/ Stainless Steel Shafts…”, price `$40.00`.
- Raw Buy `90106210`: **OmniMount Stainless Steel Speaker Mount - 20.0 WALL-W**,
  description `Stainless Steel - 20 lb`, missing price.
- Normalized titles: `omnimount wall speaker mount 20wlwh` and
  `omnimount stainless steel speaker mount 20 0 wall w`.
- Model sets: `{'20wlwh'}` and empty. Price becomes `40.0` and `None`.
- The training catalog has 655 targets. `word10` retrieves ten candidates; this target
  ranks **1st**, with binary TF-IDF cosine **0.5330460029704667**. Retrieval score is
  metadata and never becomes a feature.

The twelve features, in their unchanged order, are:

| Index | Feature | Value |
|---|---|---:|
| 0 | title_token_jaccard | 0.4 |
| 1 | title_token_containment | 0.8 |
| 2 | title_trigram_jaccard | 0.4897959184 |
| 3 | title_edit_similarity | 0.6274509804 |
| 4 | model_token_jaccard | 0 |
| 5 | model_token_conflict | 0 |
| 6 | model_tokens_both_present | 0 |
| 7 | description_token_jaccard | 0.1071428571 |
| 8 | descriptions_both_present | 1 |
| 9 | relative_price_difference | 0 |
| 10 | prices_both_present | 0 |
| 11 | numeric_token_jaccard | 0 |

The saved 32-tree custom forest averages its reached leaf fractions to
**0.6510464234575638**. The validation-selected threshold is
**0.2347583241835786**. Since score >= threshold, the pair is **accepted**.
The original mapping also labels it positive, checked only for this explanatory
training example after scoring; runtime does not consult that label. Missing price
creates `(difference, both_present) = (0, 0)`, not evidence of equal known prices.
This trace is an actual application example, not a new held-out quality measurement.

## Profiling and preservation

`tools/profile_workload.py` provides a standalone driver around the existing functions;
it is not a new benchmark implementation. It verifies the frozen run, uses only
train/validation workloads, writes to a new output directory, profiles with
`cProfile`/`pstats`, and records independent `perf_counter` and `tracemalloc` passes.
The existing `evaluate.benchmark_run` still owns indexed/exhaustive timing.

On 6,426 training candidates, exact Levenshtein accounted for 97.02% of original
profiled feature time. Normalization, token-set construction and price parsing were
already reused via `PreparedRecord`, with no calls during feature extraction.
Replacing per-cell `min` with explicit comparisons reduced paired median uncached
feature time from 1.464323292 to 0.843670458 s; validation fell from 0.492912000
to 0.282651708 s. The exact two-row algorithm and all twelve numerical definitions
remain intact. Retrieval, CART training and forest inference retain their existing
implementations because their profiles did not justify extra complexity.

Verification regenerates preparation and features separately, checks candidate
order and complete feature matrices against the original saved arrays, and compares
all four validation/test score arrays and accepted pair sets exactly. Refitting the
selected custom configurations reproduces every tree structure. Independent set
counts and grouped candidate-only AP calculations confirm the reported metrics.
No threshold/configuration is changed; no evaluation is invalidated.

The original CSVs, manifest, prepared files and all previous experiment files are
protected by before/after checksums. A copy of the frozen run is used for CLI test
evaluation and benchmark replay in a separate output directory; originals remain
under local `runs/main`. Public report copies and labeled extracts are under
`docs/evidence` so a clean checkout can inspect the measured claims. Full model
bundles, raw data and generated matrices remain local/ignored outputs.

The original benchmark's 61.84393612475235× indexed/exhaustive ratio remains the
original result. The optimized replay reports 59.965945371042665× after both
paths benefit from faster feature computation. Cross-run absolute times on unchanged code also vary;
only the alternating paired feature experiment attributes a speedup to the edit.
Neither comparison measures cold disk caches or production throughput.

Existing oracle, memory and integration tests already cover the changed recurrence,
so no redundant tests were added. All 767 pass. Fresh-process real-catalog CLI
inference reproduces the original two accepted matches with sklearn/scipy imports
and gold-loading functions forbidden. Full methodology, allocation scopes,
reproduction commands and remaining limitations are in [results.md](results.md).
