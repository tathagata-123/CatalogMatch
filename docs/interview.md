# CatalogMatch technical study notes

These notes explain the algorithms, their invariants and the measured trade-offs.
See [results](results.md) for evidence and [design](design.md) for full contracts.

## Project overview

**Problem and scale.** CatalogMatch finds listings that refer to
the same product across two catalogs despite differences in names, descriptions
and prices. Comparing A products with all B targets costs A×B pair evaluations.
That becomes expensive when a comparison includes exact edit distance and a forest.
On the training partition, exhaustive comparison would examine 424,440 pairs;
retrieval reduces that to 6,426. The matcher allows multiple matches because the benchmark
contains one-to-many entities; it does not force a one-to-one assignment.

**Candidate retrieval.** The pipeline prepares each product once and builds
an inverted index mapping each distinct title word to target record positions.
Postings contain each position once. A query visits only postings for shared words,
accumulates binary TF-IDF dot products, and normalizes them to cosine similarity.
A custom bounded min-heap keeps the best ten candidates. Its root is the worst
retained result; ties prefer smaller target positions. Character trigrams and
channel unions are also available. The validation rule chose `word10` as the
lowest candidate cost reaching 98% recall. Retrieval uses no gold labels.

**Features and learners.** Each candidate becomes twelve numerical
attribute comparisons: title word/trigram overlap, exact normalized edit similarity,
model-token agreement/conflict, description overlap/presence, price difference/presence,
and numeric-token overlap. IDs and split/gold fields never become features. CART uses stable sorting and cumulative positive counts to score legal boundaries,
then partitions only once at the winner. A forest trains 32 custom trees on bootstrap
indices, retaining duplicates, and draws a fresh feature subset at each eligible
node. Prediction averages leaf fractions, not hard class votes. NumPy provides
arrays, sorting and RNG; the split search and tree construction are custom code.

**Evaluation.** Gold-positive connected components stay entirely
within train, validation or test, preventing a product endpoint from leaking across
partitions. Every learner receives the same training rows and feature matrix.
Validation selects from a fixed depth/leaf grid and selects thresholds using
end-to-end F1. The models are frozen before test scoring. Retrieval misses remain
false negatives, while average precision is explicitly candidate-only. On test,
`word10` retrieves 217/219 gold pairs. The custom forest gets 194 TP, 29 FP and
25 FN: two retrieval misses and 23 classifier rejections. Its F1 is 0.8778 versus
0.6978 lexical, 0.8473 CART and 0.8773 sklearn forest. That tiny forest difference
is not a claim of superiority.

**Engineering.** Model JSON validates schema, feature
order, structures and provenance, and gold-free inference needs only a query, target
catalog and custom bundle. Independent full-sort, cosine, full-table DP and brute-force
split oracles test the custom algorithms; the full suite has 767 passing tests.
Profiling showed about 97% of original feature-generation time inside Levenshtein.
Replacing one `min` call per DP cell with explicit integer comparisons preserved the
exact recurrence and made feature extraction about 1.74× faster on identical inputs.
Saved predictions remained exactly equal. The original bounded benchmark measured
61.84× indexed speedup over exhaustive matching; both computed features uncached.
The project demonstrates correctness and measured trade-offs on a small old dataset,
not production throughput or semantic language understanding.

## Algorithms, invariants and complexity

Notation: V vertices, E positive edges, N targets, Z distinct term-record associations,
W vocabulary size, q distinct query terms, P postings visited, U encountered targets,
K retrieval limit, C candidate rows, n training occurrences, d columns, s selected
columns, D depth and T trees. Hash operations below use expected constant-time
lookup, not an adversarial guarantee. String hashing/normalization also costs input
length when not already cached.

| Algorithm and purpose | Main invariant and implementation | Time | Auxiliary/storage space |
|---|---|---|---|
| Connected components: prevent entity leakage | Namespaced vertices; only positive edges; DFS marks on push, then assigns whole canonical components | Graph/DFS O(V+E), canonical sorting up to O(V log V) | O(V+E) |
| Inverted index construction: reusable target lookup | Deduplicate each document's terms; append positions in ascending order; df is posting length | Expected O(Z+N) on prepared distinct terms | Index O(Z+W+N), temporary documents O(Z+N) |
| Cosine accumulation: score overlap candidates | All query terms enter norm, including unseen terms; only encountered targets enter accumulator | O(q log q + P + U), before top-K | O(q+U) |
| Binary top-K heap: bound selection cost | Parent key ≤ child keys; root is worst of best K seen; key `(score,-position)` | O(U log max(2,K)) + O(K log K) final sort | O(K); accumulator still O(U) |
| Exact Levenshtein: compare ordered title characters | Above/diagonal from previous row, left from current row | O(mn) for nonempty strings of lengths m,n | O(min(m,n)) |
| CART best split: minimize weighted Gini | Prefix counts include every occurrence; only distinct-value boundaries with legal leaf sizes compete | O(s m log m) at a node of m occurrences | O(m) arrays per feature, plus feature indices |
| Recursive CART construction: build a classifier | Children strictly smaller; stored counts equal routed occurrences; provisional leaves satisfy stopping rules | Safe capped bound O(s n D log n), plus O(nd+n) validation | O(nD) conservative active-index bound; O(n) balanced; O(D) stack; stored nodes O(n) |
| Forest bootstrapping: vary training samples | Exactly n draws with replacement per tree; duplicates remain occurrences | O(Tn) sampling plus tree fits | One O(n) bootstrap at a time; shared X/y and all fitted nodes |
| Per-node feature sampling: diversify splits | Fresh subset without replacement at each eligible node; exactly s inspected | Feature selection bounded by O(d+s log s) per node; RNG implementation determines exact cost | O(d) safe bound; no new feature matrix |
| Prediction/thresholding: turn candidates into matches | Route `<=` left; average T leaf fractions; accept `>=` saved threshold | O(C T(D+1) + Cd); sort A accepted matches O(A log A) | O(C) scoring vectors; O(A) output, excluding existing features/model |

Tree training sorts anew at each node, so a presorted-tree bound does not apply.
The forest trains sequentially and stores O(sum of tree nodes); it does not retain
bootstrap arrays or a C×T prediction matrix. The complete feature matrix is O(12C)
float64 storage, plus pairs and labels. An index can still encounter nearly all
N targets for common terms. A heap bounds selection storage, not the score map.

## Functions to reconstruct

The following functions contain the core implementations:

| Priority | File | Functions |
|---|---|---|
| Retrieval | [retrieval.py](../src/catalogmatch/retrieval.py) | `InvertedIndex.build`, `InvertedIndex.query`, `bounded_top_k`, `retrieve_candidates` |
| Heap | [retrieval.py](../src/catalogmatch/retrieval.py) | `BoundedTopK.add`, `sift_up`, `sift_down`, `best_first` |
| Dynamic programming | [similarity.py](../src/catalogmatch/similarity.py) | `levenshtein`, `edit_similarity` |
| CART | [tree.py](../src/catalogmatch/tree.py) | `_gini`, `_threshold`, `_best_split`, `DecisionTreeClassifier._build_tree`, `_select_features`, `_positive_probabilities` |
| Forest | [forest.py](../src/catalogmatch/forest.py) | `RandomForestClassifier.fit`, `predict_proba` |
| Integration | [pipeline.py](../src/catalogmatch/pipeline.py), [evaluate.py](../src/catalogmatch/evaluate.py) | `candidate_pairs`, `prepare_partition`, `match_record`, `score_pairs`, `select_threshold`, `classification_metrics` |
| Split | [data.py](../src/catalogmatch/data.py) | `build_positive_graph`, `connected_components`, `split_components`, `validate_partitions` |

### Heap and cosine

For a min-heap at index i, parent is `(i-1)//2`, children `2*i+1` and `2*i+2`.
Sift-up swaps with the parent while the child is smaller. Sift-down chooses the
smaller child and swaps while the parent is larger. When full, a new key must be
strictly larger than the root to replace it. Negating positions makes a large
position worse when scores tie. Sort only the retained heap for final output.

In `query`, sort distinct query terms to fix floating-point accumulation order.
For every shared term, add `idf(term)**2` to each posting target. Divide each
encountered target by query norm × precomputed target norm. Unknown query terms
have df=0, contribute to query norm, and have no postings. The heap operates on
these normalized scores. A union selects each channel independently, deduplicates
positions, and performs **no second truncation**.

### Levenshtein: table first, then two rows

Let `dp[i][j]` be the minimum edits turning `a[:i]` into `b[:j]`.
Base cases are `dp[i][0]=i` and `dp[0][j]=j`. Consider the final operation:

```text
delete a[i-1]:    dp[i-1][j] + 1
insert b[j-1]:    dp[i][j-1] + 1
match/substitute: dp[i-1][j-1] + (0 if a[i-1] == b[j-1] else 1)
dp[i][j] = minimum of those three costs
```

A full table costs O(mn) memory. Only the previous row and the completed current
row prefix are needed, so two reusable rows suffice. Put the shorter string in
columns, initialize `previous=range(shorter_length+1)`, set `current[0]=i`, fill
left to right, then swap buffers. Empty-string distance is the other length.
The explicit integer comparisons compute exactly the same minimum; they change no DP
state or operation costs. There is no approximation or bounded-distance cutoff.

`edit_similarity` is `1-distance/max(lengths)` for two nonempty normalized titles.
It returns zero if either title is empty: two absent fields are not positive evidence.
The primitive does not normalize text itself.

### CART: sorting, prefix counts, thresholds, partitioning

For binary labels, Gini is `2p(1-p)`. At each inspected feature:

1. Gather values for the node's occurrence indices and stably sort them.
2. Gather labels in that order and compute cumulative positive counts.
3. Boundary k is legal only if adjacent values differ and both k and m-k satisfy
   the minimum leaf size. Left positives are prefix[k-1]; right positives are
   total minus left positives.
4. Minimize `(k/m)*Gini(left) + ((m-k)/m)*Gini(right)`. Exact ties prefer the lower
   feature index, then the earliest boundary. No per-boundary row partition is built.
5. Choose a threshold between the winning adjacent values. `_threshold` uses
   `lo/2+hi/2`, falling back to lo if rounding would violate `lo <= t < hi`.
6. Partition indices once with `<= t`. Check that its left size equals the scored
   count before recursing. Repeated row indices must survive this operation.

Example: unsorted `(x,y)=[(4,1),(1,0),(3,1),(2,0)]`. Sorting gives values
`[1,2,3,4]`, labels `[0,0,1,1]`, prefixes `[0,0,1,2]`. At threshold 2.5, both
children are pure, weighted Gini is zero and gain is 0.5. The parent becomes an
internal node; the children return probabilities zero and one.

`_build_tree` first makes a provisional leaf holding sample/positive counts and
positive fraction. It stops at purity, depth limit, too few split/leaf occurrences,
or no legal split in the selected columns. A legal **zero-gain** split is allowed:
XOR needs it before informative child splits. Termination follows from strictly
smaller nonempty child populations plus the configured depth limit. Unbounded
depth remains subject to Python recursion limits.

### Random Forest: duplicates, fresh features and averages

A master `default_rng(42)` produces one seed per tree. Each tree's generator draws
n row indices with replacement, then continues its state for feature sampling.
A bootstrap `[0,0,0,1,1]` contains five occurrences, not two observations for sample
constraints: duplicates affect impurity, leaf fractions and minimum leaf sizes.
Physically expanding rows is a useful independent oracle; merely using sample
weights can give different minimum-sample behavior in another implementation.

At every eligible node, choose a new subset without replacement; with twelve
features, `sqrt` gives three. Only that subset is searched. A constant sampled
subset can leave the node unsplit; there is no fallback to extra columns.

For inference, traverse each tree iteratively and add its reached leaf fraction
into one length-C accumulator, then divide by T. Fractions `[0.6,0.6,0.0]` average
to 0.4, despite two positive hard votes. The built-in class prediction is zero
at exactly 0.5; matching instead uses the separately frozen validation threshold.

## Engineering trade-offs

| Choice | Reason and cost |
|---|---|
| Index vs exhaustive | Test candidates shrink from 47,304 to 2,053, but two gold pairs are lost before classification. The original bounded timing comparison measured 61.84× speedup. |
| Words vs trigrams | Validation `word10` retrieves 217/220 at 2,078 pairs/10,176 postings; `char10` retrieves 220/220 at 2,170 pairs/108,251 postings. Trigrams tolerate some spelling variation at higher cost. |
| K and recall | Larger K can rescue low-ranked overlap matches, at more feature work; it cannot recover zero-overlap pairs in that channel. The selected K stays frozen. |
| Heap vs full sorting | O(U log K) selection is useful for small K; full sorting is simpler but costs O(U log U). Independent full-sort tests validate the heap. |
| Exact DP vs overlap | DP captures ordered character edits but dominates feature cost. Sets lose order and are cheaper. Profiling justified a readable inner-loop improvement; it did not justify dropping the feature. |
| CART vs forest | CART test F1 0.8473; custom forest 0.8778. Averaging diverse trees costs more training, storage and traversal. This is one measured dataset, not a universal rule. |
| Depth vs overfitting | Greater depth can fit small peculiar subsets; minimum leaf constraints limit that. Validation chooses within the fixed grid, and test does not choose depth. |
| Custom vs sklearn | The custom learner exposes split/count/RNG invariants. Sklearn's compiled scoring is faster. Different RNG/bootstrap/precision/tie semantics mean equal seeds do not imply equal trees. |
| Cache vs recomputation | A compatible matrix avoids exact DP on repeated fits. Identity includes inputs, pair order, schema and retrieval settings; stale data must be regenerated. Benchmark query paths both compute uncached features. |
| Precision vs recall | Lower thresholds usually accept more pairs: recover positives but risk extra false positives. The saved threshold maximizes validation end-to-end F1, not candidate accuracy. |
| Retrieval vs classifier error | Retrieval misses have no classifier row. The custom forest's test FN count is 2 misses + 23 rejections; changing a decision threshold cannot repair missing candidates. |

## Technical questions

1. **Why split connected components instead of labeled pairs?** Pairs can share
   endpoints or be linked by positive chains. Whole components prevent the same
   entity from appearing in different partitions. Negative candidates do not define entities.
2. **Why deduplicate terms before indexing?** Binary TF counts a term once per
   document. Repeated postings would inflate df and dot products and break that model.
3. **Why do unseen query words affect cosine?** They still contribute to query
   length. Excluding their weights would inflate similarity to documents matching
   only a small known portion of the query.
4. **What is the heap's invariant?** Parent keys are no greater than their children,
   so the root is the worst retained candidate. Retaining the best K after each
   insertion follows by replacing that root only with a strictly better key.
5. **Why can DP discard old rows?** Every cell depends only on above, diagonal and
   left. Older rows cannot affect a future cell once the previous row is complete.
6. **How does CART avoid quadratic threshold recounts?** Sorting makes every
   legal split a prefix/suffix; cumulative positives score each boundary in constant
   arithmetic work after sorting. Only the winning split partitions row indices.
7. **Why carefully construct floating-point thresholds?** A rounded midpoint can
   equal the upper adjacent value. Falling back to the lower value ensures `<=`
   reproduces exactly the partition whose impurity was scored.
8. **Why allow zero information gain?** XOR has zero-gain first splits followed by
   informative children. Rejecting every zero-gain split would prevent its solution.
9. **What breaks if you deduplicate bootstrap indices?** Frequencies, class counts,
   probabilities and minimum-sample constraints change. The resulting tree would
   no longer be trained on the sampled multiset.
10. **Why average probabilities instead of hard votes?** Leaf fractions retain
    within-leaf class mixture; hard votes discard that evidence. The two rules can
    disagree, as the `[0.6,0.6,0.0]` example shows.
11. **How do you know an optimization is behavior-preserving?** Independent
    algorithmic oracles plus exact candidate/order, feature, saved-score, decision
    and final-pair comparisons. A better F1 would not excuse silently changed semantics.
12. **What did profiling change?** It identified DP as about 97% of feature time.
    Explicit minimum comparisons improved paired feature throughput about 1.74×;
    normalization was already reused. Small retrieval/CART costs did not justify a rewrite.

## One complete actual product example

This is the existing **training** example, not a new quality measurement. Raw values,
features and score come from [the saved product trace](evidence/evaluation/product_trace.json).
The [retrieval and edit-distance trace details](evidence/performance/trace_details.json) add measured lookup/DP details
and verify the original retrieval score exactly.

| Field | Abt `10102` | Buy `90106210` |
|---|---|---|
| Raw title | Omnimount Wall Speaker Mount - 20WLWH | OmniMount Stainless Steel Speaker Mount - 20.0 WALL-W |
| Raw description | Omnimount Wall Speaker Mount - 20WLWH/ Stainless Steel Shafts And All Necessary Hardware Included/ Supports Speakers Up To 20 lbs./ Sold as each / White Finish (Photo Showing Black) | Stainless Steel - 20 lb |
| Raw price | `$40.00` | empty |
| Manufacturer | absent in Abt schema | OMNIMOUNT SYSTEMS, INC |
| Normalized title | `omnimount wall speaker mount 20wlwh` | `omnimount stainless steel speaker mount 20 0 wall w` |
| Word set | `{20wlwh, mount, omnimount, speaker, wall}` | `{0, 20, mount, omnimount, speaker, stainless, steel, w, wall}` |
| Model set | `{20wlwh}` | empty |
| Parsed price | 40.0 | None |

Normalization decodes entities, applies NFKC and casefolding, extracts ASCII
alphanumeric words, and joins them with spaces. Description markup is stripped
before that process. The raw fields are preserved alongside prepared attributes.
Manufacturer and IDs do not enter the twelve features.

The training catalog has 655 targets; this Buy record is at position 629 in raw-ID
order. The query looks up these actual word-index entries:

| Query term | df / posting length | IDF | Position 629 in posting? |
|---|---:|---:|---|
| `20wlwh` | 0 | 7.486160788944089 | No |
| `mount` | 25 | 4.228064250922607 | Yes |
| `omnimount` | 5 | 5.694401319716034 | Yes |
| `speaker` | 9 | 5.183575695950044 | Yes |
| `wall` | 16 | 4.652947444887873 | Yes |

Each of the four shared postings adds its squared IDF to position 629's accumulator.
The unseen model word contributes only to query norm. Query norm is
12.444465194550146; target norm is 14.897493228091097. Cosine is
**0.5330460029704667**. The custom heap retains ten candidates and this target
ranks first; second is Buy `90140203` at 0.3466900184813965. All ten exact entries
are in the trace details. Retrieval score is metadata and is not a predictive column.

Feature computation uses the already prepared strings and sets. Title lengths are
35 and 51; exact edit distance is 19. Trigram sets have sizes 29 and 44,
intersection 24 and union 49. The full unchanged vector is:

| Index | Feature | Actual saved value |
|---|---|---:|
| 0 | title_token_jaccard | 0.4 |
| 1 | title_token_containment | 0.8 |
| 2 | title_trigram_jaccard | 0.4897959183673469 |
| 3 | title_edit_similarity | 0.6274509803921569 |
| 4 | model_token_jaccard | 0.0 |
| 5 | model_token_conflict | 0.0 |
| 6 | model_tokens_both_present | 0.0 |
| 7 | description_token_jaccard | 0.10714285714285714 |
| 8 | descriptions_both_present | 1.0 |
| 9 | relative_price_difference | 0.0 |
| 10 | prices_both_present | 0.0 |
| 11 | numeric_token_jaccard | 0.0 |

The title sets share four of ten union words, giving 0.4 Jaccard and 4/5=0.8
containment. Edit similarity is `1-19/51`. Missing Buy price gives `(difference,
presence)=(0,0)`, not evidence that two known prices are equal. Missing Buy model
also means conflict is zero: unknown does not assert agreement or disagreement.

The saved 32-tree forest routes this vector and averages its leaf fractions to
**0.6510464234575638**. The validation-selected threshold is
**0.2347583241835786**, so the pair is accepted. The original mapping labels it
positive, consulted only for this explanatory training trace after scoring.
Changing the target catalog can change IDF/ranking: this trace uses 655 training
targets, while the separate gold-free CLI check uses all 1,092 targets.

## Reconstruction exercises

Use separate scratch files for these exercises and compare with the independent
oracles below. Keep frozen experiment artifacts unchanged.

1. **Top-K heap.** Implement `sift_up`, `sift_down`, insertion/replacement and sorted
   output. Trace tied scores at K=2 and explain why a larger position loses a tie.
   Validate with `test_heap_invariants_and_shuffled_order`, `test_sift_up_and_sift_down`,
   `test_root_replacement_requires_strictly_better_key` and
   `test_top_k_sorts_only_retained_entries` in [test_retrieval.py](../tests/test_retrieval.py).
2. **Two-row Levenshtein.** First implement the full table, then remove all but two
   rows. Trace `kitten → sitting`, empty strings and unequal lengths. Compare with
   `full_table_levenshtein`, `test_levenshtein_randomized_oracle` and
   `test_levenshtein_memory_is_two_short_rows` in [test_similarity.py](../tests/test_similarity.py).
3. **Sorting/prefix Gini search.** Implement a slow partition-and-recount oracle,
   then the sorted version. Include duplicated values, tied features, repeated
   indices and adjacent floats. Use `brute_force_split`,
   `test_sorted_split_matches_independent_oracle`,
   `test_threshold_reproduces_scored_partition_at_float_boundaries` in
   [test_tree.py](../tests/test_tree.py).
4. **Recursive construction.** Draw the full XOR tree, label every node with sample
   count/positive fraction, and explain each return. Implement a provisional leaf,
   stopping conditions and winner-only partition. Use `assert_partition_tree`,
   `test_zero_gain_xor_and_depth_limit` and
   `test_bootstrap_multiplicity_affects_leaf_and_split_constraints` in the same file.
5. **Forest training and averaging.** Replay one tree's seed, bootstrap and first
   feature subset. Explain why the generator continues after sampling rows.
   Implement sequential fit and probability averaging. Use
   `test_bootstrap_indices_seeds_and_continuing_rng`,
   `test_feature_subsets_are_fresh_sorted_and_without_replacement`,
   `test_probability_averaging_and_difference_from_hard_voting` and
   `test_forest_json_roundtrip_exact` in [test_forest.py](../tests/test_forest.py).

Run the relevant test file with `python -m pytest tests/test_tree.py -q` (or the
corresponding file). A scratch implementation needs the same interface to use
these oracle checks.

## Dependencies and limitations

Custom components are the graph traversal/split logic, inverted index and score
accumulation, binary heap, exact DP, feature definitions, CART split/build/prediction,
forest sampling/averaging, and pipeline/metric orchestration. Standard libraries
handle CSV, JSON, hashing and HTML parsing. NumPy handles storage, vector arithmetic,
sorting, cumulative sums and RNG. Sklearn supplies the independent reference learner
and candidate-only AP during evaluation; pytest supplies test execution.

Embeddings, transformers and neural networks are outside this project's intended
DSA/classical-ML scope. Their absence makes algorithmic mechanics visible; it does
not prove they would be worse. The experiments do not establish state-of-the-art
matching, production-scale throughput, calibrated identity probabilities or general
superiority over sklearn.

The dataset is small and old, every source record has a known matching partner,
prices/descriptions are frequently missing, and lexical/model-number heuristics
confuse some variants. Exact DP remains expensive. The saved experiments measure
one fixed dataset and evaluation protocol.
