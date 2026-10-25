"""Binary TF-IDF retrieval with postings, a bounded heap, and validation metrics."""

from collections.abc import Iterable
from dataclasses import dataclass
import math
from time import perf_counter

from catalogmatch.data import PreparedRecord


RETRIEVAL_VERSION = "binary_tfidf_postings_heap_v1"
RECALL_TARGET = 0.98


def _validate_k(k: int) -> None:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer (at least 1)")


class BoundedTopK:
    """List-backed min-heap: the root is the worst retained (score, -position)."""

    def __init__(self, k: int) -> None:
        _validate_k(k)
        self.k = k
        self.entries: list[tuple[float, int]] = []

    def sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if self.entries[parent] <= self.entries[index]:
                break
            self.entries[parent], self.entries[index] = self.entries[index], self.entries[parent]
            index = parent

    def sift_down(self, index: int) -> None:
        size = len(self.entries)
        while 2 * index + 1 < size:
            child = 2 * index + 1
            right = child + 1
            if right < size and self.entries[right] < self.entries[child]:
                child = right
            if self.entries[index] <= self.entries[child]:
                break
            self.entries[index], self.entries[child] = self.entries[child], self.entries[index]
            index = child

    def add(self, record_position: int, score: float) -> None:
        # Each scored position is offered once by bounded_top_k.
        entry = (score, -record_position)
        if len(self.entries) < self.k:
            self.entries.append(entry)
            self.sift_up(len(self.entries) - 1)
        elif entry > self.entries[0]:
            self.entries[0] = entry
            self.sift_down(0)

    def best_first(self) -> list[tuple[int, float]]:
        # Only these retained entries are sorted, never the full score map.
        return [(-negative_position, score)
                for score, negative_position in sorted(self.entries, reverse=True)]


def bounded_top_k(score_by_id: dict[int, float], k: int) -> list[tuple[int, float]]:
    """Return (position, score) pairs, descending score then ascending position."""
    heap = BoundedTopK(k)
    for record_position, score in score_by_id.items():
        heap.add(record_position, score)
    return heap.best_first()


@dataclass
class RetrievalStats:
    """Additive counters; two-channel counts include work in both indexes."""

    postings_visited: int = 0
    distinct_candidates: int = 0


class InvertedIndex:
    def __init__(self) -> None:
        self.postings: dict[str, list[int]] = {}
        self.df: dict[str, int] = {}
        self.idf: dict[str, float] = {}
        self.n_records = 0
        self.norms: list[float] = []

    @classmethod
    def build(cls, terms_by_id: Iterable[Iterable[str]]) -> "InvertedIndex":
        """Enumerate the prepared target sequence; its offsets are the record IDs."""
        index = cls()
        documents = []
        for record_position, terms in enumerate(terms_by_id):
            distinct_terms = set(terms)
            documents.append(distinct_terms)
            for term in distinct_terms:
                # Ascending position traversal constructs sorted, unique postings.
                index.postings.setdefault(term, []).append(record_position)
        index.n_records = len(documents)
        for term, positions in index.postings.items():
            frequency = len(positions)
            index.df[term] = frequency
            index.idf[term] = 1.0 + math.log((index.n_records + 1) / (frequency + 1))
        for terms in documents:
            # fsum makes positive-weight norms stable across set iteration orders,
            # without adding a per-document sorting cost to index construction.
            squared_norm = math.fsum(index.idf[term] ** 2 for term in terms)
            index.norms.append(math.sqrt(squared_norm))
        return index

    def query(self, query_terms: Iterable[str], k: int,
              stats: RetrievalStats | None = None) -> list[tuple[int, float]]:
        """Visit only shared-term postings; retain at most k positive-overlap hits."""
        _validate_k(k)
        terms = sorted(set(query_terms))
        if not terms or not self.n_records:
            return []
        unseen_idf = 1.0 + math.log(self.n_records + 1)
        query_weights = [self.idf.get(term, unseen_idf) ** 2 for term in terms]
        query_norm = math.sqrt(math.fsum(query_weights))
        score_by_id: dict[int, float] = {}
        postings_visited = 0
        for term, squared_weight in zip(terms, query_weights):
            positions = self.postings.get(term, [])
            postings_visited += len(positions)
            for record_position in positions:
                score_by_id[record_position] = score_by_id.get(record_position, 0.0) + squared_weight
        if stats is not None:
            stats.postings_visited += postings_visited
            stats.distinct_candidates += len(score_by_id)
        for record_position, dot_product in score_by_id.items():
            # A posting implies a nonempty document and therefore a positive norm.
            score_by_id[record_position] = dot_product / (query_norm * self.norms[record_position])
        return bounded_top_k(score_by_id, k)


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    word_k: int | None = None
    gram_k: int | None = None

    def __post_init__(self) -> None:
        if self.word_k is None and self.gram_k is None:
            raise ValueError("A retrieval configuration must enable at least one channel")
        for k in (self.word_k, self.gram_k):
            if k is not None:
                _validate_k(k)


RETRIEVAL_CONFIGS = (
    RetrievalConfig("word10", word_k=10),
    RetrievalConfig("word25", word_k=25),
    RetrievalConfig("word50", word_k=50),
    RetrievalConfig("char10", gram_k=10),
    RetrievalConfig("char25", gram_k=25),
    RetrievalConfig("union10+10", word_k=10, gram_k=10),
    RetrievalConfig("union25+25", word_k=25, gram_k=25),
)


@dataclass(frozen=True)
class Candidate:
    target_position: int
    word_score: float | None
    gram_score: float | None
    channels: tuple[str, ...]


def retrieve_candidates(prepared_left: PreparedRecord, word_index: InvertedIndex,
                        gram_index: InvertedIndex, config: RetrievalConfig,
                        stats: RetrievalStats | None = None) -> list[Candidate]:
    """Select independently per channel; a union has no second truncation."""
    words = []
    grams = []
    if config.word_k is not None:
        words = word_index.query(prepared_left.title_word_tokens, config.word_k, stats)
    if config.gram_k is not None:
        grams = gram_index.query(prepared_left.title_trigrams, config.gram_k, stats)
    if config.gram_k is None:
        return [Candidate(position, score, None, ("word",)) for position, score in words]
    if config.word_k is None:
        return [Candidate(position, None, score, ("char",)) for position, score in grams]
    by_position = {}
    for position, score in words:
        by_position[position] = Candidate(position, score, None, ("word",))
    for position, score in grams:
        if position in by_position:
            word_score = by_position[position].word_score
            by_position[position] = Candidate(position, word_score, score, ("word", "char"))
        else:
            by_position[position] = Candidate(position, None, score, ("char",))
    return [by_position[position] for position in sorted(by_position)]


def build_indexes(prepared_targets: list[PreparedRecord]) -> tuple[InvertedIndex, InvertedIndex]:
    """Build both channels from the SAME partition-local target sequence."""
    words = InvertedIndex.build(record.title_word_tokens for record in prepared_targets)
    grams = InvertedIndex.build(record.title_trigrams for record in prepared_targets)
    return words, grams


def evaluate_retrieval(prepared_left: list[PreparedRecord], prepared_right: list[PreparedRecord],
                       gold_positions: set[tuple[int, int]]) -> dict:
    """Evaluate exactly seven configurations on the caller's verified validation split.

    Gold positions are used only to count hits after retrieval. Timings exclude
    preparation, index construction, gold membership checks, and report writing.
    """
    if not gold_positions:
        raise ValueError("Validation configuration selection requires at least one gold pair")
    for left, right in gold_positions:
        if not (0 <= left < len(prepared_left) and 0 <= right < len(prepared_right)):
            raise ValueError("Gold position is outside the evaluated partition")
    started = perf_counter()
    word_index, gram_index = build_indexes(prepared_right)
    build_seconds = perf_counter() - started
    index_stats = {}
    for name, index in (("word", word_index), ("char", gram_index)):
        index_stats[name] = {
            "records": index.n_records, "vocabulary": len(index.postings),
            "term_record_associations": sum(index.df.values()),
        }
    rows = []
    cartesian_pairs = len(prepared_left) * len(prepared_right)
    for config in RETRIEVAL_CONFIGS:
        stats = RetrievalStats()
        candidate_pairs = 0
        gold_hits = 0
        maximum_candidates = 0
        retrieval_seconds = 0.0
        maximum_latency = 0.0
        for left_position, record in enumerate(prepared_left):
            started = perf_counter()
            candidates = retrieve_candidates(record, word_index, gram_index, config, stats)
            elapsed = perf_counter() - started
            retrieval_seconds += elapsed
            maximum_latency = max(maximum_latency, elapsed)
            candidate_pairs += len(candidates)
            maximum_candidates = max(maximum_candidates, len(candidates))
            for candidate in candidates:
                if (left_position, candidate.target_position) in gold_positions:
                    gold_hits += 1
        rows.append({
            "name": config.name, "word_k": config.word_k, "gram_k": config.gram_k,
            "gold_hits": gold_hits, "gold_pairs": len(gold_positions),
            "candidate_recall": gold_hits / len(gold_positions),
            "candidate_pairs": candidate_pairs,
            "reduction_ratio": 1.0 - candidate_pairs / cartesian_pairs,
            "average_candidates": candidate_pairs / len(prepared_left),
            "maximum_candidates": maximum_candidates,
            "retrieval_seconds": retrieval_seconds,
            "average_latency_ms": 1000 * retrieval_seconds / len(prepared_left),
            "maximum_latency_ms": 1000 * maximum_latency,
            "postings_visited": stats.postings_visited,
            "distinct_candidates_encountered": stats.distinct_candidates,
        })
    selected = select_configuration(rows)
    selected_row = next(row for row in rows if row["name"] == selected.name)
    return {
        "query_records": len(prepared_left), "target_records": len(prepared_right),
        "cartesian_pairs": cartesian_pairs, "recall_target": RECALL_TARGET,
        "index_build_seconds": build_seconds, "indexes": index_stats,
        "comparison": rows,
        "selected_config": {"name": selected.name, "word_k": selected.word_k, "gram_k": selected.gram_k},
        "recall_target_met": selected_row["candidate_recall"] >= RECALL_TARGET,
        "selection_rule": "fewest pairs at recall >= 0.98; otherwise highest recall, then fewest pairs; fixed order breaks ties",
    }


def select_configuration(rows: list[dict]) -> RetrievalConfig:
    """Apply the prescribed rule in fixed configuration order, independent of input order."""
    by_name = {row["name"]: row for row in rows}
    if len(rows) != len(RETRIEVAL_CONFIGS) or set(by_name) != {c.name for c in RETRIEVAL_CONFIGS}:
        raise ValueError("Selection requires exactly the seven predefined configurations")
    ordered = [by_name[config.name] for config in RETRIEVAL_CONFIGS]
    qualifying = [row for row in ordered if row["candidate_recall"] >= RECALL_TARGET]
    if qualifying:
        best = min(qualifying, key=lambda row: row["candidate_pairs"])
    else:
        best = min(ordered, key=lambda row: (-row["candidate_recall"], row["candidate_pairs"]))
    return next(config for config in RETRIEVAL_CONFIGS if config.name == best["name"])
