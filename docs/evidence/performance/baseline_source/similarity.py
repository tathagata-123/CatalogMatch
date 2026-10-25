"""Exact string/set comparisons and the original exhaustive lexical baseline."""

from collections.abc import Iterator
import math
from typing import AbstractSet

from catalogmatch.data import Dataset, Partitions, partition_records, prepare_record, PreparedRecord


def levenshtein(a: str, b: str) -> int:
    """Unit-cost edit distance in O(len(a) * len(b)) time and O(min) space."""
    if len(a) < len(b):
        a, b = b, a
    # The shorter string is the column dimension. Empty prefixes cost their length.
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    current = [0] * (len(b) + 1)
    for i, a_character in enumerate(a, 1):
        current[0] = i
        for j, b_character in enumerate(b, 1):
            cost = 0 if a_character == b_character else 1
            # Above and diagonal belong to the previous row; left is already done.
            current[j] = min(previous[j] + 1,
                             current[j - 1] + 1,
                             previous[j - 1] + cost)
        previous, current = current, previous
    return previous[len(b)]


def edit_similarity(a: str, b: str) -> float:
    """Compare already normalized titles; missing titles are not positive evidence."""
    if not a or not b:
        return 0.0
    return 1.0 - levenshtein(a, b) / max(len(a), len(b))


def containment(left: AbstractSet[str], right: AbstractSet[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def jaccard(left: AbstractSet[str], right: AbstractSet[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def lexical_score(left: PreparedRecord, right: PreparedRecord) -> float:
    words = jaccard(left.title_word_tokens, right.title_word_tokens)
    trigrams = jaccard(left.title_trigrams, right.title_trigrams)
    return max(words, trigrams)


def training_baseline(dataset: Dataset, partitions: Partitions,
                      threshold: float = 0.5) -> Iterator[tuple[str, str, float]]:
    """Compare every training Abt/Buy pair and allow multiple matches per record.

    Preparation is done once per training record. Streaming accepted pairs avoids
    storing the full Cartesian product. Gold labels never enter lexical scoring.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one")
    abt = [prepare_record(record) for record in partition_records(dataset, partitions, "train", "abt")]
    buy = [prepare_record(record) for record in partition_records(dataset, partitions, "train", "buy")]
    for left in abt:
        for right in buy:
            score = lexical_score(left, right)
            if score >= threshold:
                yield left.raw.raw_id, right.raw.raw_id, score
