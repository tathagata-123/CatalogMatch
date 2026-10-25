from dataclasses import replace

import pytest

from catalogmatch.data import Dataset, RawRecord, prepare_record
from catalogmatch.similarity import jaccard, lexical_score, training_baseline


@pytest.mark.parametrize(("left", "right", "expected"), [
    ({"a", "b"}, {"a", "b"}, 1.0),
    ({"a"}, {"b"}, 0.0),
    ({"a", "b"}, {"b", "c"}, 1 / 3),
    (set(), {"a"}, 0.0), ({"a"}, set(), 0.0), (set(), set(), 0.0),
])
def test_jaccard(left, right, expected):
    assert jaccard(left, right) == pytest.approx(expected)
    assert jaccard(frozenset(right), frozenset(left)) == pytest.approx(expected)


def prepared(title):
    return prepare_record(RawRecord("abt", "unused", title, "", "", ""))


def test_lexical_score_selects_word_similarity():
    left, right = prepared("red blue"), prepared("blue red")
    assert jaccard(left.title_trigrams, right.title_trigrams) < 1
    assert lexical_score(left, right) == 1


def test_lexical_score_selects_trigram_similarity():
    left, right = prepared("abc1234"), prepared("abc1235")
    assert jaccard(left.title_word_tokens, right.title_word_tokens) == 0
    assert lexical_score(left, right) == pytest.approx(4 / 6)


def test_lexical_score_empty_and_identical_titles():
    assert lexical_score(prepared(""), prepared("")) == 0
    assert lexical_score(prepared("Camera"), prepared("CAMERA")) == 1


def test_exhaustive_baseline_is_training_only_and_many_to_many(monkeypatch):
    abt = (
        RawRecord("abt", "a1", "camera lens", "", "", ""),
        RawRecord("abt", "a2", "camera lens", "", "", ""),
        RawRecord("abt", "a_validation", "camera lens", "", "", ""),
        RawRecord("abt", "a_test", "camera lens", "", "", ""),
    )
    buy = (
        RawRecord("buy", "b1", "camera lens", "", "", ""),
        RawRecord("buy", "b2", "camera lens", "", "", ""),
        RawRecord("buy", "b_validation", "camera lens", "", "", ""),
        RawRecord("buy", "b_test", "camera lens", "", "", ""),
    )
    dataset = Dataset(abt, buy, (("a1", "b1"), ("a1", "b2")), {})
    partitions = {
        "train": (("abt", "a2"), ("buy", "b2"), ("abt", "a1"), ("buy", "b1")),
        "validation": (("abt", "a_validation"), ("buy", "b_validation")),
        "test": (("abt", "a_test"), ("buy", "b_test")),
    }
    seen = []

    def prepare_training_only(record):
        assert record.key in partitions["train"]
        seen.append(record.key)
        return prepare_record(record)

    monkeypatch.setattr("catalogmatch.similarity.prepare_record", prepare_training_only)
    expected = [("a1", "b1", 1.0), ("a1", "b2", 1.0), ("a2", "b1", 1.0), ("a2", "b2", 1.0)]
    assert list(training_baseline(dataset, partitions)) == expected
    assert len(seen) == len(set(seen)) == 4  # Prepare each training record once.
    # Scores/predictions cannot depend on labels or record IDs as attributes.
    assert list(training_baseline(replace(dataset, gold_pairs=()), partitions)) == expected


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf"), -float("inf")])
def test_baseline_invalid_threshold(dataset, partitions, threshold):
    with pytest.raises(ValueError, match="Threshold"):
        list(training_baseline(dataset, partitions, threshold))


def test_baseline_inclusive_threshold_and_exhaustiveness(dataset, partitions):
    training_abt = [r for r in dataset.abt if r.key in partitions["train"]]
    training_buy = [r for r in dataset.buy if r.key in partitions["train"]]
    all_pairs = list(training_baseline(dataset, partitions, threshold=0))
    assert len(all_pairs) == len(training_abt) * len(training_buy)
    for left, right, score in all_pairs:
        accepted = list(training_baseline(dataset, partitions, threshold=score))
        assert (left, right, score) in accepted


def test_empty_training_split(dataset):
    assert list(training_baseline(dataset, {"train": (), "validation": (), "test": ()})) == []


# Independent reference: intentionally use the full textbook table in tests only.
def full_table_levenshtein(a, b):
    table = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        table[i][0] = i
    for j in range(len(b) + 1):
        table[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            substitution = table[i - 1][j - 1]
            if a[i - 1] != b[j - 1]:
                substitution += 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1, substitution)
    return table[-1][-1]


@pytest.mark.parametrize("a,b,expected", [
    ("", "", 0), ("", "abc", 3), ("abc", "", 3), ("same", "same", 0),
    ("cat", "cart", 1), ("cart", "cat", 1), ("cat", "cut", 1),
    ("kitten", "sitting", 3), ("saturday", "sunday", 3), ("aaaa", "aa", 2),
    ("a", "abcdefgh", 7), ("ab", "ba", 2), ("café", "cafe", 1),
    ("abc", "xyz", 3), ("a" * 400, "a" * 399 + "b", 1),
])
def test_levenshtein_examples(a, b, expected):
    from catalogmatch.similarity import levenshtein
    assert full_table_levenshtein(a, b) == expected
    assert levenshtein(a, b) == expected
    assert levenshtein(b, a) == expected
    assert abs(len(a) - len(b)) <= expected <= max(len(a), len(b))


def test_levenshtein_randomized_oracle():
    import random
    from catalogmatch.similarity import levenshtein
    rng = random.Random(481)
    for _ in range(500):
        a = "".join(rng.choices("aabc 12é", k=rng.randrange(16)))
        b = "".join(rng.choices("aabc 12é", k=rng.randrange(16)))
        distance = levenshtein(a, b)
        assert distance == full_table_levenshtein(a, b)
        assert distance == levenshtein(b, a)
        assert abs(len(a) - len(b)) <= distance <= max(len(a), len(b))


@pytest.mark.parametrize("a,b", [("a" * 180, "b" * 180),
                                  ("a" * 8000, "b" * 8), ("a" * 8, "b" * 8000)])
def test_levenshtein_memory_is_two_short_rows(a, b):
    import tracemalloc
    from catalogmatch.similarity import levenshtein
    tracemalloc.start()
    try:
        assert levenshtein(a, b) == max(len(a), len(b))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # A 181x181 table alone needs >250 KB; a row of 8001 integers >60 KB.
    # Leave ample room for two short rows and interpreter temporaries.
    assert peak < 25000


@pytest.mark.parametrize("a,b,expected", [
    ("", "", 0.0), ("", "abc", 0.0), ("abc", "", 0.0),
    ("same", "same", 1.0), ("kitten", "sitting", 4 / 7),
    ("abc", "xyz", 0.0), ("ab", "abc", 2 / 3),
    ("A", "a", 0.0),  # The primitive must not normalize its inputs.
])
def test_edit_similarity(a, b, expected):
    from catalogmatch.similarity import edit_similarity
    assert edit_similarity(a, b) == pytest.approx(expected)
    assert edit_similarity(b, a) == pytest.approx(expected)
    assert 0 <= edit_similarity(a, b) <= 1


@pytest.mark.parametrize("a,b,expected", [
    (set(), set(), 0.0), ({"a"}, set(), 0.0), (set(), {"a"}, 0.0),
    ({"a"}, {"a", "b"}, 1.0), ({"a", "b"}, {"b", "c", "d"}, 0.5),
    ({"a"}, {"b"}, 0.0), ({"a", "b"}, {"a", "b"}, 1.0),
])
def test_containment(a, b, expected):
    from catalogmatch.similarity import containment
    assert containment(a, b) == expected
    assert containment(b, a) == expected
