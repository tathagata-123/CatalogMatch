import pytest

from catalogmatch.data import RawRecord, prepare_record
from catalogmatch.normalize import (
    character_trigrams, model_tokens, normalize_text, numeric_tokens,
    parse_price, word_tokens,
)


@pytest.mark.parametrize(("raw", "expected"), [
    ("CAMERA Camera camera", "camera camera camera"),
    ("ＣＡＭＥＲＡ １２３ ①", "camera 123 1"),
    ("Straße Café", "strasse caf"),
    ("A&amp;B &quot;Lens&quot;", "a b lens"),
    ("HD-TV (42\") / ZX_100", "hd tv 42 zx 100"),
    (" \tCamera\n  lens\r\n", "camera lens"),
    ("", ""),
    (" !—\t", ""),
])
def test_normalization(raw, expected):
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("<p>Camera &amp; lens</p><br>42", "camera lens 42"),
    ('<div class="product">A</div><div>B</div>', "a b"),
    ("&lt;camera&gt; &amp; lens", "camera lens"),
    ("plain 2 < 3 &amp; 4 > 1", "plain 2 3 4 1"),
    ("<p></p>", ""),
])
def test_description_markup(raw, expected):
    assert normalize_text(raw, strip_html=True) == expected


def test_distinct_words_and_numeric_tokens():
    assert word_tokens("camera lens camera") == {"camera", "lens"}
    assert word_tokens("") == frozenset()
    assert numeric_tokens("42 42 001 3d 12mp") == {"42", "001"}


@pytest.mark.parametrize(("title", "expected"), [
    ("", set()), ("a", {"a"}), ("ab", {"ab"}), ("abc", {"abc"}),
    ("ab cd", {"ab ", "b c", " cd"}), ("aaaa", {"aaa"}),
])
def test_overlapping_trigrams(title, expected):
    assert character_trigrams(title) == expected


@pytest.mark.parametrize(("title", "expected"), [
    ("Sony KDL-40.V/123", {"kdl40v123"}),
    ("ＺＸ－１００ zx100 ZX/100", {"zx100"}),
    ("AB 1234 A1 ABC 12345 A-123", {"a123"}),
    ("AB12/CD34 ab12.cd34", {"ab12cd34"}),
    ("", set()),
])
def test_model_extraction(title, expected):
    assert model_tokens(title) == expected


@pytest.mark.parametrize(("raw", "expected"), [
    ("$1,299.99", 1299.99), ("1299.99", 1299.99), ("250", 250.0),
    ("  $250.00 \t", 250.0), ("1,000,000.01", 1000000.01),
    ("0.01", 0.01), ("001.50", 1.5),
])
def test_valid_prices(raw, expected):
    assert parse_price(raw) == expected


@pytest.mark.parametrize("raw", [
    "", " \t", "-1", "$-1", "0", "$0.00", "+2", "NaN", "nan", "inf",
    "Infinity", "$Infinity", "1,00", "12,34.50", "1234,567", "1,,000",
    "1,000,00", "1.2.3", "1e3", "$.50", "250.", "$$250", "$ 250",
    "€250", "250 USD", "1 000", "１２３", "9" * 400, "0." + "0" * 400 + "1",
])
def test_invalid_or_missing_prices(raw):
    assert parse_price(raw) is None


def test_prepare_record_preserves_raw_and_uses_attributes_only():
    raw = RawRecord("abt", "ID_SECRET_9876", "ACME ZX-100 42", "<b>Lens</b> &amp; case", "", "$1,299.99")
    prepared = prepare_record(raw)
    assert prepared.raw is raw
    assert raw.name == "ACME ZX-100 42"
    assert prepared.normalized_title == "acme zx 100 42"
    assert prepared.normalized_description == "lens case"
    assert prepared.title_word_tokens == {"acme", "zx", "100", "42"}
    assert prepared.description_word_tokens == {"lens", "case"}
    assert prepared.title_trigrams == character_trigrams("acme zx 100 42")
    assert prepared.model_tokens == {"zx100"}
    assert prepared.numeric_title_tokens == {"100", "42"}
    assert prepared.price == 1299.99
    assert isinstance(prepared.title_word_tokens, frozenset)
    # Different source IDs and manufacturers cannot affect these title attributes.
    other = prepare_record(RawRecord("buy", "OTHER_1234", raw.name, raw.description, "Other", raw.raw_price))
    assert other.title_word_tokens == prepared.title_word_tokens
    assert other.model_tokens == prepared.model_tokens


def test_prepare_blank_record():
    prepared = prepare_record(RawRecord("abt", "1", "", "", "", ""))
    assert prepared.normalized_title == ""
    assert not prepared.title_word_tokens
    assert not prepared.description_word_tokens
    assert not prepared.title_trigrams
    assert not prepared.model_tokens
    assert not prepared.numeric_title_tokens
    assert prepared.price is None
