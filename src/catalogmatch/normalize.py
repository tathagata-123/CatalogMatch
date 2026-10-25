"""Small, deterministic text and USD price preparation functions."""

import html
from html.parser import HTMLParser
import math
import re
import unicodedata


NORMALIZER_VERSION = "nfkc_casefold_ascii_title_trigrams_v1"
WORD_PATTERN = re.compile(r"[a-z0-9]+")
MODEL_PATTERN = re.compile(r"[a-z0-9]+(?:[-./][a-z0-9]+)*")
PRICE_PATTERN = re.compile(r"\$?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?")


class _HTMLTextExtractor(HTMLParser):
    """Keep text nodes; spaces prevent adjacent tags from joining words."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def normalize_text(text: str, *, strip_html: bool = False) -> str:
    """Decode entities, optionally extract markup text, then normalize words.

    Parse markup before decoding text entities so an escaped literal such as
    &lt;camera&gt; remains text instead of becoming a fabricated HTML tag.
    """
    if strip_html:
        parser = _HTMLTextExtractor()
        parser.feed(text)
        parser.close()
        text = " ".join(parser.parts)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(WORD_PATTERN.findall(text))


def word_tokens(normalized_text: str) -> frozenset[str]:
    return frozenset(normalized_text.split())


def character_trigrams(normalized_title: str) -> frozenset[str]:
    if not normalized_title:
        return frozenset()
    if len(normalized_title) < 3:
        return frozenset({normalized_title})
    trigrams = set()
    for index in range(len(normalized_title) - 2):
        trigrams.add(normalized_title[index:index + 3])
    return frozenset(trigrams)


def model_tokens(raw_title: str) -> frozenset[str]:
    """Extract heuristic model evidence without joining whitespace-separated words."""
    text = unicodedata.normalize("NFKC", raw_title).casefold()
    models = set()
    for chunk in MODEL_PATTERN.findall(text):
        compact = chunk.replace("-", "").replace(".", "").replace("/", "")
        if len(compact) >= 4:
            has_letter = any("a" <= character <= "z" for character in compact)
            has_digit = any("0" <= character <= "9" for character in compact)
            if has_letter and has_digit:
                models.add(compact)
    return frozenset(models)


def numeric_tokens(normalized_title: str) -> frozenset[str]:
    numbers = set()
    for token in normalized_title.split():
        if token.isascii() and token.isdigit():
            numbers.add(token)
    return frozenset(numbers)


def parse_price(raw_price: str) -> float | None:
    """Parse positive finite decimal USD prices; blank/invalid values stay missing."""
    text = raw_price.strip()
    if not PRICE_PATTERN.fullmatch(text):
        return None
    value = float(text.removeprefix("$").replace(",", ""))
    if value <= 0 or not math.isfinite(value):
        return None
    return value
