"""Twelve attribute comparisons, ordered candidate matrices, and a small disk cache."""

from collections.abc import Sequence
import hashlib
import io
import json
import math
from numbers import Integral
import os
from pathlib import Path
import tempfile
from time import perf_counter
import zipfile
import zlib

import numpy as np

from catalogmatch.data import PreparedRecord, prepared_attributes
from catalogmatch.normalize import NORMALIZER_VERSION
from catalogmatch.similarity import containment, edit_similarity, jaccard


FEATURE_SCHEMA_VERSION = "product_pair_12_v1"
FEATURE_NAMES = (
    "title_token_jaccard",
    "title_token_containment",
    "title_trigram_jaccard",
    "title_edit_similarity",
    "model_token_jaccard",
    "model_token_conflict",
    "model_tokens_both_present",
    "description_token_jaccard",
    "descriptions_both_present",
    "relative_price_difference",
    "prices_both_present",
    "numeric_token_jaccard",
)


def pair_features(left: PreparedRecord, right: PreparedRecord) -> tuple[float, ...]:
    """Read prepared attributes only; no IDs, labels, scores, or normalization."""
    for price in (left.price, right.price):
        if price is not None and (not math.isfinite(price) or price <= 0):
            raise ValueError("Prepared price must be positive and finite, or None")
    models_present = bool(left.model_tokens and right.model_tokens)
    prices_present = left.price is not None and right.price is not None
    price_difference = 0.0
    if prices_present:
        price_difference = abs(left.price - right.price) / max(left.price, right.price)
    values = (
        jaccard(left.title_word_tokens, right.title_word_tokens),
        containment(left.title_word_tokens, right.title_word_tokens),
        jaccard(left.title_trigrams, right.title_trigrams),
        edit_similarity(left.normalized_title, right.normalized_title),
        jaccard(left.model_tokens, right.model_tokens),
        float(models_present and left.model_tokens.isdisjoint(right.model_tokens)),
        float(models_present),
        jaccard(left.description_word_tokens, right.description_word_tokens),
        float(bool(left.normalized_description and right.normalized_description)),
        price_difference,
        float(prices_present),
        jaccard(left.numeric_title_tokens, right.numeric_title_tokens),
    )
    for name, value in zip(FEATURE_NAMES, values):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"Invalid feature {name}: {value!r}")
    return values


def _pair_positions(pair, row: int, left_count: int, right_count: int) -> tuple[int, int]:
    try:
        left, right = pair
    except (TypeError, ValueError) as error:
        raise ValueError(f"Pair row {row} must contain exactly two integer positions") from error
    for side, position, count in (("left", left, left_count), ("right", right, right_count)):
        if isinstance(position, (bool, np.bool_)) or not isinstance(position, Integral):
            raise ValueError(f"Pair row {row}: {side} position {position!r} must be an integer")
        if not 0 <= position < count:
            raise ValueError(f"Pair row {row}: {side} position {position} outside [0, {count})")
    return int(left), int(right)


def feature_matrix(pairs: Sequence[tuple[int, int]],
                   left_records: Sequence[PreparedRecord],
                   right_records: Sequence[PreparedRecord]) -> np.ndarray:
    """Preserve every candidate row, including duplicates; never create new pairs."""
    matrix = np.empty((len(pairs), len(FEATURE_NAMES)), dtype=np.float64)
    for row, pair in enumerate(pairs):
        left, right = _pair_positions(pair, row, len(left_records), len(right_records))
        try:
            matrix[row] = pair_features(left_records[left], right_records[right])
        except ValueError as error:
            raise ValueError(f"Pair row {row}: {error}") from error
    return matrix


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def feature_cache_identity(pairs: Sequence[tuple[int, int]],
                           left_records: Sequence[PreparedRecord],
                           right_records: Sequence[PreparedRecord], provenance: dict) -> dict:
    """Bind metadata to actual ordered IDs/positions and prepared attribute contents.

    Provenance is supplied by the checked CLI (or a caller with verified inputs).
    IDs and split metadata identify cache rows; they never enter numeric features.
    """
    required = ("raw_sha256", "manifest_sha256", "split", "retrieval_config", "retrieval_version")
    for key in required:
        if key not in provenance:
            raise ValueError(f"Feature cache provenance is missing {key}")
    # Snapshot JSON-compatible caller metadata, avoiding references to mutable dicts.
    context = json.loads(_json_bytes(provenance))
    pair_digest = hashlib.sha256()
    for row, pair in enumerate(pairs):
        left, right = _pair_positions(pair, row, len(left_records), len(right_records))
        identifiers = [left, right, left_records[left].raw.key, right_records[right].raw.key]
        pair_digest.update(_json_bytes(identifiers) + b"\n")
    record_digests = []
    for records in (left_records, right_records):
        digest = hashlib.sha256()
        for record in records:
            digest.update(_json_bytes([record.raw.key, prepared_attributes(record)]) + b"\n")
        record_digests.append(digest.hexdigest())
    return {
        "provenance": context,
        "normalizer_version": NORMALIZER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "pair_count": len(pairs),
        "ordered_pairs_sha256": pair_digest.hexdigest(),
        "prepared_records_sha256": record_digests,
    }


def _validate_matrix(matrix: np.ndarray, pair_count: int) -> None:
    if (not isinstance(matrix, np.ndarray)
            or matrix.shape != (pair_count, len(FEATURE_NAMES))
            or matrix.dtype != np.dtype(np.float64)):
        raise ValueError("Cached matrix has incompatible shape or dtype")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0) or np.any(matrix > 1):
        raise ValueError("Cached matrix contains nonfinite or out-of-range features")


def _load_cache(path: Path, identity: dict, pairs: np.ndarray) -> np.ndarray:
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict) or metadata.get("identity") != identity:
        raise ValueError("Feature cache identity is stale or malformed")
    # Hash and decode the same bytes, even if another process replaces the file.
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != metadata.get("npz_sha256"):
        raise ValueError("Feature cache checksum mismatch")
    with np.load(io.BytesIO(contents), allow_pickle=False) as saved:
        if len(saved.files) != 2 or set(saved.files) != {"X", "pairs"}:
            raise ValueError("Feature cache must contain X and pairs")
        matrix = saved["X"]
        saved_pairs = saved["pairs"]
        if (not isinstance(saved_pairs, np.ndarray)
                or saved_pairs.dtype != np.dtype(np.int64)
                or not np.array_equal(saved_pairs, pairs)):
            raise ValueError("Cached pair positions differ from requested order")
    _validate_matrix(matrix, len(pairs))
    return matrix


def _save_cache(path: Path, identity: dict, pairs: np.ndarray, matrix: np.ndarray) -> None:
    """Replace complete files; the checksum detects interrupted or mixed generations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths = []
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
            array_path = Path(stream.name)
            temporary_paths.append(array_path)
            np.savez_compressed(stream, X=matrix, pairs=pairs)
        metadata = {"identity": identity, "npz_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest()}
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".json", delete=False) as stream:
            metadata_path = Path(stream.name)
            temporary_paths.append(metadata_path)
            stream.write(_json_bytes(metadata) + b"\n")
        os.replace(array_path, path)
        os.replace(metadata_path, path.with_suffix(".json"))
    finally:
        for temporary_path in temporary_paths:
            temporary_path.unlink(missing_ok=True)


def cached_feature_matrix(path: Path, pairs: Sequence[tuple[int, int]],
                          left_records: Sequence[PreparedRecord],
                          right_records: Sequence[PreparedRecord], *,
                          provenance: dict) -> tuple[np.ndarray, dict]:
    """Return a reusable matrix plus timing/cache status; regenerate unusable caches."""
    if path.suffix != ".npz":
        raise ValueError("Feature cache path must end in .npz")
    started = perf_counter()
    identity = feature_cache_identity(pairs, left_records, right_records, provenance)
    positions = np.asarray(pairs, dtype=np.int64).reshape(len(pairs), 2)
    reason = None
    try:
        matrix = _load_cache(path, identity, positions)
    except (OSError, ValueError, TypeError, KeyError, EOFError, zipfile.BadZipFile, zlib.error) as error:
        reason = str(error)
    else:
        return matrix, {"cache_used": True, "cache_reason": "identity and contents verified",
                        "feature_seconds": 0.0, "cache_seconds": perf_counter() - started}
    cache_seconds = perf_counter() - started
    started = perf_counter()
    matrix = feature_matrix(pairs, left_records, right_records)
    feature_seconds = perf_counter() - started
    started = perf_counter()
    _save_cache(path, identity, positions, matrix)
    cache_seconds += perf_counter() - started
    return matrix, {"cache_used": False, "cache_reason": reason,
                    "feature_seconds": feature_seconds, "cache_seconds": cache_seconds}
