"""CSV validation, record preparation, and entity-disjoint split metadata."""

from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random

from catalogmatch.normalize import (
    NORMALIZER_VERSION,
    character_trigrams,
    model_tokens,
    normalize_text,
    numeric_tokens,
    parse_price,
    word_tokens,
)


RecordKey = tuple[str, str]
GoldPair = tuple[str, str]  # (Abt raw ID, Buy raw ID), always a positive edge
Component = tuple[RecordKey, ...]
Partitions = dict[str, tuple[RecordKey, ...]]
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_VERSION = "component60_20_20_python_random42_v1"
SCHEMAS = {
    "Abt.csv": ("id", "name", "description", "price"),
    "Buy.csv": ("id", "name", "description", "manufacturer", "price"),
    "abt_buy_perfectMapping.csv": ("idAbt", "idBuy"),
}
EXPECTED_ROWS = {"abt": 1081, "buy": 1092, "gold": 1097}
EXPECTED_MISSING = {
    "abt": {"id": 0, "name": 0, "description": 0, "price": 663},
    "buy": {"id": 0, "name": 0, "description": 441, "manufacturer": 6, "price": 502},
}
EXPECTED_SHAPES = {"1_abt_1_buy": 1055, "1_abt_2_buy": 16, "2_abt_1_buy": 5}
EXPECTED_SPLITS = {
    "train": {"components": 645, "abt": 648, "buy": 655},
    "validation": {"components": 215, "abt": 217, "buy": 218},
    "test": {"components": 216, "abt": 216, "buy": 219},
}


class DataValidationError(ValueError):
    """A dataset or its split metadata violates a required invariant."""


@dataclass(frozen=True)
class RawRecord:
    source: str
    raw_id: str
    name: str
    description: str
    manufacturer: str
    raw_price: str

    def __post_init__(self) -> None:
        if self.source not in ("abt", "buy"):
            raise DataValidationError(f"Unknown source: {self.source!r}")
        if not isinstance(self.raw_id, str) or not self.raw_id.strip():
            raise DataValidationError("Record IDs must be nonblank strings")

    @property
    def key(self) -> RecordKey:
        return (self.source, self.raw_id)


@dataclass(frozen=True)
class PreparedRecord:
    raw: RawRecord
    normalized_title: str
    normalized_description: str
    title_word_tokens: frozenset[str]
    description_word_tokens: frozenset[str]
    title_trigrams: frozenset[str]
    model_tokens: frozenset[str]
    numeric_title_tokens: frozenset[str]
    price: float | None


@dataclass(frozen=True)
class Dataset:
    abt: tuple[RawRecord, ...]
    buy: tuple[RawRecord, ...]
    gold_pairs: tuple[GoldPair, ...]
    raw_sha256: dict[str, str]

    @property
    def records(self) -> tuple[RawRecord, ...]:
        return self.abt + self.buy


def prepare_record(record: RawRecord) -> PreparedRecord:
    """Prepare product attributes only; IDs and gold metadata are never tokens."""
    title = normalize_text(record.name)
    description = normalize_text(record.description, strip_html=True)
    return PreparedRecord(
        raw=record,
        normalized_title=title,
        normalized_description=description,
        title_word_tokens=word_tokens(title),
        description_word_tokens=word_tokens(description),
        title_trigrams=character_trigrams(title),
        model_tokens=model_tokens(record.name),
        numeric_title_tokens=numeric_tokens(title),
        price=parse_price(record.raw_price),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, expected_columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Read strict Latin-1 CSV, distinguishing blank values from missing fields."""
    rows = []
    with path.open(encoding="latin-1", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        try:
            if reader.fieldnames != list(expected_columns):
                raise DataValidationError(
                    f"{path.name}: expected columns {expected_columns}, got {reader.fieldnames}"
                )
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise DataValidationError(
                        f"{path.name}, line {reader.line_num}: missing fields or extra fields"
                    )
                rows.append(row)
        except csv.Error as error:
            raise DataValidationError(f"{path.name}, line {reader.line_num}: {error}") from error
    return rows


def load_records(path: Path, source: str) -> tuple[RawRecord, ...]:
    if source not in ("abt", "buy"):
        raise DataValidationError(f"Unknown source: {source!r}")
    columns = SCHEMAS["Abt.csv" if source == "abt" else "Buy.csv"]
    records = []
    seen_ids = set()
    for row in read_csv(path, columns):
        if row["id"] in seen_ids:
            raise DataValidationError(f"{path.name}: duplicate ID {row['id']!r}")
        seen_ids.add(row["id"])
        records.append(RawRecord(
            source, row["id"], row["name"], row["description"],
            row.get("manufacturer", ""), row["price"],
        ))
    return tuple(records)


def load_gold(path: Path, abt: tuple[RawRecord, ...],
              buy: tuple[RawRecord, ...]) -> tuple[GoldPair, ...]:
    abt_ids = {record.raw_id for record in abt}
    buy_ids = {record.raw_id for record in buy}
    pairs = []
    seen_pairs = set()
    for row in read_csv(path, SCHEMAS["abt_buy_perfectMapping.csv"]):
        pair = (row["idAbt"], row["idBuy"])
        if pair in seen_pairs:
            raise DataValidationError(f"{path.name}: duplicate gold pair {pair}")
        if pair[0] not in abt_ids or pair[1] not in buy_ids:
            raise DataValidationError(f"{path.name}: nonexistent gold reference {pair}")
        seen_pairs.add(pair)
        pairs.append(pair)
    return tuple(pairs)


def load_dataset(raw_dir: Path) -> Dataset:
    abt = load_records(raw_dir / "Abt.csv", "abt")
    buy = load_records(raw_dir / "Buy.csv", "buy")
    gold = load_gold(raw_dir / "abt_buy_perfectMapping.csv", abt, buy)
    hashes = {}
    for filename in SCHEMAS:
        hashes[filename] = sha256_file(raw_dir / filename)
    return Dataset(abt, buy, gold, hashes)


def audit_dataset(dataset: Dataset, *, original: bool = True) -> dict:
    """Report missing/malformed prices; optionally require original dataset counts."""
    # Revalidate the graph even for a Dataset constructed directly in Python.
    build_positive_graph(dataset.records, dataset.gold_pairs)
    missing = {}
    malformed = {}
    for source, records in (("abt", dataset.abt), ("buy", dataset.buy)):
        counts = dict.fromkeys(EXPECTED_MISSING[source], 0)
        bad_prices = 0
        for record in records:
            values = {
                "id": record.raw_id, "name": record.name,
                "description": record.description, "manufacturer": record.manufacturer,
                "price": record.raw_price,
            }
            for field in counts:
                if not values[field].strip():
                    counts[field] += 1
            if record.raw_price.strip() and parse_price(record.raw_price) is None:
                bad_prices += 1
        missing[source] = counts
        malformed[source] = bad_prices
    report = {
        "rows": {"abt": len(dataset.abt), "buy": len(dataset.buy),
                 "gold": len(dataset.gold_pairs)},
        "missing": missing,
        "malformed_prices": malformed,
        "raw_sha256": dataset.raw_sha256,
    }
    if original:
        if report["rows"] != EXPECTED_ROWS:
            raise DataValidationError(f"Unexpected original row counts: {report['rows']}; expected {EXPECTED_ROWS}")
        if missing != EXPECTED_MISSING:
            raise DataValidationError(f"Unexpected original missing values: {missing}; expected {EXPECTED_MISSING}")
        if any(malformed.values()):
            raise DataValidationError(f"Malformed original USD prices: {malformed}")
    return report


def build_positive_graph(records: tuple[RawRecord, ...],
                         gold_pairs: tuple[GoldPair, ...]) -> dict[RecordKey, set[RecordKey]]:
    """Include all vertices, but add edges ONLY from the gold-positive mapping.

    Negative candidate pairs have no input here and must never connect entities.
    Namespacing keeps identical raw IDs from different sources distinct.
    """
    graph: dict[RecordKey, set[RecordKey]] = {}
    for record in records:
        if record.key in graph:
            raise DataValidationError(f"Duplicate record key {record.key}")
        graph[record.key] = set()
    seen_pairs = set()
    for pair in gold_pairs:
        # This API accepts positive ID pairs, never labeled candidate triples.
        if len(pair) != 2 or any(not isinstance(raw_id, str) for raw_id in pair):
            raise DataValidationError(f"Expected a positive pair of string IDs, got {pair!r}")
        if pair in seen_pairs:
            raise DataValidationError(f"Duplicate gold pair {pair}")
        seen_pairs.add(pair)
        left, right = ("abt", pair[0]), ("buy", pair[1])
        if left not in graph or right not in graph:
            raise DataValidationError(f"Nonexistent gold reference {pair}")
        graph[left].add(right)
        graph[right].add(left)
    return graph


def connected_components(graph: dict[RecordKey, set[RecordKey]]) -> list[Component]:
    """Iterative DFS is O(V + E); canonical sorting happens after traversal."""
    visited = set()
    components = []
    for start in graph:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        component = []
        while stack:
            vertex = stack.pop()
            component.append(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in visited:
                    # Mark on push: each vertex enters the stack at most once.
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    components.sort(key=lambda component: component[0])
    return components


def split_components(components: list[Component], seed: int = 42) -> Partitions:
    """Assign whole entities using one RNG and matched/Abt-only/Buy-only buckets."""
    buckets: list[list[Component]] = [[], [], []]
    canonical = sorted((tuple(sorted(component)) for component in components))
    seen = set()
    for component in canonical:
        if not component:
            raise DataValidationError("Empty connected component")
        sources = {vertex[0] for vertex in component}
        if not sources <= {"abt", "buy"}:
            raise DataValidationError(f"Unknown component sources {sources}")
        for vertex in component:
            if vertex in seen:
                raise DataValidationError(f"Repeated component vertex {vertex}")
            seen.add(vertex)
        if sources == {"abt", "buy"}:
            buckets[0].append(component)
        elif sources == {"abt"}:
            buckets[1].append(component)
        else:
            buckets[2].append(component)
    rng = random.Random(seed)
    memberships: dict[str, list[RecordKey]] = {name: [] for name in SPLIT_NAMES}
    for bucket in buckets:
        if not bucket:
            continue
        rng.shuffle(bucket)
        train_end = 3 * len(bucket) // 5
        validation_end = train_end + len(bucket) // 5
        allocations = (bucket[:train_end], bucket[train_end:validation_end], bucket[validation_end:])
        for name, allocated in zip(SPLIT_NAMES, allocations):
            for component in allocated:
                memberships[name].extend(component)
    return {name: tuple(sorted(vertices)) for name, vertices in memberships.items()}


def validate_partitions(records: tuple[RawRecord, ...], gold_pairs: tuple[GoldPair, ...],
                        components: list[Component], partitions: Partitions) -> dict:
    """Check exact coverage, unique membership, component integrity and every edge."""
    if set(partitions) != set(SPLIT_NAMES):
        raise DataValidationError(f"Expected partitions {SPLIT_NAMES}")
    expected_vertices = {record.key for record in records}
    owners = {}
    for name, vertices in partitions.items():
        for vertex in vertices:
            if vertex not in expected_vertices:
                raise DataValidationError(f"Unknown partition record {vertex}")
            if vertex in owners:
                raise DataValidationError(f"Record {vertex} appears multiple times in partitions")
            owners[vertex] = name
    if set(owners) != expected_vertices:
        raise DataValidationError(f"Missing partition records: {sorted(expected_vertices - set(owners))[:5]}")
    component_counts = dict.fromkeys(SPLIT_NAMES, 0)
    component_vertices = set()
    for component in components:
        if not component:
            raise DataValidationError("Empty connected component")
        for vertex in component:
            if vertex not in expected_vertices:
                raise DataValidationError(f"Unknown component record {vertex}")
            if vertex in component_vertices:
                raise DataValidationError(f"Repeated component vertex {vertex}")
            component_vertices.add(vertex)
        component_owners = {owners[vertex] for vertex in component}
        if len(component_owners) != 1:
            raise DataValidationError(f"Connected component crosses partitions: {component}")
        component_counts[next(iter(component_owners))] += 1
    if component_vertices != expected_vertices:
        raise DataValidationError("Connected components do not cover all records")
    gold_counts = dict.fromkeys(SPLIT_NAMES, 0)
    for abt_id, buy_id in gold_pairs:
        left, right = ("abt", abt_id), ("buy", buy_id)
        if left not in owners or right not in owners:
            raise DataValidationError(f"Nonexistent gold reference {(abt_id, buy_id)}")
        if owners[left] != owners[right]:
            raise DataValidationError(f"Gold edge crosses partitions: {(abt_id, buy_id)}")
        gold_counts[owners[left]] += 1
    if sum(gold_counts.values()) != len(gold_pairs):
        raise DataValidationError("Not all gold pairs accounted for")
    sizes = {}
    for name, vertices in partitions.items():
        sizes[name] = {
            "components": component_counts[name],
            "abt": sum(vertex[0] == "abt" for vertex in vertices),
            "buy": sum(vertex[0] == "buy" for vertex in vertices),
        }
    return {"partitions": sizes, "gold_pairs_by_partition": gold_counts}


def make_manifest(dataset: Dataset, components: list[Component],
                  partitions: Partitions) -> dict:
    """Construct metadata for synthetic fixtures; never overwrite the supplied file."""
    summary = validate_partitions(dataset.records, dataset.gold_pairs, components, partitions)
    splits = {}
    for name in SPLIT_NAMES:
        splits[name] = {
            "component_count": summary["partitions"][name]["components"],
            "abt_ids": sorted(vertex[1] for vertex in partitions[name] if vertex[0] == "abt"),
            "buy_ids": sorted(vertex[1] for vertex in partitions[name] if vertex[0] == "buy"),
        }
    return {"split_version": SPLIT_VERSION, "seed": 42,
            "raw_sha256": dataset.raw_sha256, "splits": splits}


def verify_manifest(dataset: Dataset, manifest: dict,
                    components: list[Component]) -> Partitions:
    """Reuse supplied memberships only after hash, leakage and reproduction checks."""
    if not isinstance(manifest, dict):
        raise DataValidationError("Manifest must be a JSON object")
    if manifest.get("split_version") != SPLIT_VERSION or type(manifest.get("seed")) is not int or manifest["seed"] != 42:
        raise DataValidationError("Unsupported manifest split_version or seed (expected seed 42)")
    if manifest.get("raw_sha256") != dataset.raw_sha256:
        raise DataValidationError(
            f"Manifest raw SHA-256 mismatch: expected {manifest.get('raw_sha256')}, got {dataset.raw_sha256}"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(SPLIT_NAMES):
        raise DataValidationError(f"Manifest must contain exactly {SPLIT_NAMES}")
    partitions = {}
    for name in SPLIT_NAMES:
        entry = splits[name]
        if not isinstance(entry, dict) or set(entry) != {"component_count", "abt_ids", "buy_ids"}:
            raise DataValidationError(f"Invalid manifest partition structure: {name}")
        if type(entry["component_count"]) is not int or entry["component_count"] < 0:
            raise DataValidationError(f"Invalid manifest component_count: {name}")
        vertices = []
        for source in ("abt", "buy"):
            ids = entry[f"{source}_ids"]
            if not isinstance(ids, list) or any(not isinstance(raw_id, str) for raw_id in ids):
                raise DataValidationError(f"Manifest {name} {source}_ids must be a list of strings")
            for raw_id in ids:
                vertices.append((source, raw_id))
        partitions[name] = tuple(sorted(vertices))
    summary = validate_partitions(dataset.records, dataset.gold_pairs, components, partitions)
    for name in SPLIT_NAMES:
        if splits[name]["component_count"] != summary["partitions"][name]["components"]:
            raise DataValidationError(f"Manifest component_count mismatch in {name}")
    reproduced = split_components(components)
    for name in SPLIT_NAMES:
        if partitions[name] != reproduced[name]:
            raise DataValidationError(f"Manifest assignments disagree with seed-42 reproduction in {name}")
    return partitions


def verify_original(raw_dir: Path, manifest_path: Path) -> tuple[Dataset, Partitions, dict]:
    """Single checked entry point shared by CLI commands and the real-data test."""
    dataset = load_dataset(raw_dir)
    report = audit_dataset(dataset)
    components = connected_components(build_positive_graph(dataset.records, dataset.gold_pairs))
    shapes: Counter[str] = Counter()
    for component in components:
        abt_count = sum(vertex[0] == "abt" for vertex in component)
        buy_count = len(component) - abt_count
        shapes[f"{abt_count}_abt_{buy_count}_buy"] += 1
    if len(components) != 1076 or dict(shapes) != EXPECTED_SHAPES:
        raise DataValidationError(f"Unexpected original component shapes/count: {dict(shapes)}, {len(components)}")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise DataValidationError("Manifest must be a JSON object")
    partitions = verify_manifest(dataset, manifest, components)
    split_report = validate_partitions(dataset.records, dataset.gold_pairs, components, partitions)
    if split_report["partitions"] != EXPECTED_SPLITS:
        raise DataValidationError(f"Unexpected original partition sizes: {split_report['partitions']}")
    report.update(split_report)
    report["component_count"] = len(components)
    report["component_shapes"] = dict(shapes)
    report["manifest_sha256"] = sha256_file(manifest_path)
    report["checks"] = {
        "schemas_ids_foreign_keys_and_gold_unique": True,
        "raw_hashes_match_manifest": True,
        "all_records_preserved_once": True,
        "no_component_or_gold_edge_leakage": True,
        "all_gold_pairs_accounted_for": True,
        "seed_42_reproduces_manifest": True,
    }
    return dataset, partitions, report


def partition_records(dataset: Dataset, partitions: Partitions,
                      split: str, source: str) -> tuple[RawRecord, ...]:
    """Return deterministic raw-ID ordering; positions can serve as dense offsets."""
    members = set(partitions[split])
    records = []
    for record in dataset.records:
        if record.source == source and record.key in members:
            records.append(record)
    return tuple(sorted(records, key=lambda record: record.raw_id))


def prepared_attributes(prepared: PreparedRecord) -> dict:
    """Canonical JSON representation shared by preparation and its checked reader."""
    return {
        "normalized_title": prepared.normalized_title,
        "normalized_description": prepared.normalized_description,
        "title_word_tokens": sorted(prepared.title_word_tokens),
        "description_word_tokens": sorted(prepared.description_word_tokens),
        "title_trigrams": sorted(prepared.title_trigrams),
        "model_tokens": sorted(prepared.model_tokens),
        "numeric_title_tokens": sorted(prepared.numeric_title_tokens),
        "price": prepared.price,
    }


def load_prepared_partition(processed_dir: Path, dataset: Dataset, partitions: Partitions,
                            verified_audit: dict, split: str, *,
                            allow_test: bool = False) -> tuple[list[PreparedRecord], list[PreparedRecord]]:
    """Check prepared provenance and positions against verify_original's outputs.

    Test attributes require an explicit opt-in from frozen evaluation. All rows'
    membership and raw attributes are checked, but only the requested partition
    is prepared and its cached attributes checked against the current normalizer.
    """
    if split not in ("train", "validation") and not (split == "test" and allow_test):
        raise DataValidationError("Retrieval loading supports only train and validation partitions unless frozen test evaluation opts in")
    audit = json.loads((processed_dir / "audit.json").read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise DataValidationError("Prepared audit must be a JSON object")
    if audit.get("normalizer_version") != NORMALIZER_VERSION:
        raise DataValidationError("Prepared normalizer version is missing or stale; rerun prepare")
    if (audit.get("raw_sha256") != dataset.raw_sha256
            or audit.get("manifest_sha256") != verified_audit["manifest_sha256"]):
        raise DataValidationError("Prepared raw hashes or split-manifest identity do not match verified inputs")
    output_hashes = audit.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise DataValidationError("Prepared output hashes are missing")
    for filename in ("records.jsonl", "split_memberships.json"):
        if sha256_file(processed_dir / filename) != output_hashes.get(filename):
            raise DataValidationError(f"Prepared SHA-256 mismatch: {filename}")
    manifest = json.loads((processed_dir / "split_memberships.json").read_text(encoding="utf-8"))
    components = connected_components(build_positive_graph(dataset.records, dataset.gold_pairs))
    if verify_manifest(dataset, manifest, components) != partitions:
        raise DataValidationError("Prepared memberships differ from the verified partitions")
    expected_metadata = {}
    originals = {record.key: record for record in dataset.records}
    for name in SPLIT_NAMES:
        for source in ("abt", "buy"):
            for position, record in enumerate(partition_records(dataset, partitions, name, source)):
                expected_metadata[record.key] = {"split": name, "position": position}
    seen = set()
    selected: dict[str, dict[int, PreparedRecord]] = {"abt": {}, "buy": {}}
    with (processed_dir / "records.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                raw = RawRecord(**row["raw"])
                metadata = row["metadata"]
                if raw.key in seen or raw.key not in originals or raw != originals[raw.key]:
                    raise DataValidationError("Duplicate, unknown, or modified raw record")
                if (metadata != expected_metadata[raw.key]
                        or type(metadata["position"]) is not int):
                    raise DataValidationError("Incorrect partition or dense position")
                seen.add(raw.key)
                if metadata["split"] == split:
                    prepared = prepare_record(raw)
                    if row["attributes"] != prepared_attributes(prepared):
                        raise DataValidationError("Prepared attributes differ from current normalization")
                    selected[raw.source][metadata["position"]] = prepared
            except (KeyError, TypeError, ValueError) as error:
                raise DataValidationError(f"Invalid prepared row {line_number}: {error}") from error
    if seen != set(originals):
        raise DataValidationError("Prepared records do not cover all verified records")
    left = [selected["abt"][position] for position in range(len(selected["abt"]))]
    right = [selected["buy"][position] for position in range(len(selected["buy"]))]
    return left, right
