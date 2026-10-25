"""Small local products: no test downloads or original rows are invented."""

from pathlib import Path

import pytest

from catalogmatch.data import (
    build_positive_graph, connected_components, load_dataset,
    make_manifest, split_components,
)


@pytest.fixture
def fixture_dir():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def dataset(fixture_dir):
    return load_dataset(fixture_dir)


@pytest.fixture
def components(dataset):
    return connected_components(build_positive_graph(dataset.records, dataset.gold_pairs))


@pytest.fixture
def partitions(components):
    return split_components(components)


@pytest.fixture
def manifest(dataset, components, partitions):
    return make_manifest(dataset, components, partitions)
