"""Paired measurement of the original and optimized DP on identical prepared inputs."""
import importlib.util
from pathlib import Path
from time import perf_counter
import numpy as np
from catalogmatch import similarity
from catalogmatch.data import verify_original, load_prepared_partition, sha256_file
from catalogmatch.features import feature_matrix
from catalogmatch.pipeline import load_bundle, build_catalog_indexes, candidate_pairs, write_json
from catalogmatch.retrieval import RetrievalConfig
from catalogmatch.evaluate import timing_summary

original_path = Path('runs/m6/baseline_source/similarity.py')
spec = importlib.util.spec_from_file_location('baseline_similarity', original_path)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
implementations = {'before': original.levenshtein, 'after': similarity.levenshtein}
bundle = load_bundle(Path('runs/main/custom_forest.json'))
config = RetrievalConfig(**bundle.metadata['retrieval_config'])
dataset, partitions, audit = verify_original(Path('data/raw'), Path('CatalogMatch_Split_Manifest.json'))
report = {'before_source_sha256': sha256_file(original_path),
          'after_source_sha256': sha256_file(Path('src/catalogmatch/similarity.py')),
          'method': 'one uncached warmup per implementation; five paired passes with alternating implementation order; same prepared records, ordered pairs, model, threshold; equality checks outside timing',
          'partitions': {}}
try:
    for split in ('train', 'validation'):
        left, right = load_prepared_partition(Path('data/processed'), dataset, partitions, audit, split)
        pairs = candidate_pairs(left, build_catalog_indexes(right, config), config)
        with np.load(Path('runs/main/features') / f'{split}.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(pairs, saved['pairs'])
            expected = saved['X']
        scores = bundle.score(expected)
        samples = {name: [] for name in implementations}
        for name, function in implementations.items():
            similarity.levenshtein = function
            np.testing.assert_array_equal(feature_matrix(pairs, left, right), expected)
        for timing_pass in range(5):
            order = ('before', 'after') if timing_pass % 2 == 0 else ('after', 'before')
            for name in order:
                similarity.levenshtein = implementations[name]
                start = perf_counter()
                actual = feature_matrix(pairs, left, right)
                samples[name].append(perf_counter() - start)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(bundle.score(actual), scores)
        measurements = {name: timing_summary(values) for name, values in samples.items()}
        report['partitions'][split] = {'pairs': len(pairs), 'shape': list(expected.shape),
            'timings': measurements, 'speedup': measurements['before']['median_seconds'] / measurements['after']['median_seconds'],
            'exact_candidates_features_scores_decisions_and_accepted_pairs': True}
finally:
    similarity.levenshtein = implementations['after']
write_json(Path('runs/m6/paired_dp.json'), report)
print(report)
