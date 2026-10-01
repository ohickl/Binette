"""Differential controls for changes that preserve candidate and score semantics."""

from __future__ import annotations

import itertools
import random
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
import pytest
from pyroaring import BitMap

from binette import bin_manager, bin_quality, cds


@pytest.mark.xfail(
    strict=True, reason="Known exact-half N50 defect; separate semantic fix"
)
def test_n50_exact_half_boundary():
    assert bin_quality.compute_N50([1, 1, 2]) == 2


def test_duplicate_membership_retains_all_sources():
    sources = {
        "first": [{"contigs": ["a", "b"], "bin_name": "one"}],
        "second": [{"contigs": ["b", "a"], "bin_name": "two"}],
    }
    bins = bin_manager.make_bins_from_bins_info(sources, {"a": 0, "b": 1}, True)
    assert len(bins) == 1
    retained = next(iter(bins.values()))
    assert retained.origin == {"first", "second"}
    assert retained.name == "one"
    assert retained.is_original


def test_overlap_graph_matches_pairwise_oracle():
    rng = random.Random(20261001)
    for _ in range(20):
        bins = [
            bin_manager.Bin(BitMap(rng.sample(range(60), rng.randint(1, 12))))
            for _ in range(18)
        ]
        bins.append(bin_manager.Bin(BitMap([100])))
        expected = {
            frozenset((left.contigs_key, right.contigs_key))
            for left, right in itertools.combinations(bins, 2)
            if left.overlaps_with(right)
        }
        graph = bin_manager.from_bins_to_bin_graph(iter(bins))
        assert {frozenset(edge) for edge in graph.edges} == expected
        assert bins[-1].contigs_key not in graph


def test_unique_subsets_match_maximal_clique_oracle():
    rng = random.Random(20261001)
    for size in range(2, 11):
        graph = nx.Graph()
        graph.add_nodes_from(range(size))
        graph.add_edges_from(
            pair
            for pair in itertools.combinations(range(size), 2)
            if rng.random() < 0.6
        )
        expected = {
            tuple(sorted(subset))
            for clique in nx.find_cliques(graph)
            for subset in bin_manager.get_all_possible_combinations(sorted(clique))
        }
        actual = list(bin_manager.iter_unique_bin_combinations(graph))
        assert set(actual) == expected
        assert len(actual) == len(expected)


def test_shared_cliques_do_not_repeat_their_common_pair():
    graph = nx.Graph([(0, 1), (1, 2), (0, 2), (0, 3), (1, 3)])
    actual = list(bin_manager.iter_unique_bin_combinations(graph))
    assert actual.count((0, 1)) == 1
    assert set(actual) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (0, 1, 2), (0, 1, 3)}


def test_prediction_processor_is_reused_across_batches():
    bins = [bin_manager.Bin(BitMap([index])) for index in range(5)]
    processor = object()
    received = []

    def assess(batch, *args, modelProc=None):
        received.append(modelProc)
        return list(batch)

    with (
        patch.object(bin_quality, "get_modelProcessing") as loader,
        patch.object(bin_quality, "_assess_bins_quality_batch", side_effect=assess),
    ):
        loader.return_value = SimpleNamespace(modelProcessor=lambda threads: processor)
        result = bin_quality.assess_bins_quality(
            bins, {}, {}, {}, {}, 2.0, 2, postProcessor=object()
        )
    assert result == bins
    assert received == [processor, processor, processor]
    loader.assert_called_once()


def test_empty_scoring_does_not_load_models():
    with (
        patch.object(bin_quality, "get_modelProcessing") as prediction,
        patch.object(bin_quality, "get_modelPostprocessing") as post,
    ):
        assert bin_quality.assess_bins_quality([], {}, {}, {}, {}, 2.0, 2) == []
    prediction.assert_not_called()
    post.assert_not_called()


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_fails_before_loading_models(batch_size):
    with patch.object(bin_quality, "get_modelProcessing") as loader:
        with pytest.raises(ValueError, match="must be positive"):
            bin_quality.assess_bins_quality(
                [bin_manager.Bin(BitMap([0]))], {}, {}, {}, {}, 2.0, batch_size
            )
    loader.assert_not_called()


def test_metadata_pool_bounds_futures_and_preserves_input_order():
    genes = {index: ["ACD*" * (index + 1), "WWG"] for index in reversed(range(19))}
    expected = cds.get_contig_cds_metadata_flat(genes)
    pending_counts = []
    original_wait = cds.cf.wait

    def measured_wait(futures, **kwargs):
        pending_counts.append(len(futures))
        return original_wait(futures, **kwargs)

    with patch.object(cds.cf, "wait", side_effect=measured_wait):
        actual = cds.get_contig_cds_metadata(genes, threads=2)
    assert tuple(actual.values()) == expected
    assert list(actual["contig_to_aa_counter"]) == list(genes)
    assert pending_counts
    assert max(pending_counts) == 4


def test_metadata_pool_propagates_worker_failure():
    with pytest.raises(TypeError):
        cds.get_contig_cds_metadata({0: [42]}, threads=1)
