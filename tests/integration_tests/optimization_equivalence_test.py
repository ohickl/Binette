"""Bounded differential checks against the pinned upstream implementation.

These tests use the installed CheckM2 models and reference data, without a
DIAMOND database or external sequencing dataset. They require the fork's Git
history so the oracle is the audited implementation rather than a copy.
"""

from __future__ import annotations

import random
import subprocess
from collections import Counter
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import numpy as np
import pytest
from pyroaring import BitMap

from binette import bin_manager, bin_quality

BASELINE = "4f8791e9b411708b0717dbd8e1296a68ac72ae24"


def baseline_module(name: str) -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:binette/{name}.py"], cwd=root, text=True
    )
    module = ModuleType(f"baseline_{name}")
    exec(compile(source, f"{BASELINE}/{name}.py", "exec"), module.__dict__)
    return module


def test_candidate_memberships_and_selection_match_upstream():
    baseline = baseline_module("bin_manager")
    lengths = np.array([120_000, 170_000, 230_000, 90_000, 210_000, 110_000])

    def run(manager):
        original = {}
        for index, contigs in enumerate(([0, 1, 2], [1, 2, 3], [2, 3, 4], [5])):
            candidate = manager.Bin(
                BitMap(contigs), origin=f"set{index}", is_original=True
            )
            candidate.add_quality(90 - index, index, 2)
            original[candidate.contigs_key] = candidate
        bin_quality.add_bin_size_and_N50(original.values(), dict(enumerate(lengths)))
        hybrid = manager.create_intermediate_bins(
            original, lengths, 40, 10, 200_000, 10_000_000, True
        )
        for candidate in hybrid.values():
            candidate.add_quality(80, 2, 2)
        bin_quality.add_bin_size_and_N50(hybrid.values(), dict(enumerate(lengths)))
        selected = manager.select_best_bins(original | hybrid, 40, 10)
        return set(hybrid), [tuple(candidate.contigs) for candidate in selected]

    assert run(bin_manager) == run(baseline)


@pytest.mark.parametrize("seed", range(12))
def test_candidate_parent_gates_match_upstream(seed):
    baseline = baseline_module("bin_manager")
    rng = random.Random(seed)
    lengths = np.array([100_000, 200_000, 9_000_000, *([70_000] * 9)])
    inputs = [
        (
            rng.sample(range(12), rng.randint(1, 6)),
            rng.choice([0, 69, 70, 100]),
            rng.choice([0, 10, 11, 50]),
        )
        for _ in range(8)
    ]

    def run(manager):
        original = {}
        for members, completeness, contamination in inputs:
            candidate = manager.Bin(BitMap(members), is_original=True)
            candidate.add_quality(completeness, contamination, 2)
            original[candidate.contigs_key] = candidate
        bin_quality.add_bin_size_and_N50(original.values(), dict(enumerate(lengths)))
        candidates = manager.create_intermediate_bins(
            original, lengths, 70, 10, 200_000, 10_000_000, True
        )
        return set(candidates)

    assert run(bin_manager) == run(baseline)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_real_checkm2_scores_and_model_loads_match_upstream(batch_size):
    baseline = baseline_module("bin_quality")
    processing = bin_quality.get_modelProcessing()
    baseline.get_modelProcessing()
    ko_names = list(
        bin_quality.keggData.KeggCalculator().return_default_values_from_category(
            "KO_Genes"
        )
    )
    kegg = {
        i: Counter({ko_names[j]: (i + j) % 3 + 1 for j in range(i, i + 12)})
        for i in range(4)
    }
    counts = {i: 30 + i for i in range(4)}
    amino = {
        i: Counter({aa: 100 + i for aa in "ACDEFGHIKLMNPQRSTVWY"}) for i in range(4)
    }
    aa_lengths = {i: sum(amino[i].values()) for i in range(4)}

    def score(quality):
        bins = [
            bin_manager.Bin(BitMap(contigs))
            for contigs in ([0, 1], [1, 2], [2, 3], [3])
        ]
        raw_predictions = []
        processor_class = processing.modelProcessor
        general = processor_class.run_prediction_general
        specific = processor_class.run_prediction_specific

        def record_general(processor, vectors):
            result = general(processor, vectors)
            raw_predictions.extend(array.copy() for array in result)
            return result

        def record_specific(processor, vectors, length):
            result = specific(processor, vectors, length)
            raw_predictions.extend(array.copy() for array in result)
            return result

        with (
            patch.object(processor_class, "run_prediction_general", record_general),
            patch.object(processor_class, "run_prediction_specific", record_specific),
            patch.object(
                processing, "modelProcessor", wraps=processor_class
            ) as constructor,
        ):
            quality.assess_bins_quality(
                bins, kegg, counts, amino, aa_lengths, 2, batch_size, threads=1
            )
        result = [
            (b.contigs_key, b.completeness, b.contamination, b.score, b.checkm2_model)
            for b in bins
        ]
        return result, constructor.call_count, raw_predictions

    expected, baseline_loads, baseline_raw = score(baseline)
    actual, optimized_loads, optimized_raw = score(bin_quality)
    assert actual == expected
    assert len(optimized_raw) == len(baseline_raw)
    for actual_array, expected_array in zip(optimized_raw, baseline_raw, strict=True):
        np.testing.assert_array_equal(actual_array, expected_array)
    assert baseline_loads == (4 + batch_size - 1) // batch_size
    assert optimized_loads == 1


@pytest.mark.parametrize("workers,batch_size", [(1, 1), (2, 2), (4, 4)])
def test_optimized_numeric_and_spawn_scores_match_upstream(
    workers, batch_size, tmp_path
):
    from binette.features import ContigEvidence

    kos = list(
        bin_quality.keggData.KeggCalculator().return_default_values_from_category(
            "KO_Genes"
        )
    )
    kegg = {
        i: Counter({kos[j]: (i + j) % 3 + 1 for j in range(i, i + 12)})
        for i in range(4)
    }
    counts = {i: 30 + i for i in range(4)}
    amino = {
        i: Counter({aa: 100 + i for aa in "ACDEFGHIKLMNPQRSTVWY"}) for i in range(4)
    }
    lengths = {i: sum(amino[i].values()) for i in range(4)}
    memberships = ([0, 1], [1, 2], [2, 3], [3], [0, 2], [0, 1, 3], [99], [0, 2, 99])
    expected = [bin_manager.Bin(BitMap(item)) for item in memberships]
    baseline_module("bin_quality").assess_bins_quality(
        expected, kegg, counts, amino, lengths, 2, 8, threads=1
    )
    actual = [bin_manager.Bin(BitMap(item)) for item in memberships]
    info = {"_evidence": ContigEvidence.from_dicts(kegg, counts, amino, lengths)}
    cache = tmp_path / "scores"
    result = bin_quality.add_bin_metrics(
        actual,
        info,
        2,
        threads=workers,
        quality_workers=workers,
        checkm2_batch_size=batch_size,
        score_cache=cache,
        disable_progress_bar=True,
    )
    assert result is actual
    assert [
        (b.contigs_key, b.completeness, b.contamination, b.score, b.checkm2_model)
        for b in actual
    ] == [
        (b.contigs_key, b.completeness, b.contamination, b.score, b.checkm2_model)
        for b in expected
    ]
    # Sealed score recovery never starts a model owner.
    recovered = [bin_manager.Bin(BitMap(item)) for item in memberships]
    with patch(
        "binette.scoring.make_processors",
        side_effect=AssertionError("unexpected inference"),
    ):
        bin_quality.add_bin_metrics(
            recovered,
            info,
            2,
            threads=workers,
            quality_workers=workers,
            checkm2_batch_size=batch_size,
            score_cache=cache,
            disable_progress_bar=True,
        )
    assert [(b.score, b.checkm2_model) for b in recovered] == [
        (b.score, b.checkm2_model) for b in expected
    ]


def test_vectorized_model_choice_matches_upstream_boundaries_and_nonfinite():
    from types import SimpleNamespace

    from scipy import sparse

    from binette.scoring import BoundedPostprocessor

    upstream = bin_quality.get_modelPostprocessing().modelProcessor(1)
    general = np.array(
        [0.0, 40.0, 50.0, 55.0, 60.0, 70.0, 80.0, 90.0, 100.0, np.nan, np.inf, 20.0]
    )
    specific = general.copy()
    cosine = np.array([0.0, 0.4, 0.8, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, np.nan, 1.0, 0.1])
    amino = np.array(
        [0.0, 0.0, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 1e10, 0.0, 0.0, 1.0]
    )
    adapter = BoundedPostprocessor(
        SimpleNamespace(
            ref_data=sparse.eye(1, format="csr"), reduced_cutoff=upstream.reduced_cutoff
        )
    )
    adapter.maximum_cosine = lambda features: cosine
    with np.errstate(divide="ignore", invalid="ignore"):
        actual, _, chosen, _ = adapter.calculate_general_specific_ratio(
            amino, np.zeros((len(general), 1)), general, general, specific
        )
        expected = [
            upstream.cosine_decider(
                {
                    "General": g,
                    "Specific": s,
                    "Cosine_Similarity": c,
                    "AA_Ratio": aa / ((g + s) / 2),
                }
            )
            for g, s, c, aa in zip(general, specific, cosine, amino, strict=True)
        ]
    np.testing.assert_array_equal(actual, [row[0] for row in expected])
    assert list(chosen) == [row[1] for row in expected]


def test_occupied_bitmap_selection_and_streamed_reports_match_upstream(tmp_path):
    import pickle

    from binette import io_manager

    rng = random.Random(20261001)
    upstream = baseline_module("bin_manager")
    upstream_io = baseline_module("io_manager")
    for case in range(30):
        candidates = {}
        for index in range(40):
            item = bin_manager.Bin(
                BitMap(rng.sample(range(40), rng.randint(1, 12))),
                origin="raw" if index % 2 else None,
                is_original=bool(index % 2),
            )
            item.add_quality(
                rng.choice([40.0, 70.0, 80.0, 90.0]),
                rng.choice([0.0, 5.0, 10.0, 11.0]),
                2,
            )
            item.add_length(1000)
            item.add_N50(rng.choice([1, 10, 20]))
            item.add_model("Gradient Boost (General Model)")
            item.coding_density = 0.55 if index % 2 else None
            candidates[item.contigs_key] = item
        expected = upstream.select_best_bins(
            pickle.loads(pickle.dumps(candidates)), 70, 10
        )
        actual = bin_manager.select_best_bins(candidates, 70, 10)
        assert [b.contigs_key for b in actual] == [b.contigs_key for b in expected]
        assert [(b.name, b.original_name, b.origin) for b in actual] == [
            (b.name, b.original_name, b.origin) for b in expected
        ]
        left, right = tmp_path / f"old-{case}.tsv", tmp_path / f"new-{case}.tsv"
        upstream_io.write_bin_info(expected, left, add_contigs=True)
        io_manager.write_bin_info(actual, right, add_contigs=True)
        assert left.read_bytes() == right.read_bytes()
