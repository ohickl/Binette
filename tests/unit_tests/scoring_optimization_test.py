"""Controls for bounded numerical work, dtype safety and durable score shards."""

from __future__ import annotations

import pickle
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
from pyroaring import BitMap
from scipy import sparse

from binette import bin_quality, cds, diamond
from binette.bin_manager import Bin
from binette.features import ContigEvidence, feature_plan
from binette.score_store import ScoreStore
from binette.scoring import BoundedPostprocessor


def evidence():
    kos = feature_plan().kos
    return ContigEvidence.from_dicts(
        {0: Counter({kos[0]: 3}), 3: Counter({kos[1]: 2})},
        {0: 4, 3: 7},
        {0: Counter(A=5, X=1), 3: Counter(C=8)},
        {0: 6, 3: 8},
    )


def test_typed_evidence_matches_dataframe_features_and_mmap(tmp_path):
    kos = feature_plan().kos
    kegg = {0: Counter({kos[0]: 3}), 3: Counter({kos[1]: 2})}
    counts = {0: 4, 3: 7}
    amino = {0: Counter(A=5, X=1), 3: Counter(C=8)}
    lengths = {0: 6, 3: 8}
    bins = [Bin(BitMap([0, 3])), Bin(BitMap([3, 90])), Bin(BitMap([99]))]
    metadata = bin_quality.get_bins_metadata_df(bins, counts, amino, lengths)
    features, _ = bin_quality.get_diamond_feature_per_bin_df(bins, kegg)
    expected = np.concatenate(
        (metadata.iloc[:, 1:].values, features.drop(columns="Name").values), axis=1
    ).astype(float)
    actual = evidence()
    assert actual.metadata.dtype == np.uint32
    np.testing.assert_array_equal(actual.vectors([b.contigs for b in bins]), expected)
    actual.save(tmp_path)
    restored = ContigEvidence.load(tmp_path)
    assert not restored.metadata.flags.writeable
    np.testing.assert_array_equal(restored.vectors([b.contigs for b in bins]), expected)
    assert restored.vectors([]).shape == (0, len(feature_plan().columns))


def test_counts_promote_and_reject_overflow():
    value = 2**32 + 1
    item = ContigEvidence.from_dicts({}, {0: value}, {}, {})
    assert item.metadata.dtype == np.int64
    assert item.vectors([BitMap([0])])[0, -1] == 0  # last feature remains zero
    assert item.vectors([BitMap([0])])[0, feature_plan().metadata.index("CDS")] == value
    with pytest.raises(OverflowError):
        ContigEvidence.from_dicts({}, {0: 2**63}, {}, {})
    with pytest.raises(ValueError):
        ContigEvidence.from_dicts({}, {0: -1}, {}, {})


@pytest.mark.parametrize("block_rows", [1, 2, 7, 50])
def test_blocked_cosine_matches_dense_exactly(block_rows):
    rng = np.random.default_rng(91)
    reference = sparse.csr_matrix(rng.random((17, 13)))
    reference[0] = 0
    features = rng.random((8, 13))
    features[1] = 0
    adapter = BoundedPostprocessor(
        SimpleNamespace(ref_data=reference, reduced_cutoff=1), block_rows
    )
    batch = sparse.csr_matrix(features)
    norm_a = np.sqrt(reference.multiply(reference).sum(axis=1)).A.flatten()
    norm_b = np.sqrt(batch.multiply(batch).sum(axis=1)).A.flatten()
    norm_a[norm_a == 0] = norm_b[norm_b == 0] = 1
    expected = np.amax(
        reference.dot(batch.T).toarray() / (norm_a[:, None] * norm_b[None, :]), axis=0
    )
    np.testing.assert_array_equal(adapter.maximum_cosine(features), expected)


def test_metadata_does_not_scan_unrelated_contigs():
    class LookupOnly(dict):
        def items(self):
            raise AssertionError("whole-assembly traversal")

    result = bin_quality.get_bins_metadata_df(
        [Bin(BitMap([0]))],
        LookupOnly({0: 2}),
        LookupOnly({0: Counter(A=3)}),
        LookupOnly({0: 3}),
    )
    assert result["CDS"].iloc[0] == 2


def test_slots_pickle_key_alias_and_legacy_state():
    candidate = Bin(BitMap([0, 3]))
    candidate.add_model("Gradient Boost (General Model)")
    key = candidate.contigs_key
    restored = pickle.loads(pickle.dumps({key: candidate}))
    stored = next(iter(restored.values()))
    assert next(iter(restored)) is stored.contigs_key
    assert not hasattr(stored, "__dict__")
    assert stored.origin == frozenset()
    legacy = candidate.__getstate__()
    legacy["contigs_key"] = legacy.pop("_contigs_key")
    recreated = Bin.__new__(Bin)
    recreated.__setstate__(legacy)
    assert recreated.contigs_key == key


def test_shards_restore_partial_work_reject_corruption_and_isolate_identity(tmp_path):
    store = ScoreStore(tmp_path, "first")
    arrays = (
        np.array([90.0, 80.0]),
        np.array([1.0, 2.0]),
        np.array([0, 1], dtype=np.uint8),
    )
    store.write(0, arrays)
    for actual, expected in zip(store.load(0, 2), arrays, strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert store.load(2, 1) is None
    assert ScoreStore(tmp_path, "second").load(0, 2) is None
    store.write(0, arrays)
    with pytest.raises(ValueError, match="disagree"):
        store.write(0, (arrays[0] + 1, *arrays[1:]))
    with (store.root / "0.npz").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="Corrupt"):
        store.load(0, 2)


def test_coding_interval_union_matches_mask_randomized():
    rng = np.random.default_rng(7)
    for _ in range(100):
        genes = [
            SimpleNamespace(begin=int(start), end=int(end), strand=(-1) ** index)
            for index, (start, end) in enumerate(rng.integers(1, 101, size=(10, 2)))
        ]
        mask = np.zeros(100, dtype=bool)
        for gene in genes:
            mask[gene.begin - 1 : gene.end] = True
        assert cds.get_contig_coding_len(genes, 100) == int(mask.sum())


def test_chunked_diamond_retains_annotations_and_contig_names(tmp_path, monkeypatch):
    path = tmp_path / "hits.tsv"
    path.write_text(
        "contig_with_underscore_1\tr~K01810\ncontig_with_underscore_2\tr~K01810\nother_1\tr~bad\n"
    )
    original = diamond.pd.read_csv

    def small_chunks(*args, **kwargs):
        kwargs["chunksize"] = 1
        return original(*args, **kwargs)

    monkeypatch.setattr(diamond.pd, "read_csv", small_chunks)
    assert diamond.get_contig_to_kegg_id(str(path)) == {
        "contig_with_underscore": Counter(K01810=2)
    }


def test_membership_batches_bound_incidence_and_preserve_order():
    bins = [SimpleNamespace(contigs=range(size)) for size in (3, 4, 8, 20, 2, 3)]
    batches = list(bin_quality.membership_batches(bins, 3, 10))
    assert [(start, len(batch)) for start, batch in batches] == [
        (0, 2),
        (2, 1),
        (3, 1),
        (4, 2),
    ]
    assert [item for _, batch in batches for item in batch] == bins


@pytest.mark.parametrize(
    "arrays",
    [
        (np.array([np.nan]), np.array([1.0]), np.array([0], dtype=np.uint8)),
        (np.array([90.0]), np.array([1.0]), np.array([2], dtype=np.uint8)),
        (np.array([90.0]), np.array([1.0, 2.0]), np.array([0], dtype=np.uint8)),
    ],
)
def test_invalid_scores_never_receive_a_seal(tmp_path, arrays):
    store = ScoreStore(tmp_path, "invalid")
    with pytest.raises(ValueError, match="Invalid score shard"):
        store.write(0, arrays)
    assert not (store.root / "0.json").exists()


def test_partial_score_restore_infers_only_missing_batch(tmp_path, monkeypatch):
    from binette import score_store, scoring

    monkeypatch.setattr(score_store, "scoring_identity", lambda *args: "fixed")
    calls = []
    monkeypatch.setattr(scoring, "make_processors", lambda *args: (None, None))

    def predict(*args):
        calls.append(len(args[1]))
        return (
            np.full(len(args[1]), 90.0),
            np.full(len(args[1]), 1.0),
            np.zeros(len(args[1]), dtype=np.uint8),
        )

    monkeypatch.setattr(scoring, "predict_batch", predict)
    saved = (
        np.array([80.0, 85.0]),
        np.array([2.0, 3.0]),
        np.array([0, 1], dtype=np.uint8),
    )
    ScoreStore(tmp_path, "fixed").write(0, saved)
    bins = [Bin(BitMap([index])) for index in range(3)]
    actual = bin_quality.add_bin_metrics(
        bins,
        {"_evidence": evidence()},
        2,
        checkm2_batch_size=2,
        score_cache=tmp_path,
        disable_progress_bar=True,
    )
    assert calls == [1]
    assert actual is bins
    assert [item.completeness for item in bins] == [80.0, 85.0, 90.0]


def test_protein_summary_preserves_written_sequences_and_metadata(tmp_path):
    import gzip

    sequences = [("contig_one", "ATG" + "GCT" * 600 + "TAA"), ("no_gene", "N" * 100)]
    first, second = tmp_path / "legacy.faa.gz", tmp_path / "summary.faa.gz"
    proteins, coding = cds.predict(iter(sequences), str(first), threads=2)
    summaries, compact_coding = cds.predict(
        iter(sequences), str(second), threads=2, summarize=True
    )
    assert coding == compact_coding
    assert gzip.decompress(first.read_bytes()) == gzip.decompress(second.read_bytes())
    for name, values in proteins.items():
        amino = cds.get_aa_composition(values)
        assert summaries[name] == cds.ProteinSummary(
            len(values), amino, sum(amino.values())
        )
