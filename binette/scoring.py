"""Versioned CheckM2 adapter with bounded reference similarity and worker state."""

from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from binette.features import ContigEvidence, feature_plan


def telemetry(phase: str, seconds: float, **details) -> None:
    path = os.environ.get("BINETTE_SCORING_TELEMETRY")
    if not path:
        return
    memory = {}
    try:
        for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
            if line.startswith(("Rss:", "Pss:", "Pss_Anon:", "Private_Dirty:")):
                key, value, _ = line.split()
                memory[key.rstrip(":")] = int(value) * 1024
    except (OSError, ValueError):
        pass
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "phase": phase,
                    "seconds": seconds,
                    "memory": memory,
                    **details,
                }
            )
            + "\n"
        )


class BoundedPostprocessor:
    def __init__(self, upstream, block_rows: int = 1024):
        if block_rows <= 0:
            raise ValueError("Reference block size must be positive")
        self.reference = upstream.ref_data[:, :20021].tocsr()
        self.norms = np.sqrt(
            self.reference.multiply(self.reference).sum(axis=1)
        ).A.flatten()
        self.norms[self.norms == 0] = 1
        self.reduced_cutoff = upstream.reduced_cutoff
        self.block_rows = block_rows

    @classmethod
    def shared(cls, root: Path):
        from checkm2.defaultValues import DefaultValues

        obj = cls.__new__(cls)
        arrays = [
            np.load(root / f"reference_{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("data", "indices", "indptr", "norms", "shape")
        ]
        obj.reference = sparse.csr_matrix(
            tuple(arrays[:3]), shape=tuple(arrays[4]), copy=False
        )
        obj.norms = arrays[3]
        obj.reduced_cutoff = DefaultValues.AA_RATIO_COMPLETENESS_CUTOFF
        obj.block_rows = 1024
        return obj

    def maximum_cosine(self, features: np.ndarray) -> np.ndarray:
        batch = sparse.csr_matrix(features[:, :20021])
        norms = np.sqrt(batch.multiply(batch).sum(axis=1)).A.flatten()
        norms[norms == 0] = 1
        transpose = batch.T.tocsr()
        maxima = np.full(len(features), -np.inf)
        for start in range(0, self.reference.shape[0], self.block_rows):
            stop = start + self.block_rows
            numerators = self.reference[start:stop].dot(transpose).toarray()
            # Same product and division as upstream; two successive divisions
            # would introduce different rounding near model-choice thresholds.
            denominator = self.norms[start:stop, None] * norms[None, :]
            np.divide(numerators, denominator, out=numerators)
            np.maximum(maxima, np.amax(numerators, axis=0), out=maxima)
        return maxima

    def calculate_general_specific_ratio(
        self,
        AA_counts,
        features,
        general,
        contamination,
        specific,
        *,
        decision_only=False,
    ):
        mean = (general + specific) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            aa_ratio = AA_counts / mean
        reduced = (mean < 55) & (aa_ratio < self.reduced_cutoff)
        if decision_only:
            # Reduced candidates and the final mean<=40/NaN branch choose a
            # model without consulting similarity. Keep full diagnostics by
            # default; prediction consumes only the quality/model columns.
            needed = ~reduced & (mean > 40)
            cosine = np.full(len(features), np.nan)
            if np.any(needed):
                cosine[needed] = self.maximum_cosine(
                    features if np.all(needed) else features[needed]
                )
            telemetry("cosine_rows", 0, bins=len(features), needed=int(needed.sum()))
        else:
            cosine = self.maximum_cosine(features)
        with np.errstate(divide="ignore", invalid="ignore"):
            novelty = general / (cosine**2)
        cutoff = np.select(
            [mean > 90, mean > 80, mean > 70, mean > 60, mean > 50, mean > 40],
            [160, 165, 165, 170, 175, 175],
            default=np.nan,
        )
        # Upstream's final else selects specific, including a NaN mean.
        use_specific = ~reduced & ((~(mean > 40)) | (novelty < cutoff))
        chosen = np.where(
            use_specific,
            "Neural Network (Specific Model)",
            "Gradient Boost (General Model)",
        )
        return np.where(use_specific, specific, general), contamination, chosen, cosine


def make_processors(threads: int = 1, reference_root: Path | None = None):
    if importlib.metadata.version("checkm2") != "1.1.0":
        raise RuntimeError("Optimized scoring requires pinned CheckM2 1.1.0")
    from binette.bin_quality import get_modelPostprocessing, get_modelProcessing

    prediction = get_modelProcessing().modelProcessor(threads)
    post = (
        BoundedPostprocessor.shared(reference_root)
        if reference_root is not None
        else BoundedPostprocessor(get_modelPostprocessing().modelProcessor(threads))
    )
    return prediction, post


def predict_batch(evidence, memberships, prediction, post):
    start = time.perf_counter()
    vectors = evidence.vectors(memberships)
    telemetry(
        "features",
        time.perf_counter() - start,
        bins=len(memberships),
        bytes=vectors.nbytes,
    )
    start = time.perf_counter()
    general, contamination = prediction.run_prediction_general(vectors)
    telemetry("general", time.perf_counter() - start)
    start = time.perf_counter()
    specific, scaled = prediction.run_prediction_specific(
        vectors, len(feature_plan().metadata) + len(feature_plan().kos)
    )
    telemetry("specific", time.perf_counter() - start)
    start = time.perf_counter()
    completeness, contamination, models, _ = post.calculate_general_specific_ratio(
        vectors[:, 20],
        scaled,
        general,
        contamination,
        specific,
        decision_only=True,
    )
    telemetry("postprocessing", time.perf_counter() - start)
    return (
        np.round(completeness, 2),
        np.round(contamination, 2),
        np.asarray(models == "Neural Network (Specific Model)", dtype=np.uint8),
    )


_worker_state = None
_worker_thread_limit = None


def initialize_worker(root: str) -> None:
    global _worker_state, _worker_thread_limit
    from threadpoolctl import threadpool_limits

    # A fresh spawn owns its model/runtime state. Evidence is read-only mmap.
    _worker_state = (ContigEvidence.load(Path(root)), *make_processors(1, Path(root)))
    _worker_thread_limit = threadpool_limits(limits=1)
    telemetry("worker_initialized", 0)


def save_reference(root: Path) -> None:
    from checkm2.defaultValues import DefaultValues

    reference = sparse.load_npz(DefaultValues.REF_DATA_LOCATION)[:, :20021].tocsr()
    norms = np.sqrt(reference.multiply(reference).sum(axis=1)).A.flatten()
    norms[norms == 0] = 1
    for name, array in (
        ("data", reference.data),
        ("indices", reference.indices),
        ("indptr", reference.indptr),
        ("norms", norms),
        ("shape", np.asarray(reference.shape)),
    ):
        np.save(root / f"reference_{name}.npy", array, allow_pickle=False)


def score_worker(task):
    from pyroaring import BitMap

    ordinal, keys = task
    evidence, prediction, post = _worker_state
    memberships = [BitMap.deserialize(key) for key in keys]
    order = sorted(range(len(keys)), key=keys.__getitem__)
    values = predict_batch(
        evidence, [memberships[index] for index in order], prediction, post
    )
    inverse = np.argsort(order)
    return ordinal, tuple(value[inverse] for value in values)
