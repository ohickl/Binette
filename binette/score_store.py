"""Atomic score shards in an authenticated input/code/model namespace."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

import numpy as np


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scoring_identity(evidence, bins, batch_size: int, weight: float) -> str:
    from checkm2.defaultValues import DefaultValues

    digest = hashlib.sha256()
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "checkm2",
            "numpy",
            "scipy",
            "keras",
            "tensorflow",
            "lightgbm",
            "scikit-learn",
            "pyroaring",
        )
    }
    digest.update(
        json.dumps(
            {
                "schema": "binette-score-shards-v1",
                "versions": versions,
                "batch_size": batch_size,
                "weight": weight,
            },
            sort_keys=True,
        ).encode()
    )
    # Hash the immutable evidence and executable/model boundary once, not rows
    # or internal numerical operations. Evidence already incorporates DB hits.
    for array in (
        evidence.ids,
        evidence.metadata,
        evidence.ko_counts.data,
        evidence.ko_counts.indices,
        evidence.ko_counts.indptr,
    ):
        digest.update(str((array.dtype.str, array.shape)).encode())
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    for candidate in bins:
        key = candidate.contigs_key
        digest.update(len(key).to_bytes(8, "little"))
        digest.update(key)
    source = Path(__file__).parent
    for name in ("features.py", "scoring.py", "bin_quality.py", "score_store.py"):
        digest.update(bytes.fromhex(file_digest(source / name)))
    for attribute in (
        "REF_DATA_LOCATION",
        "GENERAL_MODEL_COMP_LOCATION",
        "MODEL_CONT_LOCATION",
        "SPECIFIC_MODEL_COMP_LOCATION",
        "SCALER_FILE_LOCATION",
    ):
        digest.update(
            bytes.fromhex(file_digest(Path(getattr(DefaultValues, attribute))))
        )
    import checkm2

    package = Path(checkm2.__file__).parent
    for name in (
        "modelProcessing.py",
        "modelPostprocessing.py",
        "keggData.py",
        "defaultValues.py",
    ):
        digest.update(bytes.fromhex(file_digest(package / name)))
    # Definitions/order are executable feature inputs even with identical model bytes.
    from binette.features import feature_plan

    plan = feature_plan()
    digest.update(json.dumps(plan.columns).encode())
    for group, denominator in zip(plan.groups, plan.denominators, strict=True):
        for array in (group.data, group.indices, group.indptr, denominator):
            digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


class ScoreStore:
    def __init__(self, root: Path, identity: str):
        self.root = root / identity
        self.root.mkdir(parents=True, exist_ok=True)

    def load(self, start: int, count: int):
        path = self.root / f"{start}.npz"
        receipt = path.with_suffix(".json")
        if not receipt.exists():
            return None  # An unsealed file is never trusted.
        seal = json.loads(receipt.read_text())
        if seal != {"sha256": file_digest(path), "count": count}:
            raise ValueError(f"Corrupt score shard: {path}")
        with np.load(path, allow_pickle=False) as data:
            arrays = tuple(
                data[name] for name in ("completeness", "contamination", "model")
            )
        self.validate(arrays, count)
        return arrays

    @staticmethod
    def validate(arrays, count):
        if (
            len(arrays) != 3
            or count <= 0
            or any(array.shape != (count,) for array in arrays)
            or not all(np.isfinite(array).all() for array in arrays[:2])
            or arrays[2].dtype != np.uint8
            or np.any(arrays[2] > 1)
        ):
            raise ValueError("Invalid score shard arrays")

    def write(self, start: int, arrays) -> None:
        self.validate(arrays, len(arrays[0]))
        path = self.root / f"{start}.npz"
        # One lock per namespace avoids retaining an inode for every batch.
        with (self.root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            existing = self.load(start, len(arrays[0]))
            if existing is not None:
                if not all(
                    np.array_equal(a, b) for a, b in zip(existing, arrays, strict=True)
                ):
                    raise ValueError("Concurrent score writers disagree")
                return
            descriptor, name = tempfile.mkstemp(dir=self.root, suffix=".tmp")
            temporary = Path(name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.savez(
                        handle,
                        completeness=arrays[0],
                        contamination=arrays[1],
                        model=arrays[2],
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(path)
                seal = {"sha256": file_digest(path), "count": len(arrays[0])}
                descriptor, name = tempfile.mkstemp(dir=self.root, suffix=".tmp")
                temporary = Path(name)
                with os.fdopen(descriptor, "w") as handle:
                    json.dump(seal, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                temporary.replace(path.with_suffix(".json"))
                directory = os.open(self.root, os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                temporary.unlink(missing_ok=True)
