"""Instrument the unmodified CLI contract for isolated compute-node profiling.

Phase receipts survive interruption. Candidate fingerprints are order-independent
SHA-256 sums over serialized memberships, compared only with identical ID maps.
Fingerprint overhead is recorded separately from candidate generation.
"""

import functools
import gzip
import hashlib
import importlib.metadata
import json
import os
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ["BINETTE_PROFILE_SOURCE"])
from binette import bin_manager, bin_quality, cds  # noqa: E402
from binette import main as cli  # noqa: E402

RECEIPT = Path(os.environ["BINETTE_PROFILE_RECEIPT"])


def emit(record):
    with RECEIPT.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def instrument(module, name):
    original = getattr(module, name)
    calls = 0

    @functools.wraps(original)
    def measured(*args, **kwargs):
        nonlocal calls
        calls += 1
        start = time.monotonic()
        emit({"event": "start", "phase": name, "monotonic": start})
        # A cache belongs to one immutable source/input/parameter namespace.
        # In-place operations cannot be restored by replacing a return value.
        cacheable = name in {
            "manage_protein_alignement",
            "get_contig_cds_metadata",
            "create_intermediate_bins",
            "add_bin_metrics",
        }
        cache = RECEIPT.parent / f"{name}.{calls}.pickle.gz"
        restored = cacheable and cache.exists()
        if restored:
            with gzip.open(cache, "rb") as handle:
                result = pickle.load(handle)
        else:
            result = original(*args, **kwargs)
        emit(
            {
                "event": "end",
                "phase": name,
                "wall_seconds": time.monotonic() - start,
                "restored": restored,
            }
        )
        if cacheable and not restored:
            checkpoint_start = time.monotonic()
            temporary = cache.with_suffix(cache.suffix + ".tmp")
            with gzip.open(temporary, "wb", compresslevel=1) as handle:
                pickle.dump(result, handle, protocol=5)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            temporary.replace(cache)
            emit(
                {
                    "event": "checkpoint",
                    "phase": name,
                    "bytes": cache.stat().st_size,
                    "wall_seconds": time.monotonic() - checkpoint_start,
                }
            )
        if name == "parse_input_files":
            emit(
                {
                    "event": "identity",
                    "original_bins": len(result[0]),
                    "id_map_sha256": hashlib.sha256(
                        json.dumps(result[3], sort_keys=True).encode()
                    ).hexdigest(),
                }
            )
        if name == "create_intermediate_bins":
            checksum_start = time.monotonic()
            total = 0
            for key in result:
                total = (
                    total + int.from_bytes(hashlib.sha256(key).digest(), "big")
                ) % (1 << 256)
            emit(
                {
                    "event": "candidates",
                    "count": len(result),
                    "sha256_sum": f"{total:064x}",
                    "fingerprint_seconds": time.monotonic() - checksum_start,
                }
            )
        return result

    setattr(module, name, measured)


def main():
    emit(
        {
            "event": "environment",
            "python": sys.version,
            "source": os.environ["BINETTE_PROFILE_SOURCE"],
            "hashseed": os.environ.get("PYTHONHASHSEED"),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "checkm2",
                    "numpy",
                    "networkx",
                    "pyfastx",
                    "pyrodigal",
                    "joblib",
                )
            },
        }
    )
    for module, names in (
        (cli, ("parse_input_files", "manage_protein_alignement")),
        (cds, ("get_contig_cds_metadata",)),
        (bin_quality, ("add_bin_metrics", "add_bin_size_and_N50")),
        (bin_manager, ("create_intermediate_bins", "select_best_bins")),
    ):
        for name in names:
            instrument(module, name)
    start = time.monotonic()
    try:
        cli.main()
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    emit({"event": "complete", "wall_seconds": time.monotonic() - start})


if __name__ == "__main__":
    main()
