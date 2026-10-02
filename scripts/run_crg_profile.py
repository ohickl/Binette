"""Prepare and supervise an isolated Binette experiment inside a Slurm worker.

The fixed transfer manifest supplies immutable production inputs. The technical
gate uses the canonical paired CAMI0 fixture and its recorded micro-assembly;
small deterministic partitions of that assembly are adapter controls, not bins
from a biological benchmark. No production outputs or checkpoints are modified.
"""

import argparse
import csv
import gzip
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_full(bundle, root):
    manifest = json.loads((bundle / "transfer.json").read_text())
    assembly = None
    tables = []
    artifacts = []
    for entry in manifest["artifacts"]:
        role = entry["role"]
        if entry["sample"] != "cami3_sample_13" or not (
            role == "assembly"
            or role.startswith("raw_")
            and not role.endswith("_provenance")
        ):
            continue
        source = Path(entry["remote"])
        if sha256(source) != entry["local_sha256"]:
            raise ValueError(f"Input hash mismatch: {source}")
        artifacts.append(entry)
        if role == "assembly":
            assembly = root / "assembly.fa"
            with gzip.open(source, "rb") as reader, assembly.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
        else:
            target = root / f"binette.{role}.tsv"
            with source.open() as reader, target.open("w") as writer:
                for row in csv.DictReader(reader, delimiter="\t"):
                    writer.write(f"{row['contig_id']}\t{row['bin_id']}\n")
            tables.append(target)
    if assembly is None or len(tables) != 10:
        raise ValueError(
            "Full raw-only panel requires one assembly and ten input tables"
        )
    write_json(
        root / "inputs.json",
        {
            "schema": "binette-full-profile-inputs-v1",
            "artifacts": artifacts,
            "adapted_tables": {p.name: sha256(p) for p in tables},
            "assembly_sha256": sha256(assembly),
        },
    )
    return assembly, sorted(tables)


def prepare_gate(root, fixture, micro_assembly):
    import pyfastx

    provenance = json.loads((fixture / "preparation.json").read_text())
    if (provenance["sample_id"], provenance["fraction"], provenance["seed"]) != (
        "cami3_sample_0",
        0.001,
        20260802,
    ):
        raise ValueError("Unexpected canonical fixture identity")
    reads = []
    for mate in ("R1", "R2"):
        path = fixture / "reads" / f"clean_{mate}.fastq.gz"
        if sha256(path) != provenance["reads"][mate]["sha256"]:
            raise ValueError("Canonical read hash mismatch")
        names = [
            name.removesuffix("/1").removesuffix("/2")
            for name, _, _ in pyfastx.Fastx(str(path))
        ]
        reads.append(names)
    if reads[0] != reads[1] or len(reads[0]) != 16427:
        raise ValueError("Canonical pair identity/count mismatch")
    assembly = root / "assembly.fa"
    with gzip.open(micro_assembly, "rb") as reader, assembly.open("wb") as writer:
        shutil.copyfileobj(reader, writer)
    names = [name for name, _ in pyfastx.Fastx(str(assembly))]
    if not names:
        raise ValueError("Micro assembly is empty")
    tables = [root / "binette.micro_a.tsv", root / "binette.micro_b.tsv"]
    for offset, target in enumerate(tables):
        with target.open("w") as handle:
            for index, name in enumerate(names):
                handle.write(f"{name}\tadapter_{(index + offset) % 3}\n")
    write_json(
        root / "inputs.json",
        {
            "schema": "binette-micro-adapter-v1",
            "fixture": provenance,
            "pairs_verified": len(reads[0]),
            "assembly_source": str(micro_assembly),
            "assembly_sha256": sha256(assembly),
            "contigs": len(names),
            "tables": {p.name: sha256(p) for p in tables},
            "scope": "Canonical paired-read-derived assembly; deterministic adapter partitions, not biological bins",
        },
    )
    return assembly, tables


def process_tree_rss(pid):
    pending = [pid]
    seen = set()
    anonymous = 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text()
            for line in status.splitlines():
                if line.startswith("RssAnon:"):
                    anonymous += int(line.split()[1]) * 1024
            pending.extend(
                int(child)
                for child in Path(f"/proc/{current}/task/{current}/children")
                .read_text()
                .split()
            )
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return anonymous, len(seen)


def run(
    bundle, root, variant, assembly, tables, database, threads,
    quality_workers=None, scoring_telemetry=True,
):
    target = root / variant
    target.mkdir(exist_ok=True)
    env = dict(
        os.environ,
        BINETTE_PROFILE_SOURCE=str(bundle / variant),
        BINETTE_PROFILE_RECEIPT=str(target / "phases.jsonl"),
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    command = [
        sys.executable,
        str(bundle / "scripts/profile_cli.py"),
        "--contig2bin_tables",
        *map(str, tables),
        "--contigs",
        str(assembly),
        "--checkm2_db",
        str(database),
        "--threads",
        str(threads),
        "--outdir",
        str(target / "output"),
        "--prefix",
        "sample13_profile",
        "--min_completeness",
        "70",
        "--max_contamination",
        "10",
        "--no-progress",
    ]
    if variant == "modified":
        if quality_workers is not None:
            command.extend(["--quality-workers", str(quality_workers)])
        command.extend(["--score-cache", str(target / "score_shards")])
        if scoring_telemetry:
            env["BINETTE_SCORING_TELEMETRY"] = str(target / "scoring.jsonl")
        else:
            env.pop("BINETTE_SCORING_TELEMETRY", None)
    contract = {
        "argv": command,
        "source": env["BINETTE_PROFILE_SOURCE"],
        "threads": threads,
        "hashseed": 0,
        "thread_limits": 1,
        "scoring_telemetry": scoring_telemetry,
        "sources_sha256": sha256(bundle / "sources.json"),
        "inputs_sha256": sha256(root / "inputs.json"),
    }
    command_path = target / "command.json"
    if command_path.exists() and json.loads(command_path.read_text()) != contract:
        raise ValueError("Refusing recovery across source/input/parameter namespaces")
    write_json(command_path, contract)
    start = time.monotonic()
    peak = 0
    with (
        (target / "cli.out").open("w") as stdout,
        (target / "cli.err").open("w") as stderr,
        (target / "memory.jsonl").open("w") as samples,
    ):
        child = subprocess.Popen(
            command, env=env, stdout=stdout, stderr=stderr, cwd=target
        )
        while child.poll() is None:
            rss, processes = process_tree_rss(child.pid)
            peak = max(peak, rss)
            samples.write(
                json.dumps(
                    {
                        "elapsed_seconds": time.monotonic() - start,
                        "summed_process_tree_rssanon_bytes": rss,
                        "processes": processes,
                    }
                )
                + "\n"
            )
            samples.flush()
            time.sleep(1)
    receipt = {
        "schema": "binette-cli-profile-v1",
        "variant": variant,
        "exit_code": child.returncode,
        "wall_seconds": time.monotonic() - start,
        "summed_process_tree_rssanon_peak_bytes": peak,
        "memory_limitations": "Summed process RssAnon may count shared anonymous pages repeatedly; one-second samples can miss short peaks",
    }
    write_json(target / "result.json", receipt)
    if child.returncode:
        raise RuntimeError(
            f"{variant} failed: {child.returncode}; inspect {target / 'cli.err'}"
        )
    return receipt


def validate_graphs(bundle):
    import networkx as nx

    spec = importlib.util.spec_from_file_location(
        "optimized_manager", bundle / "modified/binette/bin_manager.py"
    )
    manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(manager)
    for seed in range(12):
        graph = nx.gnp_random_graph(10, 0.5, seed=seed)
        expected = {
            tuple(sorted(subset))
            for clique in nx.find_cliques(graph)
            for size in range(2, len(clique) + 1)
            for subset in itertools.combinations(clique, size)
        }
        observed = list(manager.iter_unique_bin_combinations(graph))
        if len(observed) != len(set(observed)) or set(observed) != expected:
            raise ValueError("Unique subset enumeration differs from exhaustive oracle")


def validate_numeric(bundle, root):
    """Nonempty scoring control supplements the tiny canonical assembly path."""
    from collections import Counter
    from unittest.mock import patch

    from pyroaring import BitMap

    from binette import bin_manager, bin_quality
    from binette.features import ContigEvidence, feature_plan

    spec = importlib.util.spec_from_file_location(
        "baseline_quality", bundle / "baseline/binette/bin_quality.py"
    )
    upstream = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(upstream)
    kos = feature_plan().kos
    kegg = {i: Counter({kos[i]: i + 1, kos[i + 10]: 2}) for i in range(4)}
    counts = {i: 20 + i for i in range(4)}
    amino = {
        i: Counter({aa: 10 + i for aa in "ACDEFGHIKLMNPQRSTVWY"}) for i in range(4)
    }
    lengths = {i: sum(amino[i].values()) for i in range(4)}
    memberships = ([0, 1], [1, 2], [2, 3], [3], [0, 2], [0, 1, 3], [99], [0, 2, 99])
    expected = [bin_manager.Bin(BitMap(item)) for item in memberships]
    upstream.assess_bins_quality(
        expected, kegg, counts, amino, lengths, 2, 8, threads=1
    )
    info = {"_evidence": ContigEvidence.from_dicts(kegg, counts, amino, lengths)}

    def score():
        actual = [bin_manager.Bin(BitMap(item)) for item in memberships]
        bin_quality.add_bin_metrics(
            actual,
            info,
            2,
            threads=2,
            quality_workers=2,
            checkm2_batch_size=2,
            score_cache=root / "numeric_scores",
            disable_progress_bar=True,
        )
        def fields(bins):
            return [
                (b.contigs_key, b.completeness, b.contamination, b.score, b.checkm2_model)
                for b in bins
            ]
        if fields(actual) != fields(expected):
            raise ValueError("Frozen-runtime numeric scores differ from upstream")

    score()
    with patch(
        "binette.scoring.make_processors",
        side_effect=AssertionError("Unexpected restore inference"),
    ):
        score()
    # Model outputs survive a missing shard while sealed neighbors are retained.
    namespace = next((root / "numeric_scores").iterdir())
    for suffix in ("npz", "json"):
        (namespace / f"0.{suffix}").rename(root / f"interrupted_score_0.{suffix}")
    score()
    return {
        "bins": len(memberships),
        "workers": 2,
        "batch_size": 2,
        "exact_scores_and_models": True,
        "complete_and_partial_restore": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("gate", "baseline", "modified"), required=True
    )
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--micro-assembly", type=Path)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--quality-workers", type=int)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--no-scoring-telemetry", action="store_true")
    args = parser.parse_args()
    if args.threads <= 0 or (
        args.quality_workers is not None
        and not 1 <= args.quality_workers <= args.threads
    ):
        parser.error("Workers must be positive and fit within --threads")
    args.root.mkdir(parents=True, exist_ok=args.recover)
    sys.path.insert(0, str(args.bundle / "modified"))
    if args.variant == "gate":
        assembly, tables = prepare_gate(args.root, args.fixture, args.micro_assembly)
        validate_graphs(args.bundle)
        receipts = [
            run(args.bundle, args.root, variant, assembly, tables, args.database, 2)
            for variant in ("baseline", "modified")
        ]
        # Fresh processes reconstruct outputs from durable stage objects,
        # including coding lengths; Binette's protein-only resume is not used.
        for variant in ("baseline", "modified"):
            output = args.root / variant / "output"
            shutil.move(output, args.root / variant / "cold_output")
            run(args.bundle, args.root, variant, assembly, tables, args.database, 2)
            for file in ("final_contig_to_bin.tsv", "final_bins_quality_reports.tsv"):
                if (output / file).read_bytes() != (
                    output.parent / "cold_output" / file
                ).read_bytes():
                    raise ValueError("Cold stage recovery changed output")
        for file in ("final_contig_to_bin.tsv", "final_bins_quality_reports.tsv"):
            first = args.root / "baseline/output" / file
            second = args.root / "modified/output" / file
            if first.read_bytes() != second.read_bytes():
                raise ValueError(f"Micro output differs: {file}")
        numeric = validate_numeric(args.bundle, args.root)
        write_json(
            args.root / "gate.json",
            {
                "verdict": "PASS",
                "pairs": 16427,
                "graph_oracles": 12,
                "runs": receipts,
                "numeric_control": numeric,
                "bundle_manifest_sha256": sha256(args.bundle / "sources.json"),
            },
        )
        print(
            "binette-isolated-microgate-v2 pairs=16427 variants=2 graph_oracles=12 numeric_bins=8 workers=2 complete_restore=PASS partial_restore=PASS failed=0",
            flush=True,
        )
    else:
        gate = json.loads((args.bundle / "gate/gate.json").read_text())
        if gate["verdict"] != "PASS" or gate["bundle_manifest_sha256"] != sha256(
            args.bundle / "sources.json"
        ):
            raise ValueError("A passing gate bound to this source snapshot is required")
        assembly, tables = prepare_full(args.bundle, args.root)
        print(
            json.dumps(
                run(
                    args.bundle,
                    args.root,
                    args.variant,
                    assembly,
                    tables,
                    args.database,
                    args.threads,
                    args.quality_workers,
                    not args.no_scoring_telemetry,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
