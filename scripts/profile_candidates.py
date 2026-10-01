"""Profile exact candidate generation on a bounded, declared CAMI subpanel.

Original quality estimates are imported from the recorded successful run. This
does not benchmark gene calling, DIAMOND, quality inference or a full panel.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import resource
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import networkx as nx
import pyfastx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from binette import bin_manager, bin_quality  # noqa: E402

BASELINE = "4f8791e9b411708b0717dbd8e1296a68ac72ae24"


def anonymous_rss() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Linux RssAnon is unavailable")


@contextmanager
def phase(records: list[dict], name: str):
    start = time.perf_counter()
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    samples = [anonymous_rss()]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.01):
            samples.append(anonymous_rss())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        yield
    finally:
        stop.set()
        sampler.join()
        end_cpu = resource.getrusage(resource.RUSAGE_SELF)
        samples.append(anonymous_rss())
        records.append(
            {
                "phase": name,
                "wall_seconds": time.perf_counter() - start,
                "cpu_seconds": end_cpu.ru_utime
                + end_cpu.ru_stime
                - cpu.ru_utime
                - cpu.ru_stime,
                "anonymous_rss_start_bytes": samples[0],
                "anonymous_rss_peak_bytes": max(samples),
                "anonymous_rss_end_bytes": samples[-1],
                "sampling_seconds": 0.01,
            }
        )


def load_manager(variant: str):
    if variant == "modified":
        return bin_manager
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:binette/bin_manager.py"], cwd=ROOT, text=True
    )
    module = ModuleType("baseline_bin_manager")
    exec(compile(source, f"{BASELINE}/bin_manager.py", "exec"), module.__dict__)
    return module


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def profile(args) -> dict:
    if not 2 <= args.max_bins <= 10:
        raise ValueError("Laptop subpanel bound is 2–10 original bins")
    transfer_path = args.inputs / "transfer.json"
    transfer = json.loads(transfer_path.read_text())
    artifacts = [
        item for item in transfer["artifacts"] if item["sample"] == args.sample
    ]
    manager = load_manager(args.variant)
    records = []
    with phase(records, "load_tables_and_recorded_quality"):
        sources = {}
        quality = {}
        assembly = None
        for item in artifacts:
            path = args.inputs / item["local"]
            role = item["role"]
            if role == "assembly":
                assembly = path
            elif role.startswith("quality_"):
                name = f"binette.{role.removeprefix('quality_')}.tsv"
                text = subprocess.check_output(["zstd", "-dc", str(path)], text=True)
                for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
                    quality[name, row["original_name"]] = (
                        float(row["completeness"]),
                        float(row["contamination"]),
                    )
            elif (
                role.startswith("raw_")
                and not role.endswith("_provenance")
                or role == "magscot_joint_single_multi"
            ):
                name = f"binette.{role}.tsv"
                bins = {}
                with path.open() as handle:
                    reader = csv.DictReader(handle, delimiter="\t")
                    for row in reader:
                        bins.setdefault(row["bin_id"], set()).add(row["contig_id"])
                sources[name] = [
                    {"contigs": members, "bin_name": name}
                    for name, members in bins.items()
                ]
        if assembly is None:
            raise ValueError("Missing assembly")
        if args.sample == "cami3_sample_13":
            # Only this panel has a successful recorded quality oracle.
            sources.pop("binette.magscot_joint_single_multi.tsv")
        sources = dict(sorted(sources.items()))
        names = sorted(
            {name for bins in sources.values() for b in bins for name in b["contigs"]}
        )
        index = dict(zip(names, range(len(names)), strict=True))
        original = manager.make_bins_from_bins_info(sources, index, True)
        for b in original.values():
            first_source = min(b.origin)
            completeness, contamination = quality[first_source, b.name]
            b.add_quality(completeness, contamination, 2)
    with phase(records, "full_overlap_graph_only"):
        graph = manager.from_bins_to_bin_graph(original.values())
    clique_stats = None
    if args.clique_stats:
        with phase(records, "stream_full_clique_statistics"):
            histogram = {}
            for clique in nx.find_cliques(graph):
                size = len(clique)
                histogram[size] = histogram.get(size, 0) + 1
        pair_visits = sum(
            count * size * (size - 1) // 2 for size, count in histogram.items()
        )
        clique_stats = {
            "maximal_cliques": sum(histogram.values()),
            "size_histogram": histogram,
            "subset_visits_before_parent_gates": sum(
                count * ((1 << size) - size - 1) for size, count in histogram.items()
            ),
            "pair_subset_visits": pair_visits,
            "distinct_pair_subsets": graph.number_of_edges(),
            "repeated_pair_subset_visits": pair_visits - graph.number_of_edges(),
        }
    # Deterministic high-degree seed and neighbors: an induced subpanel, not a
    # complete connected component or a representative biological benchmark.
    seed = min(graph, key=lambda key: (-graph.degree[key], key))
    selected_keys = [seed] + sorted(
        graph.neighbors(seed), key=lambda key: (-graph.degree[key], key)
    )[: args.max_bins - 1]
    selected = {key: original[key] for key in selected_keys}
    selected_ids = {contig for b in selected.values() for contig in b.contigs}
    with phase(records, "read_selected_contig_lengths"):
        lengths = {
            index[name]: len(sequence)
            for name, sequence in pyfastx.Fastx(str(assembly))
            if name in index and index[name] in selected_ids
        }
        if set(lengths) != selected_ids:
            raise ValueError("Selected contigs are absent from assembly")
        bin_quality.add_bin_size_and_N50(selected.values(), lengths)
        length_array = bin_quality.prepare_contig_sizes(lengths)
    with phase(records, "bounded_candidate_generation"):
        candidates = manager.create_intermediate_bins(
            selected, length_array, 70, 10, 200_000, 10_000_000, True
        )
    memberships = sorted(
        [names[contig] for contig in b.contigs] for b in candidates.values()
    )
    fixture = [
        {
            "source": min(b.origin),
            "name": b.name,
            "contigs": [names[c] for c in b.contigs],
            "completeness": b.completeness,
            "contamination": b.contamination,
        }
        for b in selected.values()
    ]
    return {
        "schema": "binette-candidate-profile-v1",
        "variant": args.variant,
        "sample": args.sample,
        "panel": "all_raw_plus_magscot"
        if args.sample == "cami3_sample_1"
        else "raw_only",
        "scope": "full input overlap graph plus bounded induced candidate subpanel; imported original scores",
        "baseline": BASELINE,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "transfer_manifest_sha256": digest(transfer_path.read_bytes()),
        "bin_manager_source_sha256": digest(
            (ROOT / "binette/bin_manager.py").read_bytes()
        )
        if args.variant == "modified"
        else digest(
            subprocess.check_output(
                ["git", "show", f"{BASELINE}:binette/bin_manager.py"], cwd=ROOT
            )
        ),
        "original_bins": len(original),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "selected_original_bins": len(selected),
        "selected_contigs": len(selected_ids),
        "candidate_count": len(candidates),
        "fixture_sha256": digest(json.dumps(fixture, sort_keys=True).encode()),
        "candidate_memberships_sha256": digest(json.dumps(memberships).encode()),
        "phases": records,
        "clique_statistics": clique_stats,
        "limitations": [
            "No new CheckM2 inference",
            "No full candidate enumeration",
            "Sampling may miss brief allocation peaks",
            "Single run is not a speedup estimate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("test_data/cami3_profile"))
    parser.add_argument(
        "--sample", choices=["cami3_sample_1", "cami3_sample_13"], required=True
    )
    parser.add_argument("--variant", choices=["baseline", "modified"], required=True)
    parser.add_argument("--max-bins", type=int, default=8)
    parser.add_argument("--clique-stats", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; retain independent receipts")
    result = profile(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"{args.sample} {args.variant}: {result['candidate_count']} candidates; receipt {args.output}"
    )


if __name__ == "__main__":
    main()
