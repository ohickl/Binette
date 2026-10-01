"""Compare completed isolated full-depth runs; preserve explicit failure evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def memberships(path):
    bins = {}
    with path.open() as handle:
        for line in handle:
            contig, name = line.rstrip("\n").split("\t")
            bins.setdefault(name, set()).add(contig)
    return {name: frozenset(contigs) for name, contigs in bins.items()}


def load(root, variant):
    path = root / f"full_{variant}" / variant
    result = json.loads((path / "result.json").read_text())
    if result["exit_code"] != 0:
        raise ValueError(f"{variant} exit code {result['exit_code']}")
    phases = [
        json.loads(line) for line in (path / "phases.jsonl").read_text().splitlines()
    ]
    if not any(row["event"] == "complete" for row in phases):
        raise ValueError(f"{variant} has no CLI completion receipt")
    identity = next(row for row in phases if row["event"] == "identity")
    candidates = next(row for row in phases if row["event"] == "candidates")
    clusters = memberships(path / "output/final_contig_to_bin.tsv")
    with (path / "output/final_bins_quality_reports.tsv").open() as handle:
        metrics = {row["name"]: row for row in csv.DictReader(handle, delimiter="\t")}
    if set(metrics) != set(clusters):
        raise ValueError("Final report and assignments disagree")
    return result, phases, identity, candidates, clusters, metrics


def compare(root, baseline_root=None):
    baseline = load(baseline_root or root, "baseline")
    modified = load(root, "modified")
    if baseline[2] != modified[2]:
        raise ValueError("Original-bin count or contig integer namespace differs")
    candidate_equal = all(
        baseline[3][key] == modified[3][key] for key in ("count", "sha256_sum")
    )
    cluster_equal = set(baseline[4].values()) == set(modified[4].values())
    fields = (
        "completeness",
        "contamination",
        "score",
        "checkm2_model",
        "size",
        "N50",
        "coding_density",
        "contig_count",
    )
    quality_by_membership = []
    for data in (baseline, modified):
        quality_by_membership.append(
            {
                data[4][name]: tuple(row[key] for key in fields)
                for name, row in data[5].items()
            }
        )
    scientific_equal = quality_by_membership[0] == quality_by_membership[1]
    measurements = {}
    for variant, data in (("baseline", baseline), ("modified", modified)):
        checkpoints = sum(
            row["wall_seconds"] for row in data[1] if row["event"] == "checkpoint"
        )
        fingerprints = sum(row.get("fingerprint_seconds", 0) for row in data[1])
        measurements[variant] = dict(
            data[0],
            checkpoint_seconds=checkpoints,
            candidate_fingerprint_seconds=fingerprints,
            instrumented_wall_minus_checkpoint_and_fingerprint_seconds=data[0][
                "wall_seconds"
            ]
            - checkpoints
            - fingerprints,
            phases=[row for row in data[1] if row["event"] == "end"],
        )
    canonical = sorted(sorted(cluster) for cluster in baseline[4].values())
    return {
        "schema": "binette-full-depth-comparison-v1",
        "baseline_root": str(baseline_root or root),
        "modified_root": str(root),
        "verdict": "PASS"
        if candidate_equal and cluster_equal and scientific_equal
        else "FAIL",
        "sample": "cami3_sample_13",
        "panel": "raw_only",
        "original_bins": baseline[2]["original_bins"],
        "candidates": baseline[3]["count"],
        "candidate_fingerprint_equal": candidate_equal,
        "candidate_fingerprint_method": "count plus order-independent modular sum of SHA-256 digests of serialized memberships; probabilistic integrity comparison",
        "final_memberships_exactly_equal": cluster_equal,
        "final_scientific_metrics_exactly_equal": scientific_equal,
        "final_bins": len(canonical),
        "final_contigs": sum(map(len, canonical)),
        "final_memberships_sha256": hashlib.sha256(
            json.dumps(canonical).encode()
        ).hexdigest(),
        "measurements": measurements,
        "limitations": [
            "One run per version; no replicated speedup or scaling estimate",
            "Shared node and storage contention may affect timing",
            "Summed anonymous RSS may double-count shared anonymous pages",
            "Historical production output can differ through hash-seed-dependent tie ordering",
            "Checkpoint and fingerprint instrumentation overhead is measured separately",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    args = parser.parse_args()
    try:
        result = compare(args.root, args.baseline_root)
    except Exception as error:
        result = {
            "schema": "binette-full-depth-comparison-v1",
            "verdict": "INCOMPLETE",
            "error": str(error),
        }
    (args.root / "comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
