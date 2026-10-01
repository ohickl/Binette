"""Fetch an explicit CAMI profiling input list; never traverse a cluster tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

RUN = "/no_backup/abaud/data/secondary/rat_mag_catalog/analysis/nf-imp3_tests/cami3_binning_comparison_source/full20_e8eb416_20260930_nfs"
ASSEMBLIES = "/no_backup/abaud/data/secondary/rat_mag_catalog/benchmarking/cami/nf-imp3/runs/cami3_full20_megahit_meta_sensitive_metacarvel_20260811_f4064ad/arms/megahit_meta_sensitive_r1_control/results"
BINNERS = ("metabat2", "metacat", "metadecoder", "quickbin", "semibin2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="crg_hpc")
    parser.add_argument("--output", type=Path, default=Path("test_data/cami3_profile"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    for number in (1, 13):
        sample = f"cami3_sample_{number}"
        target = f"{sample}__megahit_meta_sensitive_r1_control"
        root = f"{RUN}/results/{target}/binning"
        groups = {"inputs": [], "reports": []}
        groups["inputs"].append(
            (
                f"{ASSEMBLIES}/{sample}/assembly/canonical/{sample}.assembly.fasta.gz",
                "assembly",
            )
        )
        for mode in ("single", "multi"):
            for binner in BINNERS:
                source = f"raw_{binner}_{mode}"
                stem = f"{target}.{mode}.{source}"
                groups["inputs"].extend(
                    [
                        (f"{root}/{mode}/{source}/{stem}.contig2bin.tsv", source),
                        (
                            f"{root}/{mode}/{source}/{stem}.binning.provenance.tsv",
                            f"{source}_provenance",
                        ),
                    ]
                )
        groups["inputs"].extend(
            [
                (
                    f"{root}/refined/magscot/magscot.contig2bin.tsv",
                    "magscot_joint_single_multi",
                ),
                (
                    f"{root}/refined/magscot/magscot.provenance.tsv",
                    "magscot_provenance",
                ),
            ]
        )
        panel = "binette" if number == 1 else "binette_raw_only"
        report_root = (
            f"{root}/refined/{panel}/binette_quality_reports/input_bins_quality_reports"
        )
        sources = (["magscot_joint_single_multi"] if number == 1 else []) + [
            f"raw_{binner}_{mode}" for binner in BINNERS for mode in ("multi", "single")
        ]
        for index, source in enumerate(sources, 1):
            groups["reports"].append(
                (
                    f"{report_root}/input_bins_{index}.binette.{source}.tsv.tsv.zst",
                    f"quality_{source}",
                )
            )
        groups["reports"].extend(
            [
                (f"{root}/refined/{panel}/binette.log", "execution_log"),
                (f"{root}/refined/{panel}/binette.provenance.tsv", "panel_provenance"),
            ]
        )
        for group, paths in groups.items():
            destination = args.output / sample / group
            destination.mkdir(parents=True)
            subprocess.run(
                [
                    "scp",
                    "-q",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=12",
                    *[f"{args.host}:{path}" for path, _ in paths],
                    str(destination),
                ],
                check=True,
            )
            for source, role in paths:
                local = destination / Path(source).name
                artifacts.append(
                    {
                        "sample": sample,
                        "role": role,
                        "remote": source,
                        "local": str(local.relative_to(args.output)),
                        "size_bytes": local.stat().st_size,
                        "local_sha256": sha256(local),
                    }
                )
        print(f"Transferred {sample}", flush=True)
    receipt = {
        "schema": "binette-cami-transfer-v1",
        "source_run": RUN,
        "verification": "SSH transfer and local SHA-256; independent remote digest verification pending",
        "artifacts": artifacts,
    }
    (args.output / "transfer.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Receipt: {args.output / 'transfer.json'}")


if __name__ == "__main__":
    main()
