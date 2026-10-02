#!/usr/bin/env bash
# Source-bound policy gate and fresh full-depth memory-cap validation.
set -euo pipefail
bundle="$1"
variant="$2"
export TMPDIR="/lustre/scratch03/abaud/ohickl/nf-imp3_tests/binette_policy/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
db_root="${IMP3_DB_ROOT:-/no_backup/abaud/data/secondary/databases/nf-imp3}"
args=(--bundle "$bundle" --database "$db_root/checkm2/v1.1.0-zenodo14897628/CheckM2_database/uniref100.KO.1.dmnd" --variant "$variant")
if [[ "$variant" == gate ]]; then
  args+=(--root "$bundle/gate" --fixture /no_backup/abaud/data/secondary/rat_mag_catalog/analysis/nf-imp3_tests/cami3_canonical_microfixture/cami3_sample_0_0p1pct_seed20260802 --micro-assembly /no_backup/abaud/data/secondary/rat_mag_catalog/analysis/nf-imp3_tests/cami3_binning_microvalidation/micro_27422307/results/cami3_sample_0/assembly/canonical/cami3_sample_0.assembly.fasta.gz)
elif [[ "$variant" == modified ]]; then
  if [[ "$SLURM_CPUS_PER_TASK" != 20 || "$SLURM_MEM_PER_NODE" != 36864 ]]; then
    echo "Policy validation requires 20 CPUs and --mem=36G" >&2
    exit 2
  fi
  args+=(--root "$bundle/trial/full_modified" --threads 20 --no-scoring-telemetry)
else
  echo "Expected gate or modified" >&2
  exit 2
fi
image=oras://ghcr.io/ohickl/nf-imp3-wave:binette-1.2.1_pyrodigal-3.7_zstd_checkm2-1.1.0_zstandard--402d8f6f71dabc27
runtime=(apptainer exec --no-home --pid --bind /no_backup --bind /users --bind /lustre --env PATH=/opt/wave/.pixi/envs/default/bin:/usr/local/bin:/usr/bin:/bin --env SSL_CERT_FILE=/etc/ssl/certs/ca-bundle.crt --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/ssl/certs/ca-bundle.crt "$image")
"${runtime[@]}" python "$bundle/scripts/run_crg_profile.py" "${args[@]}"
if [[ "$variant" == modified ]]; then
  "${runtime[@]}" python "$bundle/scripts/compare_crg_profiles.py" --root "$bundle/trial" --baseline-root /no_backup/abaud/data/secondary/rat_mag_catalog/analysis/nf-imp3_tests/binette_optimization/profile_20261001_86e515d72d97
fi
