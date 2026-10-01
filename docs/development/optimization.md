# Optimization development

The initial fork revision is `989088407625ea3dd71036a44cca60929d3de9fc`.
Differential tests load the audited upstream implementation directly from Git
revision `4f8791e9b411708b0717dbd8e1296a68ac72ae24`.

## Reproduce bounded validation

```bash
pixi install --locked
PYTHONHASHSEED=0 pixi run --locked unit
PYTHONHASHSEED=0 OMP_NUM_THREADS=1 pixi run --locked equivalence
```

`pixi.lock` records the Linux development environment, including CheckM2
1.1.0 and Pyrodigal below 3.7. The optional large external Binette test dataset
is not required by these two tasks. Its absence produces an upstream warning;
these tasks run unit controls and explicit synthetic differential fixtures.
The equivalence task uses real packaged models/reference data, without running
DIAMOND or downloading its database. Git history must include the audited
revision. Machine-local Pixi environment/cache settings belong in ignored
`.pixi/config.toml`, not the portable manifest.

## First implementation increment — 2026-10-01

- Correct retained duplicate-bin source provenance; keep the first name.
- Remove the unread KO contig-to-bin index and candidate-generation length
  cache, whose entries are never revisited after acceptance/rejection.
- Load prediction models once per `assess_bins_quality` call and pass the
  processor between batches. In parallel mode this is once per chunk, not yet
  once per persistent worker lifetime. The fallback batch helper still works
  for independent callers.
- Iterate batches without materializing another list of batch tuples.
- Bound amino-acid metadata futures to twice the worker count; preserve input
  order and propagate worker failures.

The baseline had 107 passing unit tests. New provenance/model-reuse controls
failed before changes; the bounded-future control likewise failed before its
implementation. The modified suite has 115 passes and one strict expected
failure recording the exact-half N50 defect. Four real differential controls
passed, including exact raw predictions and scaled features at batch sizes
1/2/4, identical rounded scores/model choices, and identical candidate
memberships/greedy selection on the declared small fixture. Lint and whitespace
checks passed. Version, code, fixture-definition and model-resource hashes are
recorded in `optimization_validation_2026-10-01.json`.

These are bounded correctness results. They establish neither full-depth speed
or memory gains nor multiprocess model equivalence. Curated CAMI input transfer,
phase profiling, indexed overlap construction, bounded reference similarity,
shared feature ownership, persistent worker reuse and durable partial-progress
checkpoints remain open. N50 correction and canonical contig-ID mapping require
separate behavioral qualification. Production integration requires the nf-imp3
CAMI3 microgate followed by eligible full-depth validation.

## CAMI profiling and unique subsets — 2026-10-01

`scripts/fetch_cami_profile_inputs.py` transferred only the two control
assemblies, ten raw assignment/provenance sets per sample, MAGScoT assignments,
and recorded quality reports/logs from the full20 source. The ignored
`test_data/cami3_profile/transfer.json` records 71 file identities. Compute-tier
integrity job `29167139` completed `0:0` and independently confirmed:

```text
binette-input-integrity-v1 artifacts=71 matched=71 failed=0
```

The successful sample-13 raw-only log records 1,103 original bins, 567,689
maximal cliques and 6,945,894 intermediate candidates, of which 6,686,824 were
differences. Candidate generation ran between 17:11:28 and 18:10:39. The
interval from intermediate scoring start to selection start was 18:10:40 to
20:02:27; this includes intervening result assembly and size/N50 work and is
not isolated model-inference time. Sample-1's newer main panel differs from the
earlier audit case: 988 originals, 45,297 cliques and 478,118 candidates.

Streaming full-graph statistics reproduced both recorded clique counts. For
sample 13, the old maximal-clique loop visits 276,298,222 subsets before parent
gates, including 20,134,908 pair visits for 12,138 distinct graph edges.
`iter_unique_bin_combinations` now enumerates every complete subset once via
an increasing-order depth-first traversal. It retains only its active stack
and eliminates maximal-clique materialization and repeated subset visits.
Exact enumeration can still be exponential. No thresholds or search-space
truncation were introduced. Operation-attribution counts can change with order;
candidate memberships and selected outputs are the equivalence contract.

Bounded profiling used eight-parent induced subpanels with imported quality
estimates, rather than full candidate enumeration or new CheckM2 scoring.
Three repetitions for upstream, allocation changes, and unique subsets
produced identical fixture/candidate digests: 35 candidates for sample 1 and
68 for sample 13. Receipts and timing ranges are summarized in
`cami_profile_summary_2026-10-01.json`. These non-interleaved, small runs under
variable laptop load do not establish a full-depth speedup or memory reduction.

```bash
PYTHONHASHSEED=0 pixi run --locked python scripts/profile_candidates.py \
  --sample cami3_sample_13 --variant modified \
  --output test_data/cami3_profile/profiles/new_receipt.json
```

The standalone unit task passed 117 controls with one strict N50 expected
failure. The equivalence task passed 16 controls, including 12 randomized
parent-gate cases and actual CheckM2 inference. The combined-process suite
aborted in the existing Pyfastx indexed-FASTA helper. A standalone Pyfastx
reproducer and debugger localized the native abort to index deallocation;
the pinned 2.3.1 source allocates `index_len` bytes for an explicit index path
and then writes its terminator at `index_len`. See
[upstream source](https://github.com/lmdu/pyfastx/blob/7d82c66b1f5d079d6b7d46f878d156cbaa6c01aa/src/index.c).
This dependency defect remains unresolved and is captured by the strict
expected-failure subprocess control in `pyfastx_index_safety_test.py`. Ordinary
CLI sequence streaming uses `Fastx`; the custom-index helper is not called by
the inspected CLI. Do not report the combined suite as passing.

### Isolated full-depth sample-13 experiment, 2026-10-01

User-authorized full raw-only CLI experiments use the same frozen production
image and resource allocation (16 CPUs, 96G), with a longer 12h walltime.
Baseline code is pinned to `4f8791e9b411708b0717dbd8e1296a68ac72ae24`;
modified code is a hashed snapshot of the local fork. Source snapshot:
`86e515d72d97eea4aa6f6cfd93379a987a182d856484d982dbb348599d5c16ec`.

Durable root:
`/no_backup/abaud/data/secondary/rat_mag_catalog/analysis/nf-imp3_tests/binette_optimization/profile_20261001_86e515d72d97`.

- Technical gate `29167678`: COMPLETED 0:0 in 1m57s;
  `binette-isolated-microgate-v1 pairs=16427 variants=2 graph_oracles=12 failed=0`.
- Full modified `29167714` and baseline `29167715`: submitted, results pending.
- Comparison `29167739`: compute-tier afterany dependency on both full jobs.
- Runtime Apptainer 1.5.0-1.el9; cached frozen image digest:
  `sha256:8a8779c64464320824d807c2f474c075ab054d33a5d2ee2916c272bb414e4ce6`.

The gate verifies both canonical read hashes, all 16,427 ordered mate names,
recorded fraction/seed/sample identity, and uses the previously produced canonical
micro-assembly. Its deterministic contig partitions exercise the table adapter,
ordinary CLI, actual models, output handling, and cold stage-cache recovery;
they are not biological binning outputs. Twelve graph controls independently
compare unique enumeration with the exhaustive maximal-clique subset oracle.
This gate does not establish full-depth scientific equivalence or performance.

`profile_cli.py` records phase receipts and atomic gzip/pickle stage objects.
Recovery is private to an immutable source/input/parameter namespace; corrupted
compressed objects fail on loading, and unfinished `.tmp` files are not reused.
The gene-result object includes coding lengths, so recovery avoids the existing
protein-only CLI resume path that discards coding-density information. This is
experimental harness recovery, not the future production checkpoint protocol.
The CLI retains its normal serial/parallel behavior; BLAS/OpenMP libraries are
bounded to one thread per worker. Cache and candidate-fingerprint overhead are
recorded separately. No production checkpoint or output is overwritten.

`run_crg_profile.py` validates fixed source input hashes before adapting the ten
raw tables, runs the CLI, and samples process-tree anonymous RSS at one second.
Summed process RSS can double-count shared anonymous pages and miss short peaks;
it is not proportional memory or Slurm page-cache accounting.
`compare_crg_profiles.py` compares ID namespaces, candidate count/fingerprint,
exact named final-bin memberships and quality metrics. Candidate fingerprints
are count plus modular sums of SHA-256 membership digests, an order-independent
probabilistic integrity check. Final memberships are compared directly as sets.
One execution per version is a case observation, not a replicated speedup or
parallel scaling estimate. Both jobs share a node, making contention another
explicit limitation. Profiler controls passed:
`profile-comparison-controls-v1 matched=1 mismatched=1 inconsistent=1 failed=0`.

Interim full-depth result: modified candidate generation completed in
271.197625 s (4m31s), yielding 6,945,894 candidates. Checkpoint overhead:
60.679311 s; fingerprint overhead: 5.874911 s. The order-independent
candidate SHA-256 sum is
`b78448f130bea63c74ffc8a3238eff4bd790c7b60c60889a17473c2a2579ef34`.
The simultaneous baseline and modified candidate scoring remain pending;
count agreement with the historical run does not prove output equivalence.
