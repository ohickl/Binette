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


## Bounded scoring and retention increment — 2026-10-01

The first full comparison is now terminal: baseline `29167715` completed in
2h25m16s, first improved `29167714` in 1h30m58s, comparator `29167739` passed.
All 6,945,894 candidate fingerprints agree; final memberships and scientific
metrics match exactly (26 bins, 5,053 contigs). Curated evidence is in
`crg_full_profile_comparison_2026-10-01.json`. Peak summed process-tree RssAnon
was 50.06 versus 48.52 GiB. These are single executions on shared infrastructure,
not replicated speed or memory estimates. Earlier pending claims above are
historical and superseded by this receipt.

The next implementation replaces CLI numerical scoring with a cached feature
plan and checked, immutable contig evidence: uint32 count storage promotes to
int64 when required, sparse integer KO evidence and safe integer reductions,
and unchanged float64 model inputs. Candidate incidence and model rows are
bounded; tasks contain at most 500 bins and normally at most 65,536 total
memberships (a single larger bin remains intact). No assembly-wide dictionary
scan or pandas feature round trip occurs per numerical batch.

One model pair lives per persistent spawn worker, with default ownership capped
at four and an explicit `--quality-workers` override bounded by `--threads`.
Small default workloads use one owner. Evidence/reference arrays are read-only
mmap shared files in task-local temporary storage. Queued tasks are capped at
twice the owner count; results contain only completeness, contamination and a
uint8 model flag, updating parent objects in place. TensorFlow and native BLAS
prediction thread budgets are one per owner. Reference norms are cached and
cosine maxima reduce 1,024-row blocks, preserving the upstream multiply/divide
arithmetic, zero norms, nonfinite propagation and threshold order. Per-batch
full garbage collection is removed; normal Python GC remains enabled.

`--score-cache` defaults to `OUTDIR/.score_shards`. Atomic numeric NPZ shards
and count/SHA seals live under an evidence/candidate-order/code/model/feature/
version namespace. A shared namespace lock protects concurrent writers without
one lock inode per batch. Unsealed files are not trusted, corruption fails
closed, sealed batches restore without inference, and a missing batch alone is
inferred. Scores retain float64 precision and the model flag uses uint8. These
shards recover scoring; the isolated profiler separately retains gene/coding,
DIAMOND and candidate stage objects. Protein-only legacy CLI `--resume` still
omits coding density and is not the accepted cold-recovery mechanism. Nothing
in this increment introduces Nextflow `-resume` or deletes existing stores.

Candidates now use slots, an explicit shared cached membership key, shared
immutable empty provenance, and compatible legacy pickle reads. Generation
reuses its membership dictionary and releases rejected keys before wrapping
bitmaps. An inverted membership index creates original graph edges in the
same order as the pairwise oracle. Selection uses one occupied bitmap in
unchanged ranked order, including caller dictionary-key ties. Normal CLI output
filters quality-ineligible hybrids before size/N50; debug mode retains the
all-candidate report. Original metrics and the known exact-half N50 behavior
are preserved. TSV reports stream in the same order and byte representation.

Gene futures are bounded and translated once into the exact upstream protein
output plus compact CDS/amino/coding summaries. Coding length uses interval
union rather than a per-base float64 mask. DIAMOND annotations parse in 100k-row
chunks, retaining the required KO counters only. Python protein/KO containers
are released after typed evidence is built. Optional
`BINETTE_SCORING_TELEMETRY` records feature/inference/postprocessing timings and
worker RSS/PSS snapshots; process-tree RssAnon sampling remains in the profiler.

Focused verification: `pixi run --locked unit` passed 134 tests with the known
strict N50 xfail; `pixi run --locked equivalence` passed 21 controls, including
real model inference with 1/2/4 owners, no-hit bins, exact report bytes and
complete score restore. Partial restore, bounded memberships, overflow
promotion/rejection, coding intervals and exact protein summaries also pass.
Ruff and whitespace checks pass. The optional external upstream dataset is
absent; no full dataset suite or production pipeline qualification is claimed.

Scoring-core technical gate `29170106` completed 0:0 in 2m16s on the frozen production
image. It verifies the canonical sample-0 0.1% fixture and adds a nonempty
8-bin frozen-runtime numerical control with two owners, full restore without
model construction, and one deliberately missing shard. Receipt:

```text
binette-isolated-microgate-v2 pairs=16427 variants=2 graph_oracles=12 numeric_bins=8 workers=2 complete_restore=PASS partial_restore=PASS failed=0
```

Source manifest SHA-256:
`53094ada9bcd454ee0f553ab146e6877ee011b2ae09006688cddece0efa4ebf4`.
Manifest and gate: `crg_scoring_sources_2026-10-01.json` and
`crg_scoring_microgate_2026-10-01.json`. Intermediate gate `29170065` also
passed, but its earlier snapshot is not the final qualification receipt.
Full follow-up `29170554` runs the identical sample-13 raw-only inputs at
16 CPUs/96G/12h and four model owners. Comparator `29170555` reuses the
completed baseline under `profile_20261001_86e515d72d97`; no baseline rerun.
Current isolated root:
`/no_backup/abaud/data/secondary/rat_mag_catalog/analysis/nf-imp3_tests/binette_optimization/profile2_20261001_53094ada9bcd`.
Final score-seal hardening passed gate `29171573 COMPLETED 0:0` in 2m47s,
with the same paired/nonempty/complete/partial recovery verdict. Final source
manifest `a3210501a9f34ef74d924c1d73dd80d61955b8162488170a8bd1655c11cfba3e`
adds TensorFlow/scikit-learn compatibility versions and rejects invalid arrays
before sealing. The full benchmark preserves its earlier immutable snapshot;
feature/inference/selection code is identical. Final seal manifests/receipts
are `crg_score_seal_sources_2026-10-01.json` and
`crg_score_seal_microgate_2026-10-01.json` in fork development documentation.

Follow-up `29170554 COMPLETED 0:0 03:22:17`; comparator `29170555` PASS.
Exact final memberships/scientific metrics agree: 26 bins and 5,053 contigs.
Peak summed process-tree RssAnon is 19,798,839,296 bytes (18.44 GiB), versus
48.52 GiB for the previous improved fork, 62.0% lower. Runtime is 2.22 times
longer; four versus sixteen owners is a confound, and these are single runs.
Scoring grew from 4,589.82 to 11,407.60 s, while N50/selection fell to
9.61/2.14 s. Final phase probe `29187689` attributes 54.0% of worker phase
time to postprocessing and 39.7% to specific inference. Worker sums are not
job elapsed time. Curated receipts: `crg_followup_comparison_2026-10-02.json`
and `crg_followup_phase_totals_2026-10-02.json`.
Read-only cosine format probe `29187691 COMPLETED 0:0` preserved exact
results but improved the median isolated timing by only about 1%; this does
not explain the regression. The format conversion is now cached per batch.
Fixed real-candidate probe `29187703 COMPLETED 0:0` used the same first
65,536 candidates, immutable evidence and no score-cache reuse:

| Model owners | Scoring seconds | Peak summed RssAnon GiB |
| --- | ---: | ---: |
| 4 | 144.112 | 13.22 |
| 8 | 74.627 | 15.89 |
| 16 | 48.076 | 20.85 |

All numeric scores/model choices agree exactly. Timings include startup/mmap
staging, exclude input preparation, and are a single ordered subset experiment.
These results establish a concurrency effect, not full-depth performance.
Receipt: `crg_owner_subset_probe_2026-10-02.json`.

The new decision-only prediction path skips cosine for reduced candidates and
mean<=40/NaN candidates, whose models are chosen without similarity. The
adapter's default diagnostic call still computes all cosine values. Numerical
precision, branch order and rounded quality/model outputs are preserved.
Telemetry counts all and needed cosine rows. Focused controls passed 40 tests
(scoring unit plus the full differential module); Ruff/whitespace passed.
No full repository suite or external upstream dataset suite was run.
Frozen-runtime gate `29187745 COMPLETED 0:0 00:02:18` passed:

```text
binette-isolated-microgate-v2 pairs=16427 variants=2 graph_oracles=12 numeric_bins=8 workers=2 complete_restore=PASS partial_restore=PASS failed=0
```

Source manifest `80d805cd9e9daff8215a2345cd248c923cd306106723498703a454e6cdae7159`;
receipts `crg_decision_sources_2026-10-02.json` and
`crg_decision_microgate_2026-10-02.json`.
Full follow-up `29187776` uses sixteen model owners on 16 CPUs/96G/12h;
comparator `29187779` reuses the completed baseline. Both new full timing/
memory and exact output equivalence remain pending. Isolated root:
`/no_backup/abaud/data/secondary/rat_mag_catalog/analysis/nf-imp3_tests/binette_optimization/profile3_20261002_80d805cd9e9d`.
The CLI's conservative default remains at most four; this experiment explicitly
selects sixteen. Choose a default only after full-run performance qualification.
Production nf-imp3 pins/resources are unchanged.

## Updated-core owner scaling — 2026-10-02

User-requested probe `29187855 COMPLETED 0:0 00:10:00` uses the same
65,536 real candidates as the earlier probe and checks exact scores/models.
All five counts passed. Results include model startup/mmap staging and
telemetry, exclude input preparation, and come from one ordered subset run.
The earlier probe used a different allocation without telemetry; its wall
times do not isolate the effect of the cosine change.

| Owners | Seconds | Peak summed RssAnon GiB | Speedup versus four | Relative scaling efficiency |
| --- | ---: | ---: | ---: | ---: |
| 4 | 177.638 | 13.27 | 1.00 | 100.0% |
| 6 | 120.830 | 14.60 | 1.47 | 98.0% |
| 8 | 95.019 | 15.97 | 1.87 | 93.5% |
| 12 | 70.643 | 18.41 | 2.51 | 83.8% |
| 16 | 58.751 | 21.02 | 3.02 | 75.6% |

Similarity is needed for 46,477 of 65,536 rows (70.92%); 29.08% is skipped.
Moving from twelve to sixteen owners saves 16.8% elapsed time in this subset
and adds 2.61 GiB peak summed anonymous memory. This suggests a tradeoff,
not an accepted full-run optimum. The full six-owner `29187859`/`29187862`,
twelve-owner `29187860`/`29187863`, and sixteen-owner `29187776`/`29187779`
trials remain active. Six/twelve roots add `_w6`/`_w12` to the root above;
their source/gate links share the qualified immutable snapshot and their
outputs/stores are separate. All allocations are 16CPU/96G/12h; the owner
count differs. Full timings, exact output equivalence and memory decide the
worker default. Receipt: `crg_owner_scaling_probe_2026-10-02.json`.
