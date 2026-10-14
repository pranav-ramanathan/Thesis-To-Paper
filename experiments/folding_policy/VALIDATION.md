# Verification before the first HPC policy pilot

Date: 13 October 2026. These are disposable engineering checks, not scientific
method-comparison results. Neither final test partition was evaluated.

## 14 October 2026 — Learning-rate controls and longer inference

The complete package passes **158 tests in 38.353 seconds**, including 35 new
follow-up checks. Tests cover immutable historical endpoints, the two original-
parent learning-rate controls, exact attempted-fold prefixes including traps,
deadline interruption, late-witness exclusion, CPU accounting, partial results,
paired deterioration and validation-only provenance. Shell syntax and
`git diff --check` pass. No historical training/model code or dependencies changed.

A real production-architecture integration reused a disposable **two-second**
parent and five-second descendants from the previous engineering check, trained
both new lower-rate controls for **five seconds** each, and evaluated all eight
endpoint types on all sixteen validation strings for **0.75 seconds** each.
The first sampled batch matched the original sampler for each endpoint. A
separate report integration independently checked **1,680 witness records**,
including **1,019 complete ordered-prefix fold records**, with **zero errors**.
The shortened attempt prefixes were 1, 4 and 8. The report harness bypassed only
the production frozen-protocol loader to supply these explicitly tiny budgets;
checkpoint, training-receipt, geometry and prefix checks remained active. These
artifacts are marked engineering-only and do not represent scientific results.

Artifacts: `/private/tmp/hp-followup-integration-01`, with test logs in
`/tmp/hp-followup-full-tests.log` and `/tmp/hp-followup-report-integration.log`.
Frozen preparation with the full production protocol also passed using synthetic
receipt/checkpoint fixtures, and the copied standalone bundle generated an honest
pending report with 24 tasks and zero completed results. This checks packaging,
not the real cluster model files. Artifact: `/private/tmp/hp-followup-bundle-final-check`.

Independent reviews checked launcher/worker/report interfaces, sampling order,
fixed-count coverage and reused-versus-new training provenance. Ordered records
are capped at the first 1,024 attempts, and coordinate JSON uses compact encoding.
The report checks every recorded complete prefix witness, so reporting can take
longer than the previous improvement-only report. It prints verification progress
and records its own analysis time separately from inference. In-flight sequence
snapshots remain on disk even when not yet incorporated into aggregate reports.

No HPC job has been submitted from this workstation. The new scripts prepare
24 concurrent-capable tasks with only six new training branches. Historical
results, manuscript files and dissertation remain unchanged.

## 14 October 2026 — Fine-tuning algorithm screen

The complete package passes **123 tests in 31.940 seconds**, including 40 new
checks for training objectives, parent provenance, scheduling, bounded failures,
validation-only execution and independent result reporting. Shell syntax and
`git diff --check` pass. No new dependency was installed.

Numerical checks verify identical actor and Adam updates for the preserved GRPO
reference; the Dr. GRPO and REINFORCE gradient formulas; DPO preference signs,
masked likelihoods, ties and independently checked folds; and immutable frozen
reference weights. Parent actor, optimizer moments, random states and elite pools
are retained. Tests reject altered provenance, resources, software, budgets,
checkpoint counters, late witnesses and test-set access, and preserve completed
sequence results after a later subprocess failure.

A real short integration used the production 64-wide/two-layer/four-head actor,
a disposable **two-second** PatternBoost parent, **five seconds** for each of
the five continuation methods, and **0.25 seconds** on each of the sixteen
validation strings per endpoint. All five descendants and the unchanged parent
completed, and the final independent report verified **419 witness records with
zero errors**. These tiny runs check wiring and gradients, not learning quality,
runtime scaling or the real two-hour source checkpoints on Apocrita. No synthetic
short checkpoint is passed off as a completed two-hour training run.

The first disposable integration invocation hit the macOS `/tmp` symlink because
its test harness did not resolve the root path. Resolving the harness path fixed
the check; production entry points already resolve their run directories.

Independent reviews checked algorithm equations, legality masks, the frozen
greedy-policy baseline, report metadata, and source/configuration compatibility.
Diagnostics now label cumulative versus additional training timings and the
different sequence mixtures in PatternBoost versus RL diagnostic batches.

No cluster training has been submitted from this workstation. The prepared
command submits one bounded array from the user's authenticated Apocrita shell.
The historical runs, manuscript and dissertation are unchanged.

## Saved-policy inference comparison verification

The package passes **83 tests in 27.420 seconds**, including 28 new tests for
inference orchestration, execution and reporting. The new tests check immutable
source/checkpoint copies, complete training receipts and shared-parent handoffs,
fixed endpoint identity, package/CPU guards, independent Slurm arrays, partial
submission preservation, bounded child failures and no source overwrites.

Reporting checks independently recount contacts and reject invalid coordinates,
changed checkpoint metadata and incorrect budgets. They distinguish missing
folds from zero scores, preserve partial progress, keep later improvements and
proofs out of earlier timepoints, and avoid attributing full-search fold counts
to shorter prefixes. The 600 + 7,200/16 = 1,050 second accounting is tested.

Real short CPU checks exercise model loading/inference (0.35 seconds), randomized
greedy (0.2 seconds), and CP-SAT (2 seconds) on a validation string. These are
wiring checks with disposable model states, not estimates of scientific quality
or sustained performance. Frozen preparation and a report with all tasks pending
also pass using synthetic source receipts and checkpoint bytes explicitly marked
as test fixtures; no long training is simulated as scientific evidence.

Both shell wrappers and the batch script pass Bash syntax checking. The original
model, evaluation and solver implementations remain unchanged. All 18 dissertation
file hashes match their pre-change values. No cluster job was submitted locally;
the user submits from their authenticated Apocrita shell after pulling the code.

## Follow-up allocation pilot verification

The full package now passes **55 tests in 24.237 seconds** using the documented
root-level discovery command. Added checks cover byte-identical model/Adam/RNG
continuation after save/reload, restored elite pools, both 60:60 and 90:30 branch
paths, altered parent rejection, fixed-endpoint versus selected-checkpoint
evaluation, all-16 validation coverage, paired negative changes, source freezing,
matching-seed array dependencies, scheduler warning handling and partial
submission failures. Bash syntax checks pass for all new submission/report files.

An additional short integration used the production 64-wide, two-layer policy
and full frozen dataset: PatternBoost with 4/6/8-second checkpoints, two RL
continuations, and RL from scratch. All completed with positive updates.
Standalone evaluation covered all 16 validation strings for every path at a
disposable 0.25-second per-string cap, checking **222 witness records**. Such
short inference caps deliberately expose startup/deadline handling; they do not
estimate scientific performance or determine an allocation ratio. Artifacts
are in `/tmp/hp-allocation-integrated-lrofp5kn` on the development machine.

The allocation launcher also prepares and verifies a frozen twelve-task pilot
without Slurm submission. The configured real budgets remain 120-minute final
training lineages and 60-second inference per validation string and seed.
Complete code and package metadata are frozen again on Apocrita at submission.
The original 27-test pilot verification record follows for provenance.

## Implementation checks

The combined suite passed **27 tests**:

```bash
python -m unittest discover -s experiments/folding_policy -p 'test_*.py' -v
```

- Nine core checks: independent contact agreement, invalid/incomplete folds,
  all 48 symmetry keys, coordinate/action augmentation, legal-mask
  renormalization, finite gradients, padding invariance, hidden future geometry,
  reproducible sampling, traps and deadlines.
- Two dataset checks: deterministic disjoint partitions, held-out lengths and
  rejection of reversed benchmark leakage.
- Six HPC/report checks: frozen-file mutation rejection, validation selection,
  checkpoint/dataset validation, independent geometry checks and rejection of
  corrupt improvement witnesses.
- Ten training checks: group normalization, constant groups, negative trap
  returns, elite deduplication, ordered greedy suffix completion, stratified
  data generation, supervised loss reduction, finite RL updates, and actual
  PatternBoost → RL and CP-data → RL runs with matching stage handoff hashes.

The full suite used an existing macOS Python 3.12.11 environment with Torch
2.11.0, NumPy 2.4.4 and OR-Tools 9.15.6755, one CPU thread for numerical checks.
The nine core checks also passed under Torch 2.9.0 / NumPy 2.3.4, the numerical
library versions used by the existing cluster environment. The whole pipeline
has not yet been executed under Apocrita's Python 3.11/Linux environment; the
cluster pilot is the next verification stage. No packages were installed and no
external project files were edited.

## Integrated short runs

Five training CLI paths were exercised. With the default 100,998-parameter
policy and the full synthetic sequence file, 25-second RL-from-scratch,
PatternBoost-only and PatternBoost → RL checks produced valid checkpoints.
Their short-run scores are not compared.

The initial 25-second CP-data checks had only a few seconds available for
teacher generation; they correctly failed with an empty pool. No greedy
substitution was made. Repeating the CP cloning and CP → RL wiring checks on
small independent synthetic training fixtures succeeded, including nonzero
supervised/RL updates and validation-selected checkpoints. This is a distinction
between verifying a code path and establishing an adequate data-generation
budget for larger instances.

An additional teacher feasibility probe used **eight synthetic training
sequences**, lengths 16/24/40/64 with H fractions 0.25/0.75, two seconds per CP
call and four workers. All eight returned valid folds; 52 emitted witness
records passed the independent verifier. Total solver wall time was 12.55
seconds. This supports the initial two-second cap but does not guarantee that
all 160 training strings will produce examples on the cluster. The pilot saves
per-sequence attempt and elite coverage so missing examples remain visible.

Standalone greedy, CP-SAT and model inference checks ran on validation sequence
000. Their deliberately different subsecond/seconds-long caps were functional
tests, not a fair solver comparison. The evaluator checked output coordinates,
contact counts, checkpoint provenance and cutoffs. Evaluation never invoked
solver repair on a model fold.

## Cluster packaging and reporting

- `bash -n` passed for both shell submission files.
- Fresh preparation with `--dry-run` froze ten tasks, four validation IDs,
  source/config/data hashes and resource settings, without submitting a job.
- The frozen report ran successfully on an empty campaign and generated honest
  pending-task output, with no invented scores.
- The standard PGFPlots figure source compiled locally; an empty pilot produces
  empty axes. Real curves require independently verified improvement witnesses.
- Runtime checks enforce the frozen Python/package versions and CPU allocation.
  Failed subprocess timing remains in the launch record.
- Only the new experiment package and root README are changed. The dissertation
  file hashes were preserved; historical worker code, results and manuscript
  data are unchanged.

## Submission status and unresolved questions

**No HPC job has been submitted by this session.** A read-only connection to the
configured `apocrita` SSH host failed with `Permission denied (password)` in
noninteractive mode. Submit the prepared pilot from the user's already
authenticated cluster terminal using the command in README.md.

The pilot must establish actual EHC memory/throughput, adequate teacher coverage,
nonzero policy-learning signal and stable validation performance. It does not
establish convergence or superiority. The eight public strings and the 80 novel
synthetic test sequences remain excluded from pilot evaluation. No larger run
or automatic extension is submitted by these scripts.
