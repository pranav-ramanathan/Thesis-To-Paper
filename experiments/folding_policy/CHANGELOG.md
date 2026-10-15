# Experiment work log

## 2026-10-15 — Bound final-report memory and schedule analysis

The final report process was killed after checking 885,786 witness records.
Inspection found that it retained every checked ordered-prefix coordinate array
across evaluation tasks. This explains growing memory use; the shell's `Killed`
message alone does not establish the operating system's reason for termination.

- Release raw fold arrays after independent verification and before reading the
  next evaluation file. Retain compact timing, score, bound and count summaries.
- Verify retained report values against the prior behavior and check that all
  earlier tasks' 1,024-attempt arrays are released before loading another task.
- Add an analysis-only Slurm wrapper requesting one EHC core, 8 GiB and at most
  two hours. Update collection commands to run full verification on a compute node.
- Preserve experiment outputs, frozen bundles, scoring and missingness rules.
  The separate failed CP-SAT seed still requires its error log for diagnosis.

## 2026-10-14 — Frozen final policy comparison

The completed validation follow-up reported 24 tasks and 382,129 verified witness
records without errors. The user approved a final comparison of all four
PatternBoost fine-tuners, continued PatternBoost, three matched scratch RL
controls, CP-SAT and randomized greedy.

- Freeze five seeds, the existing architecture and train/test partition, and
  fixed training endpoints. Fresh two-hour parents feed half-hour branches;
  scratch RL receives 2.5 hours. Endpoint learning rates are fixed at `3e-5`.
- Evaluate 80 synthetic test sequences and eight public benchmarks separately,
  for ten minutes per method/seed/input. Report 60/300/600-second outcomes,
  1,024-attempt diagnostics, length-transfer strata and seed-level contrasts.
- Submit eleven arrays: 55 scheduler tasks, up to 50 concurrent endpoints, four
  EHC cores and 16 GiB each. Parent dependencies match the seed. Planned compute
  is 240 training + 2,933.33 inference = **3,173.33 allocated CPU-hours**.
- Verify complete training lineage, runtime/source/data hashes and fold geometry;
  preserve partial failures, negative differences and missing outcomes.
- Local verification uses short synthetic fixtures. A real train-to-report test
  covers parent, branch, scratch, model inference, greedy and CP-SAT. The frozen
  55-job preparation and pending 100-split report also pass without search.
- Final test folds remain unevaluated locally. The user submits on Apocrita.
  Existing scientific results, manuscript and dissertation are preserved.

Protocol and commands: [FINAL_PLAN.md](FINAL_PLAN.md), [README.md](README.md).

## 2026-10-14 — Fine-tuning controls and longer inference

Objective: test whether a tenfold smaller learning rate explains part of the
fine-tuning pilot's gains, and separate quality per attempt from sampling speed.
The user's completed pilot reported 18 tasks and 1,804 checked witness records
without errors. Dr. GRPO, REINFORCE and DPO each beat continued PatternBoost on
eleven validation sequence means and tied on five; these are screening results.

- Reuse all eighteen saved endpoints. Train only three PatternBoost and three
  GRPO controls for thirty minutes at `3e-5`, from their original two-hour parents.
- Evaluate all 24 endpoints for ten minutes on each of sixteen validation
  strings. Compare 60/300/600-second search outcomes and exact ordered prefixes
  of 128/512/1,024 attempts, including traps. Check every recorded complete prefix
  fold independently and expose missing or interrupted prefixes.
- Preserve source code, data, parent and endpoint receipts, runtime versions,
  Adam/RNG/elite handoffs and fixed checkpoint identity. Record CPU/affinity and
  verification/write overhead. Reports retain negative paired differences.
- One array allows 24 concurrent tasks, each four EHC cores, 16 GiB and four
  hours. Expected longest compute path is about 3 hours 10 minutes plus overhead.
  New planned allocation: 12 training + 256 inference = **268 CPU-hours**.
- All **158 tests pass**. The tiny production-architecture integration verified
  **1,680 witness records with zero errors** across eight endpoint types; those
  disposable runs have no scientific-result status. Frozen preparation and
  pending reporting pass separately using clearly labelled protocol fixtures.
- No jobs submitted locally, no final-test evaluation or automatic extension.
  Prior code/results and the dissertation remain unchanged.

Protocol and commands: [FINETUNE_FOLLOWUP_PLAN.md](FINETUNE_FOLLOWUP_PLAN.md),
[README.md](README.md).

## 2026-10-14 — Fine-tuning from shared PatternBoost parents

Objective: screen continued PatternBoost, current GRPO, Dr. GRPO, REINFORCE with
a frozen greedy-policy baseline, and DPO from the same saved two-hour parents.
The completed inference pilot supplied by the user motivates this screen:
PatternBoost remained stronger than the current hybrids on most sequence means.

- Added a separate training module, preserving historical code and actor
  architecture. Restore actor/Adam/RNG/elite state; freeze reference weights;
  charge example generation, reference computations and diagnostics to training.
- Freeze three parents and submit 15 half-hour branches plus three unchanged
  parent evaluations, one array with up to 18 concurrent four-core tasks. Every
  endpoint receives one minute on each of sixteen validation strings.
- Independently verify fold witnesses and checkpoint provenance; report paired
  changes against both parent and continued PatternBoost, retaining deterioration.
  Missing or unfinished observations remain explicit.
- This is a fixed-configuration screen with disclosed learning-rate differences,
  not a matched tuning study or algorithm-superiority claim. Both final test sets
  remain unevaluated. There is no automatic longer campaign.
- All **123 tests pass**; the short production-architecture integration completed
  all five objectives and parent inference, with **419 verified witness records
  and zero errors**. These disposable checks have no scientific result status.
- Planned new allocation is 49.2 CPU-hours before overhead; each task requests
  four EHC cores, 16 GiB and 75 minutes. No job was submitted locally.

Protocol and commands: [FINETUNE_PLAN.md](FINETUNE_PLAN.md), [README.md](README.md).

## 2026-10-13 — Saved-policy inference comparison

Objective: reuse the completed two-hour allocation endpoints and compare their
standalone search with fresh CP-SAT and randomized greedy searches. The user
reported all 12 allocation jobs complete, 18 evaluated endpoints, and 1,634
verified witness records without errors. Those are user-supplied cluster results;
the new comparison has not yet run.

- Added frozen checkpoint/data/source provenance checks and two independent
  arrays covering 18 tasks, three seeds and all 16 validation strings.
- Models receive 600 seconds per sequence; CP-SAT and greedy receive 1,050.
  Reports show common 60/300/600-second inference points and the explicit N=16
  amortized training-cost comparison. No model training or output repair occurs.
- Retain actual fixed endpoints, source model/solver implementation and all
  source results. Both final test sets remain excluded. No automatic follow-up.
- Full suite: **83 tests passed in 27.420 seconds**, including 28 new inference
  checks. Real short model, greedy and CP-SAT wiring checks passed. Independent
  cross-review found no source-receipt or worker/report interface mismatch.
- Frozen preparation and pending reporting passed using clearly labelled
  synthetic receipt/checkpoint fixtures; these do not establish model quality.
  Shell syntax, whitespace checks and all 18 dissertation file hashes passed.
- New search allocation is up to 240 CPU-hours before overhead; scheduler
  reservation ceiling is 264 CPU-hours. No jobs submitted from this workstation.

Protocol and submission instructions: [INFERENCE_PLAN.md](INFERENCE_PLAN.md).

## 2026-10-13 — Two-hour shared-pretraining allocation pilot

Objective: implement the authorized three-seed comparison of PatternBoost 120
minutes, PatternBoost 60 → RL 60, PatternBoost 90 → RL 30, and RL 120 minutes.
The user reported all ten first-pilot jobs complete, with 419 verified witness
records and zero errors. Those pilot results support further validation work;
they do not yet isolate a positive contribution from RL fine-tuning.

- Added a separate training runner with complete Adam, RNG, model and elite-pool
  snapshots, fixed initial-example allowance, and verified shared-parent branches.
- Added four dependent Slurm arrays, source/data/software freezing, distinct
  per-arm wall limits, email requests and failure-preserving submission records.
- Evaluate actual pre-RL/post-RL states on all 16 validation sequences. Keep the
  original pilot's selected-checkpoint evaluator intact through an explicit
  validator argument. No test-set evaluation or automatic campaign extension.
- Added independent endpoint verification, paired differences that preserve
  deterioration, full budget/lineage checks, and physical versus per-method cost
  accounting. The matched 120-minute lineages physically require 66 CPU-hours of
  training; endpoint evaluation adds 19.2 CPU-hours before overhead.
- Full suite: **55 tests passed in 24.237 seconds**, including exact continuation
  and both branches. An initial test invocation used the wrong working directory;
  the documented root-level command passes. Slurm-mock isolation was corrected
  after a full-suite check exposed interaction with macOS CPU discovery.
- Short production-architecture integration completed PatternBoost, both RL
  branches and scratch RL, followed by all 16 validation strings per path:
  **222 witness records independently verified**, with no final-test inference.
  These deliberately tiny budgets are wiring checks, not learning results.
- Shell syntax and frozen preparation checks passed. No cluster job was submitted
  from this workstation; the user will submit from their authenticated shell.

Protocol and submission instructions: [ALLOCATION_PLAN.md](ALLOCATION_PLAN.md).
Next decision: inspect the new validation results before choosing a longer run.

## 2026-10-13 — Initial bounded policy pilot

Objective: compare PatternBoost → RL with single-stage controls using one
variable-length Transformer and standalone inference.

- Frozen the prospective plan, common lattice rules and disjoint sequence
  generator, including reversal exclusion and entirely untrained test lengths.
- Parallel implementation of the policy/geometry, elite-pool and GRPO-style
  training, and immutable Apocrita array pilot.
- Confirmed existing local Python environments contain Torch and OR-Tools;
  no dependency installation needed for local correctness checks.
- Read primary PatternBoost, DeepSeekMath and variational-annealing papers.
  The annealing paper's reported benchmark is 2D; this is a proposed 3D adaptation.
- Configured Apocrita SSH host exists, but noninteractive authentication returned
  `Permission denied (password)`. Remote pilot submission needs the user's
  authenticated cluster shell. No cluster job has been submitted by this agent.

Verification and remaining questions are recorded in `VALIDATION.md`. The next
scientific decision follows the validation-only cluster pilot; final test
evaluation and longer campaigns are not automatic.
