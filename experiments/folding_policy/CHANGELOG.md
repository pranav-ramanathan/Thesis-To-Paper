# Experiment work log

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
