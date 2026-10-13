# Experiment work log

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
