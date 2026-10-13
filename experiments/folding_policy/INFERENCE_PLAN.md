# Saved-policy inference comparison

Compare the four actual endpoints from the completed two-hour allocation pilot
with fresh CP-SAT and randomized greedy searches. This is a validation experiment
using existing trained policies; it performs no training or checkpoint selection.
The user's reported allocation pilot completed all 18 endpoints with 1,634 checked
witness records and no verification errors. Its results do not establish whether
the shared policies outperform either nonlearned baseline under comparable costs.

## Frozen methods and budgets

Use all 16 validation strings and seeds 0, 1 and 2. Each task requests four EHC
CPU cores and 16 GiB. The model architecture, weights, sampling temperature,
batch size, legal-move mask and fold domain are inherited from the allocation
pilot. Each trained seed is a separate shared policy across all 16 strings.

| Arm | Saved endpoint | Search per sequence and seed |
|---|---|---:|
| PatternBoost | `stage_7200.pt` | 600 seconds |
| Balanced: PatternBoost 60 min then RL 60 min | `final.pt` | 600 seconds |
| More pretraining: PatternBoost 90 min then RL 30 min | `final.pt` | 600 seconds |
| RL from scratch | `final.pt` | 600 seconds |
| CP-SAT | Fresh full-domain search | 1,050 seconds |
| Randomized greedy | Fresh construction and restarts | 1,050 seconds |

Model loading and setup are charged within each per-sequence inference budget.
No solver or greedy repair is applied to model outputs. Complete self-avoiding
folds only are scored, using nonconsecutive HH contacts on the common cubic
lattice. CP-SAT may stop early after proving optimality in the declared full
cube. Its incumbents are not a count of all generated folds.

## Two comparisons from each search

1. **Equal inference allowance:** best verified contacts at 60, 300 and 600 seconds.
2. **Training charged over a declared workload:** at N=16 future sequences, each
   model seed has 7,200/16 = 450 seconds of training charged per sequence. Compare
   its 600-second inference result with the baselines at 1,050 seconds. This is
   an amortized accounting scenario, not a claim that training costs disappear.

Each model lineage is charged the full two hours, including example generation,
even though the original run shared pretraining between arms. For a different N,
the model cost is 7,200/N + inference seconds. The saved baseline curves only
support comparisons within their measured 1,050-second horizon. Do not infer
single-instance, 24-hour, or larger-workload outcomes from these caps.

Keep all three seeds separate; do not pool their best folds into one method run.
Report mean, sample SD, valid/complete/planned counts, individual seeds,
improvement curves, generated-fold counts and valid-fold rates. Show CP-SAT's
full-domain bounds and proof status with their recording times. A proof recorded
after an earlier timepoint does not retroactively prove optimality at that point.
Missing and incomplete outcomes remain visible and are never assigned zero.

This is a comparison of equal allocated hardware and time. The greedy program
is serial; Slurm CPU utilization and memory accounting remain separate from
allocated CPU-hours. Neither policy seeds nor validation sequences are pooled
as extra independent training seeds. No significance or general superiority
claim follows automatically from a three-seed validation screen.

## Preservation and scheduling

Before submission, verify the source allocation manifest, completed training
receipts, endpoint hashes and shared-parent provenance. Copy the frozen original
code/data and required checkpoints into a fresh run directory, adding only the
new inference orchestration/reporting files. Preserve source results. Software
versions and resource settings are checked again on the compute node.

Submit two independent job arrays: 12 model tasks (three-hour scheduler limit)
and six baseline tasks (five-hour limit). Each task handles all 16 validation
strings for its arm and seed. Up to 18 tasks can run concurrently, subject to
cluster availability; there are no training dependencies.

- Model search: 12 x 16 x 600 seconds = 32 summed job-hours.
- Baseline search: 6 x 16 x 1,050 seconds = 28 summed job-hours.
- New evaluation budget: **240 allocated CPU-hours** before overhead, lower if
  CP-SAT proves optimality early. Scheduler reservation ceiling: **264 CPU-hours**.
- With all tasks starting together, model search takes 2 h 40 min per task and
  baseline search up to 4 h 40 min, plus process and reporting overhead.

The final public eight-sequence benchmark and synthetic final test split remain
unevaluated. No automatic retries, training, 24-hour campaign, or test evaluation
follow this comparison. A failed task preserves its completed sequence records.

## Decision after the pilot

Assess whether policy search remains competitive as the baselines receive more
time, and whether it remains competitive after the stated training charge.
If PatternBoost is competitive but the combined endpoints deteriorate, investigate
fine-tuning stability from the saved parent states. If either nonlearned baseline
comfortably wins, reconsider the training/search design before extending budgets.
Freeze a final method and evaluation protocol before opening the held-out tests.
