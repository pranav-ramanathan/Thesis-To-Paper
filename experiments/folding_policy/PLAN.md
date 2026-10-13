# One folding policy: PatternBoost pretraining and RL refinement

Status: implementation and validation pilot, 13 October 2026. This is a new
experiment; none of the previous per-sequence attention-DQN results are results
for this policy. No production campaign follows automatically.

The authorized follow-up to the completed first pilot is now specified in
[ALLOCATION_PLAN.md](ALLOCATION_PLAN.md): three seeds, two-hour training lineages,
shared PatternBoost prefixes, and actual before/after-RL validation endpoints.
It supersedes the provisional subsequent-study allocation below.

## Question and decision

Does training a single variable-length policy on diverse elite folds, then
fine-tuning the same weights with RL, improve standalone folding on sequences
excluded from training? Compare PatternBoost → RL against RL from scratch and
PatternBoost only at equal total training/data-generation budgets, then compare
their inference curves with randomized greedy construction and CP-SAT.

The first run is a small **validation-only pilot**. It tests implementation,
learning signal, throughput and allocation. It cannot establish a paper result
or select a winner confidently with two seeds and four validation strings.

## Frozen domain, partitions and references

All arms use integer coordinates on the 3D cubic lattice, unit axis-aligned
backbone bonds, self-avoidance, and one contact for each nonconsecutive H–H pair
at Manhattan distance one. The origin and first +x bond are fixed; coordinates
lie in the cube `abs(coordinate) <= floor(length/2)`, matching the earlier CPU
comparison. No incomplete fold enters a best-contact result. Complete zero-contact
folds are valid; missing witnesses are not zero. Model evaluation uses only
masked autoregressive sampling. Neither CP-SAT nor greedy search repairs a model
output during evaluation.

`dataset.py` freezes sequence identities, reversal exclusion and a SHA-256 split
fingerprint, before generating any folds. Exact benchmark strings come from the
existing cluster protocol. Training, validation and test sequences and their
reversals are disjoint. Training sequences within Hamming distance two of a
public benchmark sequence or its reversal are also excluded.

| Partition | Size | Lengths | Permitted use |
|---|---:|---|---|
| Synthetic training | 160 | 16, 20, 24, 28, 32, 40, 46, 50, 60, 64 | Greedy/CP examples, elite refresh, supervised and RL updates |
| Synthetic validation | 16 | 24, 40, 60, 72 | Checkpoint selection, pilot diagnostics, subsequent hyperparameter decisions |
| New synthetic test | 80 | 20, 24, 25, 36, 46, 48, 50, 58, 64, 72 | Final evaluation after protocol/model selection is frozen |
| Public benchmark | 8 | Exact 3d1–3d8 strings | Final external comparison only |

Each synthetic partition spans H fractions approximately 0.25, 0.40, 0.60 and
0.75. Sequences are random permutations with fixed H counts; they are not chosen
by solver score. Test lengths 25, 36, 48, 58 are absent from training, and 72
tests extrapolation beyond the longest training chain. Validation contains 72,
so claims about a previously unexamined length must distinguish training
exclusion from validation exposure. Report results by seen/unseen training
length and composition, and record nearest training-sequence similarity.

The public eight sequences have already influenced this research. They are
held out from this new model's training, but cannot retrospectively become a
previously unseen research benchmark. The new synthetic test set provides
stronger prospective evidence. Do not open either test set during this pilot.

| Public sequence | Length | Published best-known contacts | Proof status established here |
|---|---:|---:|---|
| 3d1 | 20 | 11 | CP-SAT proved 11 in the declared cube |
| 3d2 | 24 | 13 | CP-SAT proved 13 in the declared cube |
| 3d3 | 25 | 9 | CP-SAT proved 9 in the declared cube |
| 3d4 | 36 | 18 | Best reported reference; our campaign did not prove optimality |
| 3d5 | 46 | 35 | Best reported reference; our campaign did not prove optimality |
| 3d6 | 48 | 31 | Best reported reference; our campaign did not prove optimality |
| 3d7 | 50 | 34 | Best reported reference; our campaign did not prove optimality |
| 3d8 | 58 | 44 | Best reported reference; our campaign did not prove optimality |

Reference: [Espitia, Pang and Gumbart, Table 1](https://arxiv.org/html/2412.20329v1#S3).
These are published reference values, not an exhaustive claim of current world
records. The remaining optimum statuses are deliberately unresolved here.

## Identical policy and controlled training stages

Use one width-64, two-layer, four-head Transformer for every learned arm. It
observes the full HP sequence and the placed partial fold; unplaced residues
carry no future coordinates. Relative/rotary position encoding and padding
masks support variable lengths. Six logits represent absolute lattice moves.
Mask occupied sites and out-of-cube moves before normalizing probabilities.
If no legal move remains, terminate the trajectory and log a failure. The fixed
first +x move carries no learning gradient. Disable dropout for stable rollout
and policy-ratio calculations.

| Condition | Training allocation within the common cap |
|---|---|
| Randomized greedy | No learned model; equal inference-time construction/restarts |
| RL from scratch | Entire training allowance for GRPO-style updates |
| PatternBoost only | Greedy pool, supervised updates, iterative generation and local suffix completion |
| CP-SAT behavior cloning | Bounded CP-SAT fold generation and supervised updates |
| PatternBoost → RL | First half PatternBoost, second half RL on the same weights |
| CP-SAT data → RL | First half CP-SAT generation/cloning, second half RL on the same weights |

CP-SAT search is additionally an evaluation control, not a learned condition.
Time spent producing greedy/CP examples counts toward the training cap. No
method receives a free offline corpus. The two-stage conditions split their
allowance; they do not receive the full single-stage allowance twice. Log
actual stage times and counts because CPU wall-clock equality does not imply
equal optimizer steps. Keep architecture, dtype, optimizer family, sampling
temperature and group size fixed. Pair initialization seeds across conditions;
do not claim their subsequent random streams remain identical.

PatternBoost-style training retains a bounded elite pool per training sequence.
Start with greedy legal moves maximizing new contacts, randomizing ties, and
restart failed constructions within budget. Alternate next-move likelihood
updates, model proposals, and prefix/suffix greedy regrowth. Keep the best
score, admit distinct competitive folds, and preserve structural diversity.
Canonicalize lattice symmetries before pool deduplication. Augmentation must
transform both coordinates and action labels; with a fixed +x anchor, use the
anchor-preserving subgroup or re-anchor consistently. Never reverse a chain's
coordinates without also reversing its HP sequence.

RL samples groups of eight trajectories for the same training sequence. Sum
new-contact rewards and assign incomplete/trapped trajectories a strictly
negative return, below any completed fold. Include failures in the group;
resampling until every trajectory succeeds biases the objective. The initial
pilot uses outcome-based group-normalized advantages, tokenwise clipping at
0.2 and no value model. It is a **GRPO-style adaptation with KL coefficient
zero**, not a reproduction of DeepSeekMath's complete configuration. Incremental
contact rewards are logged; using their terminal sum does not constitute a
separate process-supervision experiment. Equal-reward groups produce zero
advantage and are explicitly counted. Any later diversity/entropy/PPO change
must be selected on validation and receive a new configuration version.

The handoff records matching pretraining-end/RL-start parameter hashes. It
continues the same policy object; optimizer-state treatment is recorded. Save
both final and validation-selected checkpoints, with partition hashes and
training/data costs. Select checkpoints only with the fixed validation metric;
report ties and whether selection occurred before RL made any update. Test
scores never choose checkpoints.

## Staged execution and stop rules

1. **Local correctness checks, seconds to minutes.** Independent contact oracle,
   symmetry/action consistency, mixed lengths and padding, no future-coordinate
   leakage, illegal-action probability zero, trap handling, finite gradients,
   policy-log-probability consistency, two-stage weight continuity, split
   exclusion, deadline handling and immutable bundle tests. Use tiny disposable
   synthetic fixtures, never benchmark folds. These are not performance results.
2. **Apocrita pilot.** Ten array tasks: five learned conditions × two seeds.
   Request four EHC CPU cores and 16 GiB/task, concurrency ten, 45-minute scheduler
   limit. Each task gets 20 minutes including example generation/training and
   its checkpoint-selection work, then 60 seconds inference on each of four
   fixed validation sequences. Two tasks additionally run matched greedy and
   CP-SAT controls (four minutes per control). Longest expected task is about
   32 minutes plus startup and saving; scheduler ceiling is 45 minutes. Queue
   time is additional. No final-test evaluation is launched.
3. **Inspect before extending.** Require all tasks to finish cleanly, no invalid
   accepted folds, nonempty training pools, correct stage hashes, actual RL
   updates in each combined arm, finite losses, and usable policy-gradient
   reward variation. Review peak RSS, thread utilization and folds/second.
   A failed or silent learning pipeline is repaired and re-piloted. A clean but
   inconclusive learning result justifies a longer validation study, not a
   superiority claim. Do not silently drop a weak baseline.
4. **Subsequent validation study, separately selected.** Provisional design:
   five seeds × five learned conditions × two hours training, with a fixed
   50:50 split for combined conditions. This costs 50 job-hours / 200 allocated
   CPU-hours at four cores, plus stated evaluation/control budgets. Change
   thread count, model size or training duration only after the resource pilot
   and freeze the resulting protocol before test evaluation. This stage is
   planned, not automatically submitted by the pilot.
5. **Final evaluation, once frozen.** Reuse trained policies across all eight
   public and 80 synthetic test sequences. Evaluate all registered conditions,
   not only a selected winner. A first equal-compute proposal is 300 total
   seconds per sequence per method, divided into five independent 60-second
   runs. Extend every method under comparison to common longer budgets (for
   example 30 and 120 total minutes divided over five seeds), rather than
   extending only the favored method. Pilot validation determines feasible
   final caps; no test-informed extension or early winner selection.

At maximum concurrency the pilot should complete roughly within one scheduler
allocation window after starting. Its reservation ceiling is
`10 × 4 cores × 0.75 hours = 30 allocated CPU-hours`, with at most 160 GiB
reserved across all tasks. These are requested resources, not measured usage.
The existing 24-hour DQN results are historical context, not this experiment's
RL-from-scratch control.

## Accounting and analysis

The pilot compares each method with **120 total inference seconds per validation
sequence**, split across two 60-second independent runs. For a five-seed final
comparison, every method including CP-SAT receives the same total budget, split
the same way. Never compare five model runs' pooled best with a single CP-SAT
run without accounting for the fivefold compute. Label per-run curves separately
from aggregate cumulative-compute curves. Parallel elapsed time is not the sum
of compute allocations.

Within each run, retain independently verified best complete folds at common
elapsed checkpoints and record when the witness was available. Inference
includes model loading, sampling, validation and logging; timing overhead and
post-cutoff saving are recorded. A fold finishing after cutoff cannot improve
the budget result. Report attempted generations, completed valid generations,
unique symmetry classes where measured, trapped episodes and deadline-censored
attempts separately. CP-SAT solution callbacks count incumbent witnesses, not
generated folds or internal search nodes. Label these differing counts clearly.

Report per sequence and condition: per-seed best contacts, mean and sample SD,
best across seeds with its total compute, valid-fold rate, generated-fold count,
runtime, hardware, trained-policy/checkpoint IDs, data cost and training cost.
Save best-contact-vs-cumulative-inference-time data; plot step curves with
per-seed traces and a mean/band, making censoring explicit. Include CP-SAT
status, incumbent, upper bound and `(bound-incumbent)/max(1,abs(incumbent))`
maximization gap. A high-scoring model witness proves no optimality.

Main contrasts are PatternBoost → RL minus RL-from-scratch and minus
PatternBoost-only, paired by training seed on the fixed test sequences. With
only eight public strings, report differences transparently and avoid treating
every sampled fold as an independent experimental replicate. Use the synthetic
test set for length/composition strata and seed/sequence-aware uncertainty. The
CP-data contrasts address whether pretraining source matters. Report both
absolute contact differences and fractions of published reference counts;
do not interpret a reference ratio as a certified approximation guarantee.

For a trained seed with total data/training cost `T` and inference cost `I`,
amortized cost per future sequence is `T/N + I` for explicitly stated deployment
counts such as `N=8, 100, 1000`. If deploying all five seeds as an ensemble, sum
all five training and inference costs. Report elapsed cost and allocated
core-hours separately; do not assume training is free or full utilization.
Count hyperparameter-development/pilot expense separately from a fixed model's
training cost. CPU-only results support CPU-only conclusions.

## Research basis and limits

- [PatternBoost, Charton et al.](https://arxiv.org/html/2411.00566v1): alternating
  local search and training a generator on good constructions. Ordered-chain
  suffix regrowth is an HP adaptation, not unrestricted placement of residues.
- [DeepSeekMath, Shao et al., §4.1](https://arxiv.org/html/2402.03300v1#S4.SS1):
  group-relative policy optimization without a learned critic; the clipping,
  group normalization and KL choices must be stated explicitly.
- [Lattice Protein Folding with Variational Annealing](https://arxiv.org/html/2502.20632v1):
  relevant masked autoregressive energy optimization. Its reported benchmarks
  are **two-dimensional**. A 3D implementation would be a new adapted baseline;
  its published numbers are not directly comparable to these cubic strings.
- [Apocrita job arrays](https://docs.hpc.qmul.ac.uk/using/arrays/): resource and
  concurrency requests are per array element, subject to scheduler availability.

This study tests a specific reusable-policy hypothesis. No claim of novelty,
convergence or superiority follows merely from implementing the combination.
