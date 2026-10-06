# Revised arXiv paper outline

Working title: **DecisionBoost for HP Protein Folding: Solver-Guided Decisions
and CPU Comparisons**.

The corresponding manuscript is [paper.tex](paper.tex). This is a working draft
and execution protocol, not a claim that the cluster experiments have finished.
The previous laptop manuscript is preserved verbatim in
[archive/paper_laptop_draft_20261006.tex](archive/paper_laptop_draft_20261006.tex).
No dissertation files are part of this revision.

## 1. Introduction

Introduce HP folding as a discrete optimisation benchmark, existing exact and
learning approaches, and the cost of per-sequence RL training. The central
hypothesis is that a transformer constructor and learned decision controller can
use bounded solver feedback effectively. Distinguish this contribution from
hybrid optimisation generally; do not claim first-ever use without a defensible
literature audit.

## 2. Problem and methods

- Define complete self-avoiding cubic-lattice folds and contacts counted once,
  with `i < j`, excluding consecutive residues. Energy is negative contacts.
- Specify the exact problem domain for each experiment, anchoring, independent
  checker, missing-solution handling, and scope of optimality certificates.
- Describe the compact coordinate CP-SAT model rather than presenting the old
  dense occupancy model as the new executable implementation.
- Describe the bundled project attention DQN accurately: five-feature residue
  observation, five relative actions, duelling transformer, prioritised replay,
  Double DQN, fixed schedule, and restoration of training mode after evaluation.
  The saved project settings are not labelled as exact published reproduction.
- Define the fixed RL-plus-CP hybrid as an additional baseline; it does not
  include the learned DecisionBoost controller or CP feedback in RL replay.
- Define DecisionBoost's frozen open encoder, small constructor, repair-region
  controller, bounded CP teacher and verified feedback. Separate known witnessed
  repair gains from exact action values. Include fixed/static/random/cheap controls.

## 3. Predeclared CPU campaign

Use a verified homogeneous Apocrita CPU selection. Keep primary methods' CPU
allocation, common cube and elapsed budget fixed. Five seeds × eight benchmark
strings × three methods gives **120 primary runs**. Exact benchmark lengths are
20, 24, 25, 36, 46, 48, 50 and 58, not the earlier generated length-eight corpus.
The saved protocol lacks a 3d8 architecture; its proposed row stays labelled.

Run separate disposable one-hour CP-SAT and RL resource pilots first. Apocrita's one-hour/ten-day advice is
about scheduler requests, not the scientific budget. Request **ten days** for
campaign jobs and apply a separate configurable cutoff, provisionally **24 hours
per primary run**. Inspect the CPU pilot before selecting that budget; freeze
the choice before viewing final comparison outcomes. An absolute campaign
deadline expires ten days after preparation, including queue time.

The reported Apocrita resource pilots now support eight cores among the tested
1/2/4/8 counts. The corrected RL run checked all eight configurations at eight
threads, including full replay checkpoints, with reported peak memory 10.92G.
One four-thread large-batch cell reached its engineering deadline. Stop broad
profiling and assess a separate 24-hour seed-0 pass on 3d4/3d6/3d8 before
freezing five-seed budgets. Extrapolated 200,000-episode durations at eight
threads are approximately 9.9/39.0/15.7 days; these are short-episode throughput
forecasts, with no convergence claim. See
[the reported pilot findings](experiments/cluster/APOCRITA_PILOT_02.md).

Record quality at one, two, twelve and twenty-four hours, training completion,
greedy policy outcomes and best validated search folds separately. If the fixed
200,000-episode training finishes early, identify the subsequent frozen-policy
sampling phase. Solver proofs are valid only for the searched domain.

Analyse per-seed variability and paired comparisons. Include completion/coverage
counts and missing witnesses; do not turn missing solutions into zero. Show
actual compute time, queue time and allocated CPU-hours. Interrupted jobs preserve
partial findings. Checkpoint/resume does not reset the deadline or refund lost work.

## 4. Separate fresh-sequence DecisionBoost replication

Five independent pipelines; frozen train/validation/test split excluding all
earlier corpus reversal equivalents. Train lengths 16–36; test lengths 16–58.
The 72 fresh test sequences × five seeds produce 360 case comparisons. The
optional bundle adds five jobs to the primary campaign, **125 jobs total**.

Keep the tested architecture and single-worker CP caps fixed. Compare learned
repair selection against fixed ten-residue suffix repair and training-selected
static selection as the main controls, with random and cheap controls as secondary.
Use the original complete reachable coordinate domains. This is a separate
call-budget experiment, not a fourth row in the primary restricted-cube elapsed
comparison. Encoder/training/teaching costs and online costs require separate
accounting; pretrained encoder training cost is unknown.

## 5. Results

Keep the completed local pilot in a clearly labelled motivating subsection.
Its five-pipeline improvement over static selection does not establish superiority
over RL, an equal-time end-to-end advantage, or learning convergence.

Populate new primary tables only from verified cluster outputs. Present contact
trajectories, completion fractions, time-to-target with censoring, seed variability,
and domain-scoped solver bounds. Present the separate DecisionBoost comparison
and cost breakdown independently. Decimal scores are means of integer contacts.

Keep historical MPS measurements and original-paper values as contextual results
with precise source and budget labels. They cannot be pooled into new CPU means
or used as an equal-time baseline. Verify earlier manuscript reproduction claims
against artifacts before reusing them.

## 6. Discussion

Discuss CPU-specific practicality, training amortisation across new sequences,
what the controller learns, and failure cases. Identify specialised CPSP/H-core
results as a strong domain reference with their own offline work and coverage.
Do not equate a general-purpose CP-SAT comparison with the frontier of exact HP
solving. State the limited benchmark size and difference between lattice folding
and physical protein folding.

## Reproducibility

Include source hashes, exact sequence/configuration JSON, model revision, CPU
architecture, dependency versions, random streams, checkpoint state, deadline
rules, independent witness validation and scripts. Cluster instructions:
[experiments/cluster/README.md](experiments/cluster/README.md).

Official documentation checked 6 October 2026:

- [Runtime recommendation and hard limits](https://docs.hpc.qmul.ac.uk/using/submittingjobs/runtime/).
- [EHC hardware and per-job core limit](https://docs.hpc.qmul.ac.uk/nodes/ehc/).
- [Slurm partitions](https://docs.hpc.qmul.ac.uk/using/submittingjobs/partitions/).
- [Node constraints](https://docs.hpc.qmul.ac.uk/using/submittingjobs/constraints/).
- [Job arrays](https://docs.hpc.qmul.ac.uk/using/arrays/).
- [Python distributions](https://docs.hpc.qmul.ac.uk/using/python_distributions/).
