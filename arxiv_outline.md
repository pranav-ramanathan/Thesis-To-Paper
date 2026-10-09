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

## 3. CPU comparison protocol and provenance

Use a verified homogeneous Apocrita CPU selection. Keep primary methods' CPU
allocation, common cube and elapsed budget fixed. The current main comparison
is **CP-SAT versus RL**, with five seeds × eight benchmark strings × two methods
giving **80 observations**. Six completed seed-0 observations are reused under
the unchanged 24-hour protocol; the continuation queues **74 new tasks** at
up to 24 concurrent allocations, eight cores/32 GiB each. The inherited
16 October 2026 20:38:37 UTC deadline is not reset. Four waves need about four
compute days before queue delays, with 14,208 additional allocated CPU-hours
before saving overhead. Fixed-hybrid results are secondary feasibility evidence.
The earlier generic three-method/120-run protocol remains available separately.
Exact benchmark lengths are
20, 24, 25, 36, 46, 48, 50 and 58, not the earlier generated length-eight corpus.
The saved protocol lacks a 3d8 architecture; its proposed row stays labelled.

Run separate disposable one-hour CP-SAT and RL resource pilots first. Apocrita's one-hour/ten-day advice is
about scheduler requests, not the scientific budget. Request **ten days** for
campaign jobs and apply a separate configurable cutoff, provisionally **24 hours
per primary run**. Inspect the CPU pilot before selecting that budget; freeze
the choice for prospective runs. The feasibility seed-0 scores were viewed
before selecting the two-method scope; disclose their provenance rather than
claiming all 80 observations were unseen confirmatory data. An absolute campaign
deadline expires ten days after original preparation, including queue time.

The reported Apocrita resource pilots now support eight cores among the tested
1/2/4/8 counts. The corrected RL run checked all eight configurations at eight
threads, including full replay checkpoints, with reported peak memory 10.92G.
One four-thread large-batch cell reached its engineering deadline. Stop broad
profiling and assess a separate 24-hour seed-0 pass on 3d4/3d6/3d8 before
freezing five-seed budgets. Extrapolated 200,000-episode durations at eight
threads are approximately 9.9/39.0/15.7 days; these are short-episode throughput
forecasts, with no convergence claim. See
[the reported pilot findings](experiments/cluster/APOCRITA_PILOT_02.md).

The reported 24-hour seed-0 feasibility array has now completed on the three
representatives. CP-SAT/RL/fixed-hybrid contact counts were respectively
18/10/14 on 3d4, 31/16/22 on 3d6 and 41/17/29 on 3d8. The cluster analyser
reported 1,892 checked witnesses and zero errors; Slurm reported all nine
tasks completed, with peak RSS about 11.30 GiB. These are one-seed best-search
outcomes, not greedy policy quality, a multi-seed ranking or a learned
DecisionBoost evaluation. Inspect trajectories, training progress and solver
bounds before selecting the subsequent budget. See
[the reported seed-0 findings](experiments/cluster/APOCRITA_SEED0_24H.md).

Follow-up diagnostics show that CP-SAT's one-hour incumbents already exceeded
the 24-hour RL and fixed-hybrid incumbents on all three strings. Eight of nine
best scores were unchanged between the 12- and 24-hour checkpoints. All RL
runs were incomplete (roughly 2.4–10.0% of the 200,000-episode schedule), with
final epsilon approximately 0.61–0.89 and weak sampled greedy policies.
Full-search solver statuses were FEASIBLE, with loose bounds rather than
optimality proofs. Empty deadline evaluations must be excluded from policy
summaries, while genuine zero-contact folds remain visible. The user's selected
continuation measures practical cold-start budget performance despite incomplete
training. A static baseline audit documents the inherited unmasked Double-DQN
bootstrap target; no correction is silently applied while reusing old runs.
See [the baseline audit](experiments/cluster/BASELINE_AUDIT.md). Numerical learning
diagnostics and any corrected variant remain separate future experiments.

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

Present the completed cluster seed-0 pass separately from the pending repeated
campaign: CP-SAT obtained the strongest witnessed score on all three selected
strings, while the fixed hybrid improved on RL. Attribute hybrid search gains
to the measured combined procedure; do not infer learned policy improvements
or benefits of the unevaluated learned controller from those best-fold scores.

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
