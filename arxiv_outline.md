# Updated arXiv manuscript outline

Title: **CP-SAT and Attention-Based Deep Q-Learning for HP Protein Folding under CPU Time Budgets**.

The manuscript is [paper.tex](paper.tex). This revision incorporates the completed
Apocrita report supplied on 13 October 2026. It does not submit a new arXiv version.
The [earlier laptop draft](archive/paper_laptop_draft_20261006.tex) and dissertation
remain unchanged.

## 1. Introduction and related work

Define cubic-lattice HP folding as an optimisation benchmark. Explain direct
constraint search and attention-based DQN, including the cost of per-sequence
training. State the completed question: best verified complete folds under a
common cold-start CPU budget. Cite Berger and Leighton for cubic HP hardness;
optimisation is NP-hard, while the associated decision problem is NP-complete.
Specialised CPSP/H-core solvers remain relevant but were not benchmarked here.

## 2. Problem and methods

Specify the exact strings and common cube, origin/+x anchoring, self-avoidance,
unit backbone bonds and nonconsecutive HH contacts counted once. Energy equals
negative contacts. Describe the compact coordinate CP-SAT formulation and the
preserved project duelling attention DQN, action mask, replay, Double-DQN target,
FP32, Adam and one update per episode. Table the saved configurations, including
the proposed 3d8 architecture. Document the inherited unmasked bootstrap argmax;
no corrected-target experiment was performed. Evaluation restores training mode.

## 3. Completed protocol and provenance

Eight sequences of lengths 20, 24, 25, 36, 46, 48, 50 and 58, five seeds (0–4),
and two methods give 80 observations. Eight CPU cores and `32G` RAM were requested
per run on the homogeneous EHC selection. Scientific budget: 24 hours per run,
including training, inference, solver/model construction, checking and output.
CP-SAT can finish earlier with a full declared-cube optimality proof.

The 74 new tasks in array 30988549 all reported COMPLETED with exit code 0:0.
Six unchanged seed-0 observations on 3d4/3d6/3d8 came from the feasibility pass;
those outcomes were seen before the two-method scope was selected. No result
is counted twice. The original ten-day deadline, 16 October 2026 at 20:38:37 UTC,
was inherited rather than reset. Slurm's ten-day ceiling is not each run's
scientific budget. Resource pilots selected eight cores among tested 1/2/4/8
settings; they are engineering measurements, not learning experiments.

All 80 observations have full budget coverage or an applicable cube proof.
The cluster analyser reported 363,488 witness-record checks and zero errors.
These are records, not necessarily unique folds. The local manuscript revision
uses the supplied report; original remote fold/event artifacts were not downloaded
or independently re-audited here. Preserve both remote campaigns for archival.

## 4. Results

Primary table: means and sample SDs of five seed-wise best-fold scores, with
higher contacts better. Every method/sequence has 5/5 valid budget outcomes.
The [reported-results transcription](experiments/cluster/results/README.md)
contains the per-seed values and recomputed summaries. The contacts figure shows
sample-SD error bars, not confidence intervals.

| Sequence | CP-SAT mean ± SD | RL mean ± SD |
|---|---:|---:|
| 3d1 | 11.00 ± 0.00 | 11.00 ± 0.00 |
| 3d2 | 13.00 ± 0.00 | 13.00 ± 0.00 |
| 3d3 | 9.00 ± 0.00 | 6.40 ± 0.55 |
| 3d4 | 18.00 ± 0.00 | 10.80 ± 0.84 |
| 3d5 | 34.00 ± 0.71 | 17.00 ± 1.58 |
| 3d6 | 31.00 ± 0.00 | 15.80 ± 1.10 |
| 3d7 | 33.40 ± 0.55 | 14.60 ± 0.55 |
| 3d8 | 41.40 ± 1.14 | 17.60 ± 0.89 |

The two methods tie on 3d1/3d2. CP-SAT leads in all 30 paired seeds across
3d3–3d8, and its one-hour incumbent exceeds the corresponding RL 24-hour
incumbent in every pair. Describe this fixed benchmark result without asserting
population-wide statistical significance. All 15 CP-SAT runs on 3d1–3d3 report
OPTIMAL for the declared cube; the other 25 report FEASIBLE with loose bounds.

Separate best-search quality from greedy-policy quality. Only the five 3d1 RL
runs complete 200,000 training episodes; their millions of total episodes include
subsequent frozen-policy sampling. The other 35 runs remain undertrained.
Report episode/exploration ranges and latest nonempty within-budget evaluations.
Zero-contact completed folds remain zero; empty deadline evaluations are excluded.
Ten evaluations of one policy are not ten independent training seeds.

## 5. Published results and limitations

Compare published best scores contextually, without pooling them into CPU means.
Exact strings match Liu/Iba's rows 1–6 for our 3d1/3d2/3d3/3d4/3d6/3d7;
their best contacts are 11/13/9/18/29/29. Our 46- and 58-residue strings are absent.
Their seven-row study includes a different 60-residue string. GPU hardware,
episode schedules and project configurations differ. The archived draft's
attribution of 46-/58-residue values is historical and is not repeated here.
Primary source: https://arxiv.org/html/2504.15634v1#S3.SS1.

State CPU-specific scope, eight fixed strings, five seeds, seen/reused outcomes,
undertraining, the inherited target limitation, missing random-search controls,
and lack of specialised-solver comparison. Do not infer causes of weak learning
or general RL/CP superiority. These remain toy lattice models rather than
physical all-atom folding.

## 6. Conclusion and prospective studies

The completed finding is CP-SAT's stronger quality under the specified cold-start
CPU budget. Additional work may test the baseline's target mask, training
amortisation or an independent SciML continuous-relaxation/gradient-flow arm.
SciML decoding must produce complete valid lattice folds with its costs counted;
no SciML results or new experimental campaign are presented in this revision.

## Appendix: separate exploratory hybrid evidence

Keep the fixed-hybrid seed-0 findings (3d4/3d6/3d8: 14/22/29 contacts) separate
from the five-seed primary comparison. CP folds improve its incumbent but are
not fed into RL replay. No learned controller was tested in that arm.

Describe the local DecisionBoost constructor/controller/solver pilot as exploratory,
with its own held-out data, call caps and costs. The separate cluster replication
remains unexecuted. It does not supply evidence of equal-time superiority over
RL or CP-SAT, learning convergence or first-ever hybrid optimisation.

## Verification and release

Recompute all 16 means/sample SDs and paired/timepoint claims from the transcription;
check citation mappings and domain-scoped proofs; compile the manuscript and inspect
layout. Keep raw-data provenance distinct from the local checks. Preserve numerical
worker sources and all dissertation files. Update the repository manuscript only;
arXiv submission requires a separate explicit instruction.
