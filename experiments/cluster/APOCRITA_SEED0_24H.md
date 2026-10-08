# Reported Apocrita 24-hour seed-0 comparison — 8 October 2026

Transcribed from the user's cluster accounting output and generated report.
The complete frozen bundle, checkpoints and original events remain on Apocrita;
they have not been downloaded or independently audited in this workspace.
The cluster analyser reported nine tasks, 1,892 independently checked witnesses
and zero verification errors. This is a one-seed feasibility comparison, not
the full eight-sequence/five-seed campaign or the learned DecisionBoost study.

## Execution and scores

Array `30608441` ran all nine tasks concurrently once resources became available.
Each task had eight allocated CPU cores, 32 GiB requested memory and a 24-hour
application cutoff. Slurm reported every task COMPLETED with exit `0:0`.
Reported wall times were 24 hours plus 5–23 seconds. The module was
`python/3.11.7-gcc-12.2.0`. The frozen absolute deadline was
`2026-10-16T20:38:37.922608+00:00`.

The absent job in `squeue` is consistent with completed accounting records;
the `Invalid job id specified` response alone would not establish completion.

| Sequence | Length | CP-SAT | RL | Fixed RL + CP-SAT | Hybrid minus RL | CP minus hybrid |
|---|---:|---:|---:|---:|---:|---:|
| 3d4 | 36 | 18 | 10 | 14 | +4 | +4 |
| 3d6 | 48 | 31 | 16 | 22 | +6 | +9 |
| 3d8 | 58 | 41 | 17 | 29 | +12 | +12 |

Scores are the best validated non-backbone HH contact counts at the common
24-hour cutoff. Larger counts are better; energy is their negative. The
report's decimal values are means over a single seed, so 18.00 denotes 18
contacts, not a fractional fold score. No between-seed SD is available.
The analyser reports budget coverage for all nine rows. Follow-up diagnostics
reported FEASIBLE, not OPTIMAL, for all three full CP searches. Their final
contact upper bounds were 40, 125 and 183 respectively. These loose bounds do
not certify the witnessed scores as optima.

CP-SAT has the strongest witnessed score on every tested sequence. The fixed
hybrid has higher witnessed scores than RL on every sequence. It closes 50%,
40% and 50% of these observed RL-to-CP contact gaps respectively. Those are
descriptive within-seed differences, not significance tests or general gains.

Hybrid repairs use the same RL reference with fixed ten-residue windows every
1,000 episodes, capped at 30 seconds and charged to the same budget. Repair
folds are not inserted into replay. Better hybrid incumbents therefore do not
by themselves demonstrate a better learned policy. No learned repair controller
was evaluated in this array. These results support investigating solver help,
but do not establish a DecisionBoost advantage or RL learning convergence.

## Reported progress and learning diagnostics

| Sequence | Method | Best at 1h | 2h | 12h | 24h |
|---|---|---:|---:|---:|---:|
| 3d4 | cp_sat | 18 | 18 | 18 | 18 |
| 3d4 | rl | 8 | 9 | 10 | 10 |
| 3d4 | rl_cp_sat | 8 | 11 | 14 | 14 |
| 3d6 | cp_sat | 31 | 31 | 31 | 31 |
| 3d6 | rl | 9 | 12 | 15 | 16 |
| 3d6 | rl_cp_sat | 9 | 12 | 22 | 22 |
| 3d8 | cp_sat | 40 | 40 | 41 | 41 |
| 3d8 | rl | 14 | 16 | 17 | 17 |
| 3d8 | rl_cp_sat | 14 | 24 | 29 | 29 |

Eight of nine best-fold scores were unchanged between the supplied 12-hour
and 24-hour checkpoints. The exception was 3d6 RL, gaining one contact.
CP-SAT's one-hour incumbents already exceeded every corresponding 24-hour
RL and hybrid incumbent in this seed. This is strong evidence of the practical
gap under the tested cold-start CPU budget, not evidence of converged RL
performance or a general method ranking.

| Sequence | Method | Episodes | Updates | Final epsilon | Repairs | Checkpoint MiB | Greedy contacts near/before 12h |
|---|---|---:|---:|---:|---:|---:|---:|
| 3d4 | rl | 19,962 | 19,932 | 0.6075 | 0 | 85.3 | 0 |
| 3d4 | rl_cp_sat | 19,993 | 19,963 | 0.6071 | 19 | 85.2 | 5 |
| 3d6 | rl | 4,954 | 4,910 | 0.8837 | 0 | 306.0 | 3 |
| 3d6 | rl_cp_sat | 4,811 | 4,767 | 0.8868 | 4 | 306.0 | 0 |
| 3d8 | rl | 12,471 | 12,467 | 0.7349 | 0 | 210.3 | 1 |
| 3d8 | rl_cp_sat | 12,949 | 12,945 | 0.7263 | 12 | 210.2 | 0 |

All six runs were reported as training_complete=false: only about 2.4–10.0%
of the fixed 200,000-episode schedule was executed. Final epsilon is the random
action probability, so exploration remained approximately 61–89%. The three
pure RL counts broadly match the original short-pilot forecasts. High epsilon
does not by itself explain low greedy scores, since greedy evaluation uses zero
epsilon. All initial greedy means were one contact; the supplied later samples
are weak and variable. Full evaluation curves and the latest usable evaluations
are needed before claiming absence of learning or convergence.

Five of six supplied final evaluation records attempted zero episodes after
the cutoff, with null scores. These are empty evaluations, not zero-contact
folds or crashes. The supplied last 3d8 hybrid evaluation was nonempty at
about 23 hours: all ten folds completed, with mean zero contacts. That zero is
an observed policy outcome. The current analyser now excludes empty and
post-budget evaluations, reports the latest nonempty evaluation within budget,
and exposes attempted/completed counts and evaluation times. It retains the
original records and preserves genuine zeros and missing outcomes separately.
Ten greedy rollouts from one deterministic policy are not ten independent
training seeds. Worker status `stopped` records the planned deadline stop;
Slurm's COMPLETED/0:0 records do not mean all training episodes finished.

Static inspection also found that the preserved reference masks invalid actions
during selection but takes an unmasked argmax for the Double-DQN bootstrap
target (`vendor/rl_reference.py`, update method). That mismatch warrants a
controlled correctness/learning check. It is inherited reference behaviour,
not a confirmed cause of these results. The reference, optimizer, schedule,
and frozen executed bundle have not been changed.

## Reported resource usage

TotalCPU below uses each task's allocation record, without double-counting its
steps. Peak RSS uses the compute step `.0`. The provided `K` values are converted
to GiB by dividing by 1,048,576. CPU utilisation is TotalCPU divided by eight
times reported elapsed time.

| Task suffix | Sequence | Method | Elapsed | Total CPU hours | CPU utilisation | Peak GiB |
|---|---|---|---|---:|---:|---:|
| 0 | 3d4 | cp_sat | 24:00:05 | 190.906 | 99.42% | 1.235 |
| 1 | 3d4 | rl | 24:00:06 | 119.328 | 62.15% | 3.223 |
| 2 | 3d4 | rl_cp_sat | 24:00:07 | 119.302 | 62.13% | 3.299 |
| 3 | 3d6 | cp_sat | 24:00:07 | 190.552 | 99.24% | 1.016 |
| 4 | 3d6 | rl | 24:00:09 | 115.622 | 60.21% | 11.257 |
| 5 | 3d6 | rl_cp_sat | 24:00:23 | 117.287 | 61.07% | 11.297 |
| 6 | 3d8 | cp_sat | 24:00:06 | 190.749 | 99.34% | 1.169 |
| 7 | 3d8 | rl | 24:00:09 | 129.531 | 67.46% | 10.934 |
| 8 | 3d8 | rl_cp_sat | 24:00:07 | 128.358 | 66.85% | 10.467 |

The allocations used about **1,728.176 allocated CPU-hours**, including the
reported saving/startup elapsed overhead, and **1,301.635 utilised CPU-hours**.
TotalCPU is accumulated across cores: approximately eight CPU-days for a
24-hour CP task does not mean that task ran for eight wall days.
Field definitions: [Slurm accounting documentation](https://slurm.schedmd.com/sacct.html).

Peak memory stayed below the 32-GiB request on these three configurations.
The largest reported value was about 11.30 GiB. The 3d8 RL peak was about twice
its short resource-pilot value, reinforcing that short-run RSS was not a
long-run upper bound. These records do not prove stable memory over longer
budgets or on all eight sequences. Keep the existing 32-GiB templates for now.
RL's lower CPU utilisation alone does not justify fewer cores: the resource
pilots found eight threads fastest among 1/2/4/8 on the representatives.

## Next decision

The follow-up evidence does not justify launching the remaining 111 runs
unchanged yet. First obtain the latest nonempty greedy evaluations, inspect
the learning curves and run a bounded reference-learning audit on the cluster.
Before attributing hybrid gains to learned construction, compare it against a
cheap valid random constructor with the same repair policy and explicit solver
and elapsed-cost accounting. That control and any masked-target variant need
a separate declared protocol; they have not been run here. The learned repair
controller remains a separate study.

Do not select a winner from one seed or launch the remaining jobs automatically.
If the unchanged 24-hour protocol remains appropriate, these nine runs can be
retained as seed-0 observations and the remaining 111 primary tasks submitted
without duplicates. The current full-campaign command still creates all 120
tasks: exclusion and aggregation across the two frozen bundles must be
implemented and verified before using it to continue this experiment.
Any changed budget, model or training schedule needs a clearly separate protocol;
do not merge these 24-hour results into a changed-budget final endpoint.
The optional learned DecisionBoost replication needs its own experiment and
cost accounting. No new jobs were submitted while recording these findings.
