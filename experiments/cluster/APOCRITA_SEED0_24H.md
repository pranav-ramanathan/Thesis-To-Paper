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
The analyser reports budget coverage for all nine rows. Proof status and final
solver bounds were not included in the pasted table; do not label these scores
as proven optima.

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

Inspect the saved 1/2/12/24-hour trajectories, greedy policy evaluations,
episode/update counts, final epsilon, repair counts and full-search CP bounds.
These distinguish policy learning from exploratory/repair search and reveal
whether useful improvement was still occurring late in the run.

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
