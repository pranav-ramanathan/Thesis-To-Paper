# Completed CPU comparison: reported results

The user supplied the final Slurm accounting and cluster analyser report on
13 October 2026. Array **30988549** contains 74 new tasks, all reported
`COMPLETED` with exit code `0:0`. The report combines them with six unchanged
seed-0 observations from `seed0_24h_01`, giving **80/80 valid budget outcomes**:
eight strings, seeds 0–4, CP-SAT and the preserved project attention DQN.

Each run received eight CPU cores, a `32G` memory request and a 24-hour
application compute budget. Full declared-cube optimality proofs permit early
completion. The inherited deadline was 16 October 2026, 20:38:37 UTC.

[JSON transcription](rl_vs_cp_24h_01_reported.json) contains per-seed 1/2/12/24-hour
contact scores, final solver statuses/bounds, reported RL counters and summary
statistics. [CSV transcription](rl_vs_cp_24h_01_reported.csv) provides the 80
contact trajectories. These are **transcriptions of user-pasted output**, not
original remote event files, checkpoints or fold coordinates. Local checks
recomputed all final means/sample SDs and the 30 paired one-hour/24-hour
inequalities. They do not independently repeat the remote geometry audit.

The cluster analyser reported **363,488 witness records checked**, with zero
verification errors. This count is not a count of distinct conformations.
All 16 sequence/method rows have five valid budget results.

| Sequence | Length | CP-SAT mean ± sample SD | RL mean ± sample SD |
|---|---:|---:|---:|
| 3d1 | 20 | 11.00 ± 0.00 | 11.00 ± 0.00 |
| 3d2 | 24 | 13.00 ± 0.00 | 13.00 ± 0.00 |
| 3d3 | 25 | 9.00 ± 0.00 | 6.40 ± 0.55 |
| 3d4 | 36 | 18.00 ± 0.00 | 10.80 ± 0.84 |
| 3d5 | 46 | 34.00 ± 0.71 | 17.00 ± 1.58 |
| 3d6 | 48 | 31.00 ± 0.00 | 15.80 ± 1.10 |
| 3d7 | 50 | 33.40 ± 0.55 | 14.60 ± 0.55 |
| 3d8 | 58 | 41.40 ± 1.14 | 17.60 ± 0.89 |

Higher contacts are better; energy is their negative. These means summarise
each seed's **best witnessed complete fold**, not its final greedy-policy score.
CP-SAT's one-hour incumbent exceeds the paired RL 24-hour incumbent in all
30 runs on 3d3–3d8. All 15 CP-SAT runs on 3d1–3d3 report `OPTIMAL` for the
declared cube; the remaining 25 report `FEASIBLE`, without proofs of optimality.

Only the five 3d1 RL runs complete the 200,000-episode training schedule.
Their millions of total episodes include subsequent frozen-policy sampling.
The other 35 RL runs are undertrained at the cutoff. The preserved unmasked
bootstrap target and different configurations/budgets prevent presenting these
results as an exact reproduction of fully trained published GPU RL.
Six feasibility observations were inspected before selecting the two-method
scope. Fixed-hybrid and local DecisionBoost pilots remain separate evidence;
SciML is a proposed future comparison and has no results in this table.

Original raw campaigns remain on Apocrita under `.cluster-runs/`; they are not
in this checkout. For further witness auditing, timing analysis or publication
archival, retain both original campaign directories and their frozen bundles.

The manuscript figure source is [cpu_24h_contacts.tex](../../../figures/cpu_24h_contacts.tex).
It plots these means with sample-standard-deviation error bars, not confidence
intervals. The archived laptop draft remains unchanged and is historical only.
