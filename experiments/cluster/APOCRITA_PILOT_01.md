# First Apocrita resource pilot — reported 6 October 2026

These are engineering measurements transcribed from the user's cluster reports
and diagnostic output. The full original artifacts remain on Apocrita; they
have not been downloaded or independently audited here. Source revision:
`962751e9ae38174f10ccf1368fb8d98303142e7e`. Both jobs ran on `ehc3`, with eight
allocated CPUs and 32 GiB each, using module `python/3.11.7-gcc-12.2.0`.
There was no learning-quality comparison or full seed experiment.

| Job | Method | Wall duration | Total CPU | Slurm reported memory |
|---|---|---|---|---|
| 30571792 | CP-SAT | 19m 15s | 1h 3m 59s | 654.84 MB |
| 30571793 | RL | 36m 18s | 1h 13m 9s | 10.89 GB |

All twelve CP representative cells and five other-string checks completed.
Full searches used a 15-second cap, repairs a five-second cap. Five cold
searches returned five witnesses except 3d8/one worker (four of five). The
repair seed was an easily constructed two-row snake; its quick repairs do not
establish repair cost for strong learned incumbents. Short full-search runs do
not select the optimal long-run worker count or prove unrestricted optimality.

The RL parent job exited normally, but eleven of seventeen cells were
unfinished. Every unfinished cell's launch record has `timed_out=True`, with
no reported exception in the diagnostic output. Per-cell limits were 175
seconds for representatives and 40 seconds for other configurations, including
library startup. The pilot performed 22 updates per representative cell: two
warmups, five isolated updates, and fifteen updates across three exploration
regimes. This was too much work for those limits.

| Sequence | Threads | Isolated update samples | Median update s | Epsilon .25 episode samples | Median episode s | Cell complete |
|---|---:|---:|---:|---:|---:|---|
| 3d4 | 1 | 4 | 10.001 | 3 | 10.233 | no |
| 3d4 | 2 | 5 | 6.124 | 5 | 6.214 | yes |
| 3d4 | 4 | 5 | 5.043 | 5 | 5.092 | yes |
| 3d4 | 8 | 5 | 3.177 | 5 | 3.267 | yes |
| 3d6 | 1 | 1 | 40.389 | 0 | missing | no |
| 3d6 | 2 | 1 | 24.334 | 1 | 24.838 | no |
| 3d6 | 4 | 2 | 19.679 | 1 | 19.702 | no |
| 3d6 | 8 | 3 | 12.609 | 2 | 12.908 | no |
| 3d8 | 1 | 2 | 19.697 | 1 | 20.600 | no |
| 3d8 | 2 | 3 | 11.350 | 3 | 12.078 | no |
| 3d8 | 4 | 4 | 8.728 | 4 | 9.234 | no |
| 3d8 | 8 | 5 | 4.975 | 5 | 5.647 | yes |

Other configuration checks completed for 3d2/3d3, while 3d1/3d5/3d7 reached
their shorter limits. The 3d1 worker log contains a PyTorch nested-tensor
performance warning, without a traceback. Cell launch diagnostics, rather
than that warning, establish its recorded deadline stop.

## Provisional resource interpretation

Eight threads gave the fastest observed updates for every representative.
Completed 3d4 episode measurements improve from 5.092 seconds at four threads
to 3.267 seconds at eight (about 1.56x throughput). The smaller partial samples
support eight as a candidate for 3d6/3d8, but do not complete the predeclared
five-repetition screen. Aggregate Slurm CPU efficiency includes the deliberate
1/2/4/8-thread sweep and should not select future core counts by itself.

Memory requests of 4 GiB for CP and about 20 GiB for RL are provisional. The
large-batch replay checkpoint was not reached in 3d6, and longer solver runs
may grow memory. Keep the corrected engineering pilot's 32-GiB request until
all checkpoint checks finish. A scientific comparison must separately freeze
its common CPU allocation and elapsed budget.

| Sequence | Eight-thread episode samples | Approx. episodes/hour | Extrapolated 200,000 episodes |
|---|---:|---:|---:|
| 3d4 | 5 | 1,102 | 181.5 hours / 7.6 days |
| 3d6 | 2, unfinished cell | 279 | 717.1 hours / 29.9 days |
| 3d8 | 5 | 638 | 313.7 hours / 13.1 days |

These extrapolations use the median of short filled-buffer episodes at epsilon
0.25. They exclude periodic evaluation/checkpoint costs and hybrid repairs,
and do not model changing epsilon, replay occupancy or learned trajectories.
They are neither confidence intervals nor estimates of convergence. Original
per-repeat values were not provided here, so no uncertainty interval is
invented. A 200,000-episode completion target for the longest configurations
does not fit a ten-day job under these provisional rates.

## Corrected follow-up

Repeat only the RL resource pilot, in a new directory. Each representative now
uses two warmups plus five full epsilon-.25 operations, reusing their update
durations instead of adding isolated updates: seven updates total. Separate
rollout-only probes retain exploratory/greedy and evaluation-mode timing.
Architecture, batch, FP32, optimizer and the scientific training loop are
unchanged. Revised cell caps include startup, allow the slowest representative
400 seconds, and allow other configuration checks 120 seconds. Caps plus kill
grace/report reserves fit the 50-minute application ceiling inside one hour.
No campaign or new CP pilot follows automatically.
