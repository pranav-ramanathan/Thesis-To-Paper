# Corrected RL resource pilot — reported 6 October 2026

Transcribed from the user's Apocrita report. Full original artifacts remain on
the cluster and have not been downloaded or independently audited here. This
is disposable engineering work, not a completed learning experiment. Job
`30587846` ran the revised seven-update workload on `ehc2`, with eight allocated
CPUs and 32 GiB, using Python module `python/3.11.7-gcc-12.2.0`.

| Scheduler item | Reported value |
|---|---|
| State / exit | COMPLETED / 0:0 |
| Wall duration | 32m 15s |
| Total CPU, allocation | 57m 55.918s |
| Total CPU, compute step | 57m 55.853s |
| Compute-step MaxRSS | 10.92G |
| Largest representative checkpoint | 526.1 MiB (3d6) |

Sixteen of seventeen cells completed. The four-thread 3d6 cell reached its
200-second limit after saving all five integrated episode/update samples; its
checkpoint was not recorded. The reports label it `cell_deadline` and exclude
it from automatic core selection and full-run extrapolation. All twelve cells
saved five completed episode timings. All eight configurations completed their
eight-thread checks and checkpoint writes, including the large replay buffers.

## Representative measurements

Each row has five completed epsilon-.25 episode samples. Episodes include
action selection, environment steps, filled-buffer insertion, one update,
independent complete-fold checking and trace writing. The model/replay are
disposable. Profiling phase-write overhead is tracked separately.

| Sequence | Threads | Median update s | Median episode s | Full cell complete |
|---|---:|---:|---:|---|
| 3d4 | 1 | 11.964 | 12.071 | yes |
| 3d4 | 2 | 7.280 | 7.354 | yes |
| 3d4 | 4 | 6.085 | 6.157 | yes |
| 3d4 | 8 | 4.210 | 4.261 | yes |
| 3d6 | 1 | 47.100 | 47.395 | yes |
| 3d6 | 2 | 29.201 | 29.408 | yes |
| 3d6 | 4 | 24.635 | 24.817 | no: cell deadline |
| 3d6 | 8 | 16.706 | 16.850 | yes |
| 3d8 | 1 | 22.304 | 23.524 | yes |
| 3d8 | 2 | 13.203 | 14.015 | yes |
| 3d8 | 4 | 10.047 | 10.650 | yes |
| 3d8 | 8 | 6.396 | 6.797 | yes |

Eight threads were fastest among the tested counts on every representative.
Even the saved four-thread 3d6 operation timings are slower than eight; the
missing four-thread checkpoint does not require another broad resource pilot
to choose the fully completed eight-thread candidate. Thread counts above
eight were not tested, so this is not a globally optimal CPU count.

Other eight-thread configuration checks completed for 3d1/3d2/3d3/3d5/3d7,
with median episode timings 0.122/0.676/2.821/16.109/5.785 seconds respectively.
These have one repetition each, so they are compatibility checks rather than
reliable duration forecasts. Checkpoint sizes were 12.1/46.8/47.2/393.5/239.4 MiB.

## Resource and throughput recommendation

Use eight allocated CPU cores on the verified `ehc` feature for a common CPU
comparison. RL's observed memory and 50% headroom support a provisional
20-GiB request (the analyser's rounded minimum is 17 GiB). The earlier CP pilot
supports a provisional four-GiB request for short solver calls. Long solver
memory growth has not been measured. Current campaign templates retain a
conservative 32-GiB allocation; verify memory stability before reducing it.

| Sequence | Episodes/hour (observed range) | Approx. episodes in 24h | Extrapolated 200,000 episodes |
|---|---:|---:|---:|
| 3d4 | 845 (825–852) | 20,300 | 236.7h / 9.9 days |
| 3d6 | 214 (212–214) | 5,100 | 936.1h / 39.0 days |
| 3d8 | 530 (522–536) | 12,700 | 377.6h / 15.7 days |

Two-hour counts are approximately 1,690/428/1,060. All forecasts extrapolate
short filled-buffer episodes at epsilon .25, exclude periodic evaluation,
checkpoint and hybrid repair costs, and do not model changing policy/episode
length or replay occupancy. They do not predict convergence or folding quality.
The within-pilot ranges are descriptive minima/maxima, not confidence intervals.

Eight-thread median episodes were about 20–31% slower than the first pilot's
values. The node and benchmark profile changed, and other node load was not
provided; no causal explanation is established. This variation is much larger
than the five-repeat range, so allow roughly 30% scheduling slack in planning
rather than treating those narrow ranges as a reliable long-run uncertainty
bound. Concurrency can further change shared-node performance.

## Recommended next experimental stage

Stop broad engineering profiling. First assess a **24-hour seed-0 feasibility
pass** on the three representatives, comparing CP-only, RL and the fixed
RL/CP hybrid at a common allocation and elapsed budget. This is nine runs,
at most 1,728 allocated CPU-hours before saving overhead at eight cores. Check
learning curves, independent best folds, greedy policy quality, checkpoint
recovery and sustained memory before freezing a five-seed campaign. The
optional learned DecisionBoost replication retains its separate corpus and
cost accounting; these primary RL forecasts do not cover that model.

If that pass supports a 24-hour common budget, the full 120-run primary campaign
uses at most 23,040 allocated CPU-hours before saving overhead. With 24 jobs
concurrent, five waves take about five compute days before queue/startup time,
leaving margin within the absolute ten-day campaign window. A 48-hour budget
already consumes about ten compute days before delays and saving, so it does
not provide that margin. Both are scheduling scenarios, not promises of job
availability or learning quality.

The fixed 200,000-episode epsilon schedule is unchanged. Its exploratory phase
may occupy most of a 24-hour run on the slowest configurations; the seed-0 pass
must assess that behaviour. Any subsequent schedule/model change would be a
separately documented method, not a silent adjustment of the reference.

Apocrita's ten-day request is the scheduler ceiling, while the application
cutoff defines the experimental budget. [Official runtime guidance](https://docs.hpc.qmul.ac.uk/using/submittingjobs/runtime/).
No new pilot or learning campaign is submitted by this recommendation.
