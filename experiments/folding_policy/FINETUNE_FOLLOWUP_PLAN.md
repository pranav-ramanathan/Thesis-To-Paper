# Fine-tuning learning-rate controls and longer inference

## Decision and evidence

The user supplied the completed `policy_finetune_30m_01` report: 18 complete
tasks, 1,804 independently checked witness records and zero verification errors.
These are reported cluster findings, not results reproduced on this workstation.
At sixty seconds per validation sequence, Dr. GRPO, rollout-baseline REINFORCE
and DPO each exceeded continued PatternBoost on eleven sequence means, tied on
five and lost on none. Individual seed pairs can still deteriorate. Current
GRPO won seven, tied three and lost six against continued PatternBoost.

Two questions remain. The current GRPO and PatternBoost branches used a learning
rate of `3e-4`, while the other three objectives used `3e-5`. Also, continued
PatternBoost generated substantially fewer folds during the timed evaluation.
The next screen adds the missing lower-rate controls and measures both search
quality over time and quality after a fixed number of attempts.

## Frozen experiment

Use the same architecture, dataset, original two-hour parent for each seed,
optimizer moments, random states, elite pool, training code and legal masks.
The source is the completed **fine-tuning** pilot, not a new allocation run.
Copy and verify its checkpoints and receipts before submission.

| Arm | New training per seed | Starting checkpoint | Evaluation per validation sequence |
|---|---:|---|---:|
| Parent | None | Saved two-hour PatternBoost parent | 600 seconds |
| Continued PatternBoost, `3e-4` | None | Saved 2.5-hour endpoint | 600 seconds |
| Current GRPO, `3e-4` | None | Saved 2.5-hour endpoint | 600 seconds |
| Dr. GRPO, `3e-5` | None | Saved 2.5-hour endpoint | 600 seconds |
| REINFORCE, `3e-5` | None | Saved 2.5-hour endpoint | 600 seconds |
| DPO, `3e-5` | None | Saved 2.5-hour endpoint | 600 seconds |
| PatternBoost, `3e-5` | 1,800 seconds | Original two-hour parent | 600 seconds |
| GRPO, `3e-5` | 1,800 seconds | Original two-hour parent | 600 seconds |

All eight arms use seeds 0, 1 and 2 and the sixteen existing validation strings.
The new controls branch from the original parent; they do not continue the
already fine-tuned descendants. Training includes data generation and diagnostics.
Save the actual final training endpoint without checkpoint selection.

## Measurement and interpretation

The primary outcome is each seed's best independently verified complete fold
at 60, 300 and 600 seconds, including model setup and loading in the budget.
Retain negative paired differences and missing results. Evaluate every arm with
the same sampling implementation, batch size, temperature and seed convention.
No model outputs are repaired by CP-SAT.

The diagnostic compares the first **128, 512 and 1,024 attempted folds** in
sampling order. Traps consume attempts. Deadline-interrupted work cannot
establish a completed fixed-attempt prefix. An unreached prefix remains missing;
it is never assigned a zero score or filled from a later independent run.
Preserve the verified best fold, complete/trapped counts and elapsed time at
each reached prefix. This diagnostic reuses the ten-minute search rather than
launching an additional search. Prefix recording and verification overhead count
toward elapsed inference time and are reported where available.

Report each method's validity and throughput, process CPU time and available CPU
affinity, hostname and Slurm resource information. Differences at equal time can
reflect both sample quality and throughput. Equal-attempt differences help
separate those effects but do not remove training-seed uncertainty.

Compare GRPO at the two learning rates, PatternBoost at the two learning rates,
and the three candidate objectives against the lower-rate controls and unchanged
parents. Do not choose the most favorable sequence or seed after seeing results.
Three trained seeds remain three independent policy replications, even though
each is evaluated on sixteen inputs. The validation set has already informed
this follow-up; neither final test partition is accessed.

## Resources and execution

Submit one Slurm array, `0-23%24`, with up to 24 concurrent tasks and no
dependencies. Each requests four EHC CPU cores, 16 GiB RAM, and a four-hour
scheduler limit. The eighteen reused endpoints need about 2 hours 40 minutes
plus startup and reporting. The six training controls need about 3 hours
10 minutes plus startup and saving. Queue time and resource availability can
extend campaign completion time; the scheduler limit is not extra training.

| Allocation | Planned CPU-hours |
|---|---:|
| Six new 30-minute training branches, four cores each | 12 |
| 24 endpoints × 16 sequences × 600 seconds × four cores | 256 |
| New compute before startup and reporting | **268** |
| Scheduler reservation ceiling, 24 × four hours × four cores | **384** |

Previously trained checkpoints are reused physically. Scientific comparisons
still charge the parent two hours and each descendant 2.5 hours of training.
The earlier CP-SAT and greedy amortized-budget comparisons used a different
training charge and do not establish end-to-end superiority of these descendants.
This follow-up does not launch new classical baselines.

Resource conventions follow the [Apocrita EHC documentation](https://docs.hpc.qmul.ac.uk/nodes/ehc/)
and [Slurm array guidance](https://docs.hpc.qmul.ac.uk/using/arrays/).
The scripts request END/FAIL email; `HP_MAIL_USER` can specify a destination.
No jobs are submitted from this workstation. The user submits from the cluster
terminal using the command in [README.md](README.md).

## Stopping and next decision

There is no automatic retry, checkpoint selection, test evaluation, 24-hour run
or extension. Existing directories and source results are preserved. Incomplete
training or inference remains visible and excluded from completed comparisons.

If the lower-rate GRPO control closes the gap, the earlier result is evidence
for a better configuration rather than a benefit unique to Dr. GRPO or another
objective. If candidate objectives retain gains at equal inference time and
equal attempts, consider a subsequent matched tuning comparison. A change in
ranking at longer inference budgets is itself a result to retain. Freeze the
eventual final-test protocol only after this validation screen is reviewed.
