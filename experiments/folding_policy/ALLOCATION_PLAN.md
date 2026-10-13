# Two-hour training-allocation pilot

This is the next authorized validation experiment after Apocrita array 31406208.
It tests whether RL fine-tuning improves a PatternBoost-trained policy and whether
continued PatternBoost is a better use of the same additional training time.
The earlier pilot showed useful pretraining but did not isolate a positive RL
contribution. Its combined condition spent only half its total allowance on
PatternBoost, and its validation selector could return a pre-RL checkpoint.

## Frozen comparisons

Use three independently initialized policy seeds (0, 1, 2), the existing
100,998-parameter width-64/two-layer/four-head Transformer, FP32, and four EHC
CPU cores per task. Retain the existing Adam learning rate, group size eight,
masked GRPO-style objective, and training algorithms. No new architecture,
entropy term, PPO update, or solver repair is introduced.

| Arm | PatternBoost including example generation | RL | Training lineage |
|---|---:|---:|---:|
| `patternboost` | 120 minutes | 0 | 120 minutes |
| `balanced` | 60 minutes | 60 minutes | 120 minutes |
| `more_pretrain` | 90 minutes | 30 minutes | 120 minutes |
| `rl_scratch` | 0 | 120 minutes | 120 minutes |

For each seed, execute one shared PatternBoost trajectory and save full training
states at 60, 90, and 120 minutes. Restore the 60-minute state for the balanced
RL branch and the 90-minute state for the more-pretraining branch. The saved
state includes the model, Adam state, Python and Torch random states, elite
pool, counters, configuration, and dataset identity. Verify the branch's
starting parameter hash against its parent. A branch starts with the parent's
weights and optimizer state, without reinitializing or choosing a better seed.

Initial greedy example generation has one fixed cap of 15 minutes, charged
within the shared PatternBoost training clock; it can end early when its pool
is full. The cap does not change with the eventual branch length. Subsequent
elite refresh and supervised updates are charged as training. Initialization
and resume loading consume the respective task's training allowance. Record
operation-boundary overrun and checkpoint-writing overhead separately rather
than representing a wall-clock cap as an exact update count.

## Evaluation and interpretation

Evaluate the actual 60/90/120-minute PatternBoost states, both actual post-RL
states, and the actual RL-from-scratch endpoint. Use all **16 validation strings**
(lengths 24/40/60/72 crossed with four H fractions), with 60 seconds of standalone
sampling per state, sequence, and seed. This expands the first pilot's four
diagonal validation examples to separate length from composition. Each final
arm receives 180 total inference seconds per sequence across three seeds.

These are fixed training endpoints, with no best-checkpoint replacement. A
separate evaluation entry point explicitly labels them as such; the original
pilot's requirement for validation-selected deployment checkpoints is preserved.
Endpoint sampling includes model loading, and an independent geometry/contact
verifier checks best folds and every improvement record. No incomplete fold
counts as a result, no missing fold becomes zero, and no output receives
CP-SAT or greedy repair. Intermediate evaluations do not consume training RNG.

The report retains individual seeds, means and sample SD, generated-fold counts,
valid-fold rates, elapsed inference time, and improvement curves. For every
validation sequence and matching seed, compare:

1. Balanced post-RL against its exact 60-minute PatternBoost starting state.
2. More-pretraining post-RL against its exact 90-minute starting state.
3. Each combined 120-minute lineage against continued PatternBoost at 120 minutes.
4. The combined arms against RL from scratch, and against each other.

Negative changes remain visible. Missing paired endpoints are excluded from a
paired contrast and their counts are shown. Sharing a pretraining trajectory
creates paired, correlated arms; only the three independently trained seeds
are independent repetitions. These are exploratory validation findings, not
proof of the best allocation at 24 hours or a final benchmark ranking.

No final synthetic test sequence or public 3d1–3d8 sequence is evaluated. No new
CP-SAT comparison is run in this allocation pilot: its purpose is to isolate
the training-stage decision. The earlier validation controls remain context;
a subsequent solver comparison must match sequence identities and inference
budgets and report training cost separately.

## Scheduling and resource accounting

Submit four three-element arrays: PatternBoost and RL from scratch independently,
then the two RL continuation arrays with `aftercorr` dependencies on the
corresponding PatternBoost seed. Each uses four EHC cores, 16 GiB, and END/FAIL
mail requests. A failed parent cannot silently launch a substitute branch.
The driver records every accepted array ID and preserves partial submissions.

| Array | Tasks | Scheduler limit per task |
|---|---:|---:|
| PatternBoost plus three endpoint evaluations | 3 | 210 minutes |
| RL from scratch plus endpoint evaluation | 3 | 150 minutes |
| Balanced RL continuation plus endpoint evaluation | 3 | 90 minutes |
| More-pretraining continuation plus endpoint evaluation | 3 | 60 minutes |

Training physically executes `3 × (120 + 120 + 60 + 30)` minutes = 16.5 job-hours,
or **66 allocated CPU-hours**. Eighteen endpoint evaluations each contain 16
one-minute searches: 4.8 job-hours, or **19.2 CPU-hours**. Expected useful-budget
allocation is therefore 85.2 CPU-hours before startup/checkpoint overhead;
the summed scheduler reservation ceiling is **102 CPU-hours**.

For scientific per-method accounting, each final policy has its full two-hour
training lineage, including the shared prefix. Executing every arm separately
would cost 24 job-hours / 96 allocated CPU-hours of training. Do not present
shared execution savings as a cheaper individual training method.

Parents evaluate their three states after training, before their dependent
arrays begin. With concurrent starts and immediate scheduling, the longest
path is approximately `120 + 48 + 60 + 16 = 244` minutes, plus startup/saving.
Plan for roughly four hours and ten minutes after jobs start; queue delay is
additional. Requested limits allow overhead but do not extend training budgets.

All runs are frozen into a fresh directory before submission. No automatic
retry, extension, 24-hour run, or production campaign follows this pilot.

## Submission

From the authenticated Apocrita repository checkout:

```bash
git pull --ff-only
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/allocation_pilot.sh "$PWD/.cluster-runs/policy_allocation_2h_01"
```

Use `--dry-run` with a different new directory to preview without submission.
See README.md for monitoring and collection commands.
