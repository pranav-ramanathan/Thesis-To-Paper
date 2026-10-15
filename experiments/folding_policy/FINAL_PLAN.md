# Frozen final comparison of reusable HP folding policies and classical search

## Decision and scope

The completed validation follow-up supplied by the user contains 24 completed
tasks, 382,129 checked witness records and no verification errors. At ten minutes
per validation sequence, lower-rate GRPO improved on lower-rate continued
PatternBoost by 0.94 contacts averaged equally across the sixteen sequences.
At 1,024 attempted folds the corresponding difference was 0.96. These are
exploratory validation findings, not test results or a confirmed algorithm ranking.

The user authorized a final comparison retaining all four fine-tuners, continued
PatternBoost, matched RL-from-scratch controls, CP-SAT and randomized greedy.
This campaign freezes the resulting settings before evaluating test folds.
It trains fresh models for every seed; it does not resume earlier validation
checkpoints. Existing campaigns, the arXiv manuscript and the dissertation are
preserved. No final-test inference or full training is run on the local laptop.

The primary question is whether an HP policy trained on other sequences benefits
from PatternBoost pretraining and subsequent fine-tuning, and how its standalone
search compares with classical search under equal inference resources. The model
generates its own final folds; CP-SAT never repairs a model output.

## Data and evaluation partitions

Reuse the original deterministic dataset, generated with seed `20261013`. Freeze
its complete data file and split hash in the campaign bundle. Reusing sequences
does not reuse any fold or model checkpoint from previous pilots.

| Partition | Sequences | Role |
|---|---:|---|
| Training | 160 | The only sequences used to generate training folds and update weights |
| Validation | 16 | Already used to select this protocol; no further tuning or checkpoint selection |
| Synthetic test | 80 | Primary final evaluation, previously unevaluated |
| Public benchmark | 8 | Separate external comparison with historically observed results |

Training lengths are 16, 20, 24, 28, 32, 40, 46, 50, 60 and 64, with H fractions
around 0.25, 0.40, 0.60 and 0.75. Lengths 25, 36, 48, 58 and 72 are absent from
training. Length 72 has appeared in validation; it is a transfer test relative
to training, not a length that the research process has never examined.
The split excludes duplicate and reversed sequences across partitions and
near-copies of the public benchmark under the existing dataset rule.

The eighty synthetic strings provide the primary held-out claim. Public strings
`3d1`–`3d8` remain separate because earlier experiments and published scores have
already been examined. Their reported reference contact counts are 11, 13, 9,
18, 35, 31, 34 and 44 respectively. These are literature context; historical
proofs in our declared cube exist for the first three only. Preserve the dataset's
source and proof annotations rather than treating every reported best count as
a proven global optimum. The [published benchmark source](https://arxiv.org/html/2412.20329v1#S3)
does not supply a new matched hardware comparison for this campaign.

All methods use the same cubic lattice, integer coordinates, self-avoidance,
unit backbone bonds, origin/+x first-bond anchors, and coordinate radius
`floor(length / 2)`. A contact is one unordered, nonconsecutive adjacent H–H
pair. Energy is the negative contact count. Incomplete folds never count as
valid outcomes; independent geometry verification recounts each retained score.

## Frozen methods and training budgets

Use five seeds, `0,1,2,3,4`. Every learned method has a 9,000-second training
lineage, including initialization, example generation and training diagnostics.
Save the actual final endpoint. There is no validation-selected or test-selected
checkpoint. Seed labels 0–2 were also used in development pilots; fresh training
does not turn those identifiers into an independent unseen validation exercise.

| Final method | Initial stage | Additional stage | Total training per seed |
|---|---|---|---:|
| PatternBoost-only | 2 h PatternBoost, LR `3e-4` | 30 min PatternBoost, LR `3e-5` | 2.5 h |
| PatternBoost → GRPO | Same 2 h parent | 30 min GRPO, LR `3e-5` | 2.5 h |
| PatternBoost → Dr. GRPO | Same 2 h parent | 30 min Dr. GRPO, LR `3e-5` | 2.5 h |
| PatternBoost → REINFORCE | Same 2 h parent | 30 min REINFORCE, LR `3e-5` | 2.5 h |
| PatternBoost → DPO | Same 2 h parent | 30 min DPO, LR `3e-5` | 2.5 h |
| GRPO from scratch | Random policy | 2.5 h GRPO, LR `3e-5` | 2.5 h |
| Dr. GRPO from scratch | Random policy | 2.5 h Dr. GRPO, LR `3e-5` | 2.5 h |
| REINFORCE from scratch | Random policy | 2.5 h REINFORCE, LR `3e-5` | 2.5 h |
| CP-SAT | None | Fresh search on each evaluation string | None |
| Randomized greedy | None | Fresh randomized construction and restarts | None |

The same-seed five PatternBoost branches share one newly trained parent, including
weights, Adam moments, RNG state and elite pool. This saves physical work; each
method is still charged its full training lineage in scientific cost accounting.
Scratch controls start without supervised updates, elite examples or a trained
checkpoint. They share the same model architecture and initialization convention.

The policy is the existing 100,998-parameter Transformer: width 64, two layers,
four heads, rotary positions, padding masks and zero dropout. It encodes the H/P
sequence and current placed/unplaced state and relative coordinates, then predicts
six lattice moves. Legal-move masking, batch size eight, temperature one, FP32,
Adam and gradient clipping remain fixed. There is no architecture search.

PatternBoost initializes examples using randomized greedy folds, performs
supervised next-move learning from a diverse elite pool, and refreshes that pool
with policy samples, suffix regrowth and randomized greedy completion. Symmetry
augmentation transforms coordinates and move labels consistently; symmetry-aware
deduplication retains diverse complete folds. CP-SAT is not part of this loop.
This is the project's HP adaptation of [PatternBoost](https://arxiv.org/abs/2411.00566).

GRPO uses group-relative, standard-deviation-normalized rewards and per-trajectory
action normalization. Dr. GRPO removes that reward scaling and uses the fixed
maximum training-action denominator. Both retain the current one-on-policy-update
per-group implementation, with no KL term. Although a clipped surrogate is
implemented, this single-update regime does not exercise repeated-update clipping.
These declared GRPO-style objectives are not a claim to reproduce every detail of
[DeepSeekMath](https://arxiv.org/abs/2402.03300) or the
[Dr. GRPO study](https://arxiv.org/abs/2503.20783).

REINFORCE uses a frozen-policy greedy-rollout baseline: the two-hour parent for
the pretrained branch and the initial random policy for its scratch control.
Baseline rollout cost counts toward training. Consequently, its pretraining
comparison concerns the whole initialization recipe, including baseline quality,
rather than isolating only actor weights. DPO uses online best/worst complete-fold
preferences relative to the frozen parent, with beta 0.1; ties and incomplete
folds do not form preference pairs. [DPO](https://arxiv.org/abs/2305.18290) is
preference optimization, not an additional RL-from-scratch condition.

The reward is completed-fold contact count, with a trap penalty of −10. Traps
remain in the policy-gradient objective; deadline-interrupted trajectories do not
become completed examples. The same algorithm settings are used for corresponding
scratch and pretrained branches. These are fixed configurations selected from
validation, not an equally extensive hyperparameter search for every algorithm.

## Inference and primary comparisons

Each method receives 600 seconds for each sequence and seed, on four EHC CPU
cores with 16 GiB RAM. Setup and model loading count toward the sequence budget.
Each policy is frozen and reusable across all 88 evaluation strings. Each
classical run starts afresh with the corresponding seed. All model outputs are
standalone; neither test-time training nor solver repair is permitted.

Record best verified contacts at 60, 300 and 600 seconds, along with the complete
improvement curve, valid-fold rate, attempted/complete folds, elapsed time,
hardware and provenance. The model prefix at exactly 1,024 attempted folds is
a secondary diagnostic. Traps consume attempts; unreached or
deadline-interrupted prefixes remain missing. CP-SAT incumbents are not an
enumeration of generated folds, so its generated-fold total remains unavailable.

Primary contrasts are each fine-tuned method versus equal-total-budget
PatternBoost-only and each pretrained RL method versus its matching scratch RL
control. Report all four fine-tuners, including deterioration, rather than selecting
only the strongest test result. Compare every method against CP-SAT and greedy at
common inference cutoffs. Show synthetic and public tables separately and describe
seen-length and held-out-length strata without retrospectively selecting favorable subsets.
Retain each sequence's H fraction for descriptive composition analysis.

For each sequence and method, report each seed's best fold, their mean, sample
SD and valid/complete/planned counts. Aggregate sequence contrasts by first
computing each seed's paired average over the declared split. The five trained
policies remain five independent training replications; 80 sequences do not make
400 independently trained policies. Aggregate differences are descriptive unless
a separately specified statistical analysis accounts for this shared structure.

Missing witnesses or incomplete budgets are not assigned zero and do not enter
completed comparisons. Early full-domain CP optimality proofs can establish
coverage at later cutoffs, never earlier cutoffs. Report bounds and gaps only
with their actual timestamps and full-domain scope. A late bound or proof must
not be silently treated as available at 600 seconds. Finding a valid fold never
constitutes a model-produced optimality proof.

The earlier 24-hour per-instance DQN versus CP-SAT campaign remains a distinct
cold-start study with its original architecture, eight-core allocation and
training budget. Its numbers are historical context, not a matched baseline for
this reusable policy experiment.

## Compute accounting

The primary comparison is equal inference resources, accompanied by training and
data-generation cost. Shared physical parent execution does not erase any final
method's scientific training lineage.

| Work | Allocated CPU-hours |
|---|---:|
| Five new two-hour parents, four cores | 40.00 |
| Twenty-five new half-hour branches, four cores | 50.00 |
| Fifteen new 2.5-hour scratch runs, four cores | 150.00 |
| Fifty endpoints × 88 sequences × 600 seconds × four cores | 2,933.33 |
| **Planned new compute** | **3,173.33** |
| Scheduler reservation ceiling | **3,380.00** |
| Scientific training charge across forty learned lineages | 400.00 |

Early CP-SAT proofs can reduce actual inference time. Slurm allocation products
are not measured CPU utilization: retain `TotalCPU` and `MaxRSS` accounting.
Record checkpoint serialization, reporting and verification overhead separately
where the implementation distinguishes them. Verification performed inside a
search consumes that search budget.

An optional deployment scenario uses `training / N + inference` per future
sequence, with `N` stated explicitly. For 9,000 seconds of training and 600 seconds
inference, the model charge is 1,725 seconds at N=8, 712.5 seconds at N=80 and
about 702.27 seconds at N=88. These are accounting scenarios, not measured solver
performance at those budgets. This campaign runs CP-SAT and greedy for 600 seconds
only and therefore makes no new equal-total-cost superiority claim at the larger
amortized budgets. Do not count five policy seeds as one inexpensive ensemble or
compare their pooled best result with one classical seed.

## Parallel execution and deadline

There are eleven arrays, each `0-4%5`, containing 55 scheduler tasks: five parent
tasks plus fifty final method/seed endpoints. Each task requests one process,
four CPU cores, 16 GiB RAM and the `ehc` constraint on the `compute` partition.

| Arrays | Tasks | Scheduler limit per task | Dependency |
|---|---:|---:|---|
| PatternBoost parents | 5 | 3 hours | None |
| Five continued/fine-tuned methods | 25 | 16 hours | Corresponding successful parent seed |
| Three scratch RL methods | 15 | 18 hours | None |
| CP-SAT and greedy | 10 | 16 hours | None |

The five branch arrays use Slurm `aftercorr` dependencies on the parent array.
Branch seed 2 can start once parent seed 2 completes successfully, without waiting
for all other parent seeds. Branch methods do not depend on each other. Scratch
RL and classical jobs can run while parents train. This follows the documented
[Slurm dependency semantics](https://slurm.schedmd.com/sbatch.html) and
[array task correspondence](https://slurm.schedmd.com/job_array.html).

At full availability, peak concurrency is fifty tasks: 200 cores and 800 GiB RAM
across nodes. The `%5` limit applies to each array, not to the whole campaign.
The 88 sequential sequence searches take about 14 hours 40 minutes per endpoint.
The longest training-plus-evaluation path is about 17 hours 10 minutes plus
startup and saving; queue delays can extend overall elapsed duration. An 18-hour
scheduler request is not an 18-hour training budget.

Freeze an absolute UTC deadline ten days after preparation. It includes queue
waiting. Workers do not begin a training segment or sequence search when its full
budget cannot fit before that deadline. Unfinished work stays incomplete, and no
automatic extension, retry, requeue or replacement seed is authorized. Inspect
failed dependencies and remaining queued tasks rather than submitting another
copy of the same experiment. Scheduler wall limits remain additional limits.

## Cluster commands

Submit from the Apocrita terminal after updating the repository:

```bash
cd /data/home/ah23335/Thesis-To-Paper &&
git pull --ff-only &&
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/final_campaign.sh \
  "$PWD/.cluster-runs/policy_final_01"
```

An optional `--dry-run` prepares a separate new directory and prints submission
commands without submitting. Use a distinct directory such as
`.cluster-runs/policy_final_dryrun_01`; a prepared directory is not a retry target.
Set `HP_MAIL_USER` to an actual preferred address before submission if needed.
The batch scripts request END, FAIL and INVALID_DEPEND notification; delivery
uses the account's configured address unless explicitly overridden.

Check jobs from the repository directory:

```bash
squeue -u "$USER"
sacct -X -j "$(cat .cluster-runs/policy_final_01/job_ids.txt)" \
  --format=JobID,State,Elapsed,ExitCode
```

Collect accounting and submit a separate job for independently checking the
saved results and generating the report:

```bash
cd /data/home/ah23335/Thesis-To-Paper &&
git pull --ff-only &&
sacct -j "$(cat .cluster-runs/policy_final_01/job_ids.txt)" --units=G \
  --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem \
  > .cluster-runs/policy_final_01/accounting.txt &&
sbatch --output="$PWD/.cluster-runs/policy_final_01/report-%j.out" \
  --error="$PWD/.cluster-runs/policy_final_01/report-%j.err" \
  experiments/folding_policy/final_report.sbatch \
  "$PWD/.cluster-runs/policy_final_01"
```

The reporting job requests one EHC core, 8 GiB and a two-hour maximum. It uses
the current reporter to verify the existing frozen campaign, without retraining,
new search or changes to its saved experiment files. Raw coordinates are released
after verification; compact verified summaries remain for aggregation. After
completion, read `.cluster-runs/policy_final_01/report.md`. Reporting resource
use is separate from the frozen training and inference budgets.

Saved report status is not a live Slurm query. Reading a partial report does not
authorize tuning, checkpoint selection or extensions against observed test
scores. Preserve the frozen complete and incomplete results, then report the
actual outcomes of every declared method.
