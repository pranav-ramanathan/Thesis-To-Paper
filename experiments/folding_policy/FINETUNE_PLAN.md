# Fine-tuning after PatternBoost: bounded algorithm screen

## Question and evidence

Does a different fine-tuning objective improve the saved variable-length folding
policy more than continued PatternBoost under equal additional CPU time?

The completed validation inference pilot reported 18 complete tasks and 2,391
independently checked witness records with no verification errors. At ten minutes
of inference, the balanced hybrid's sequence mean exceeded PatternBoost on one
of 16 strings, tied on four, and fell below it on eleven. The more-pretraining
hybrid exceeded it on one, tied on three, and fell below it on twelve. These are
descriptive sequence comparisons from the user's cluster report, not independent
replications or final-test results. They motivate changing the training objective;
they do not establish that reinforcement learning cannot improve folding.

## Frozen starting states and scope

Reuse `policy_allocation_2h_01/tasks/patternboost_seed{0,1,2}/training/stage_7200.pt`.
All five descendants of a seed start from the exact same actor weights, Adam
moments, RNG states and diverse elite pool. The declared learning rate is then
applied to the retained Adam state. No model enlargement, pretrained LLM,
mixed precision, new training data, or test-set access is introduced.

The actor remains the custom 64-wide, two-layer, four-head Transformer with
rotary positions and six legal-masked lattice actions. It receives the complete
HP string and partial-fold geometry. It is not GPT-2. All algorithms use the
same domain and contact definition as the source allocation experiment.

The source bundle and dependencies are frozen. Historical model, environment,
training and evaluation files remain byte-identical in the copied bundle; the
new training, scheduling and reporting files have explicit hashes. Source
checkpoints are copied, never overwritten. Preparation rejects inconsistent
receipts, different Python/package versions and reused destination directories.

## Five fixed configurations

| Arm | Additional training | Learning rate | Primary change |
|---|---:|---:|---|
| `patternboost` | 1,800 seconds | 0.0003 | Continue supervised elite-pool training and refresh |
| `grpo` | 1,800 seconds | 0.0003 | Preserve the existing group-normalized update |
| `dr_grpo` | 1,800 seconds | 0.00003 | Remove within-group reward SD and per-trajectory action-count normalization |
| `reinforce` | 1,800 seconds | 0.00003 | Compare sampled outcomes with a frozen parent's deterministic rollout |
| `dpo` | 1,800 seconds | 0.00003 | Prefer verified higher-contact folds over lower-contact folds of the same input |

This first screen fixes settings in advance. It does **not** perform the small
learning-rate calibration proposed for a later matched tuning comparison.
The smaller new-method rate is a candidate setting, not an empirically selected
optimum. Objective scaling and learning rates differ, so outcomes compare these
configurations rather than establishing an intrinsic algorithm ranking.

Groups contain eight sampled folds of one uniformly chosen training sequence.
Policy-gradient arms use complete-fold contacts and a reward of -10 for trapped
folds. Interrupted samples are excluded. Dropout is zero; there is one update
per group, gradient norm clipping at one, and no KL or entropy bonus. GRPO's
ratio clip is 0.2; with one on-policy update it does not constrain the optimizer
step that follows the initial unit ratios. Absence of a KL term is intentional,
not a diagnosis of the earlier results.

Dr. GRPO and REINFORCE use a single global denominator equal to the maximum
number of learned actions across the training partition (62 for the preserved
dataset), rather than dividing each trajectory by its own realized length.
All completed folds of one sequence have the same action count, so language
model verbosity effects do not transfer directly; reward scaling and early
traps still make this comparison informative.

The REINFORCE baseline is a **frozen neural policy's greedy argmax fold**, with
first-index tie handling and the same legality masks. It is distinct from the
randomized greedy contact-maximizing search in the previous inference pilot.
Reference rollout computation counts against the training budget.

DPO uses the frozen parent as reference with beta 0.1. Within each on-policy
group, independently verify complete folds, pair the highest and lowest scores
without reusing a trajectory, and discard tied pairs. Compare summed masked
sequence log probabilities, excluding fixed anchor moves. Traps are excluded
from preference pairs; this different treatment is part of the configuration.
Pair sampling, scoring and reference likelihood computation count as training.
DPO is online preference fine-tuning, not a policy-gradient RL arm.

The model parameters are unchanged throughout every lineage. The frozen
reference is a copy used for training and diagnostics, not an inference ensemble.
Diagnostics periodically inspect existing training samples and record reward
spread, zero-advantage groups, structural diversity, gradient norm and sampled
current-minus-parent log-probability drift. The latter is not an exact KL estimate.

## Evaluation and decision

Evaluate the three unchanged parents plus all 15 final descendants on all 16
validation strings. Each endpoint receives 60 seconds per string, including
model loading; complete folds are generated by the model alone. CP-SAT does not
repair outputs. No checkpoint is selected by validation performance. Neither
the eight public benchmark strings nor the synthetic final test set is evaluated.

Independently verify final-best and improvement witnesses. Keep incomplete runs
and missing folds visible, never substitute zero, and include only complete
training and evaluation endpoints in means. Report per-sequence mean, sample SD,
best result, generated complete folds and validity, together with seed-paired
changes against the unchanged parent and continued PatternBoost. Preserve
negative changes. The three trained seeds remain three trained seeds when
evaluated on sixteen strings.

Use this screen to check finite updates, stable fold generation and promising
paired improvements. A later study can allocate equal modest tuning budgets and
longer inference to promising configurations. No 24-hour run, further training,
test evaluation or automatic extension follows. The old 17.5-minute classical
search results do not constitute a matched end-to-end comparison for these new
2.5-hour training lineages. Recalculate amortization and baseline budgets before
making any new model-versus-solver cost claim.

## Scheduling and cost

One Slurm array `0-17%18`, without dependencies:

| Array indices | Arm |
|---|---|
| 0–2 | Continued PatternBoost, seeds 0–2 |
| 3–5 | Current GRPO, seeds 0–2 |
| 6–8 | Dr. GRPO, seeds 0–2 |
| 9–11 | Rollout-baseline REINFORCE, seeds 0–2 |
| 12–14 | DPO, seeds 0–2 |
| 15–17 | Unchanged parents, seeds 0–2; evaluation only |

Each requests four EHC CPU cores, 16 GiB and a 75-minute scheduler limit. With
concurrent starts, a training task needs about 30 + 16 = 46 minutes plus startup
and serialization; a parent task needs about 16 minutes plus overhead. Availability
determines actual concurrency. END/FAIL email requests use the account destination
or `HP_MAIL_USER`. Per-phase timeouts and Slurm bound execution; failed phases
retain their logs and completed sequence results.

New planned training: `15 × 0.5 h × 4 = 30` allocated CPU-hours. Evaluation:
`18 × 16 × 1/60 h × 4 = 19.2`, total **49.2 CPU-hours** before overhead. The
scheduler reservation ceiling is `18 × 1.25 h × 4 = 90` CPU-hours. These are
allocated core-wall products, not actual CPU utilization. Consult `sacct` for
utilization and peak RAM.

Historical shared parent training physically cost 24 allocated CPU-hours and is
not repeated. Scientific accounting charges each final method lineage the full
two-hour parent plus its additional half hour. Parent-only evaluation is a
diagnostic control with a two-hour lineage, not an equal-total-training contender.

Commands and collection steps are in [README.md](README.md). The output directory
is `.cluster-runs/policy_finetune_30m_01`; raw checkpoints, event logs, launch
records, independent evaluations, report tables and paired CSV remain auditable.

## Primary references

- [PatternBoost](https://arxiv.org/abs/2411.00566): iterative training on strong examples and new candidate generation.
- [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300): group-relative policy optimization; our exact existing settings are preserved as a reference arm.
- [Dr. GRPO](https://arxiv.org/abs/2503.20783): change group and trajectory normalization; distinct from Decoupled Reward Policy Optimization (DRPO).
- [Attention, Learn to Solve Routing Problems!](https://arxiv.org/abs/1803.08475): REINFORCE with a greedy rollout baseline; the present frozen-parent HP adaptation does not implement that paper's baseline-update schedule.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290): sequence preference objective; pairs here are generated online and ranked by verified HP contacts.

Evidence from these papers motivates hypotheses; it does not establish improved
HP-folding performance for the new configurations.
