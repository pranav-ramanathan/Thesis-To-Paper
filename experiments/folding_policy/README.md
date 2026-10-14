# PatternBoost → RL: standalone variable-length HP folding policy

This package contains the validation pilots and frozen final comparison. One small
Transformer is reused across pretraining and RL; at inference it generates folds
without solver repair. It is separate from the historical per-sequence DQN
campaign and does not modify its code or results.

The initial pilot in [PLAN.md](PLAN.md) ran five learned conditions with two seeds: RL from scratch,
PatternBoost only, CP-SAT cloning, PatternBoost → RL and CP-SAT data → RL. It
also runs greedy and CP-SAT validation controls. It does **not** evaluate the
eight public benchmark strings or the new synthetic final test set.

## Final comparison: five seeds and all methods

The completed validation follow-up motivates the frozen [final protocol](FINAL_PLAN.md).
Train fresh models and compare **PatternBoost-only; PatternBoost → GRPO,
Dr. GRPO, REINFORCE and DPO; GRPO, Dr. GRPO and REINFORCE from scratch;
CP-SAT; and randomized greedy**. Each learned method receives a full 2.5-hour
training lineage. All methods receive ten minutes per sequence and seed.

The **80 previously unevaluated synthetic sequences** are the primary test set.
The eight public benchmark sequences appear in a separate table because their
historical results have already been examined. Training uses only the 160 training
sequences. Fixed endpoints prevent validation or test checkpoint selection.

```bash
cd /data/home/ah23335/Thesis-To-Paper &&
git pull --ff-only &&
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/final_campaign.sh \
  "$PWD/.cluster-runs/policy_final_01"
```

This submits **55 scheduler tasks in eleven arrays**: five shared parents and
fifty method/seed endpoints. Each endpoint processes 88 sequences sequentially.
Up to **50 endpoints run concurrently**, each with **four EHC cores and 16 GiB**.
Each fine-tuning branch waits for its corresponding successful parent seed;
scratch RL and classical search start independently.

Allow about **17 hours 10 minutes plus startup, saving and queue time** at full
availability. Planned new compute is **3,173.33 allocated CPU-hours**; the scheduler
reservation ceiling is 3,380 CPU-hours. The absolute campaign deadline is ten
days after preparation. There is no automatic extension or resubmission.
END/FAIL/invalid-dependency email is requested per array; set `HP_MAIL_USER` for an
explicit destination. Use a fresh directory. For a preview, append `--dry-run`
using a different destination; that directory cannot later be submitted again.

```bash
squeue -u "$USER"
sacct -X -j "$(cat .cluster-runs/policy_final_01/job_ids.txt)" --format=JobID,State,Elapsed,ExitCode
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/final_report.sh "$PWD/.cluster-runs/policy_final_01"
```

After completion, retain resource accounting:

```bash
sacct -j "$(cat .cluster-runs/policy_final_01/job_ids.txt)" --units=G --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem > .cluster-runs/policy_final_01/accounting.txt
```

Reporting independently checks retained witnesses, so the full report can take
several minutes and retains millions of attempted-fold records. Reports distinguish
physical training from per-method lineage charges; they do not claim equal-total-
cost superiority over the ten-minute classical runs. The original 24-hour DQN
experiment remains a separate study.

## Previous pilot: learning-rate controls and longer inference

The [fine-tuning follow-up](FINETUNE_FOLLOWUP_PLAN.md) reuses all eighteen saved
endpoints from the completed fine-tuning pilot. Only six tasks train: three
PatternBoost controls and three GRPO controls, each starting from its original
two-hour parent for another thirty minutes at learning rate `3e-5`.

All 24 endpoints receive ten minutes on each of the sixteen validation strings.
The report compares best contacts at one, five and ten minutes, and after the
first 128, 512 and 1,024 attempted folds, including traps. This checks both the
learning-rate difference and the observed sampling-speed difference.

```bash
cd /data/home/ah23335/Thesis-To-Paper &&
git pull --ff-only &&
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/finetune_followup_pilot.sh \
  "$PWD/.cluster-runs/policy_finetune_30m_01" \
  "$PWD/.cluster-runs/policy_finetune_followup_01"
```

One array allows **up to 24 concurrent tasks**, each with **4 EHC cores, 16 GiB
and a four-hour scheduler limit**. There are no dependencies. With concurrent
starts, allow about **3 hours 10 minutes plus startup and saving**, and any queue
time. Reused-endpoint tasks take about 2 hours 40 minutes plus overhead. New
planned compute is **268 allocated CPU-hours**: 12 training and 256 inference.
The scheduler reservation ceiling is 384 CPU-hours. END/FAIL email is requested;
set `HP_MAIL_USER` to specify a destination. Use a fresh output directory, or
append `--dry-run` to prepare without submitting.

```bash
squeue -u "$USER"
sacct -X -j "$(cat .cluster-runs/policy_finetune_followup_01/job_ids.txt)" --format=JobID,State,Elapsed,ExitCode
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/finetune_followup_report.sh "$PWD/.cluster-runs/policy_finetune_followup_01"
```

After completion, retain resource accounting:

```bash
sacct -j "$(cat .cluster-runs/policy_finetune_followup_01/job_ids.txt)" --units=G --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem > .cluster-runs/policy_finetune_followup_01/accounting.txt
```

Only validation sequences are evaluated. Saved endpoints are frozen and copied;
new controls preserve parent weights, Adam moments, random states and elite pools.
No solver repair, final-test access, automatic extension or checkpoint selection
occurs. Existing scientific results are preserved.

The report independently checks each recorded complete fold among the first
1,024 attempts, so it can take several minutes. It prints progress after every
six evaluated tasks. CPU usage, affinity and inference verification/write overhead
are available in `throughput.csv` and `analysis.json`; report analysis time is
recorded separately. In-flight sequence snapshots are retained under each task's
`sequences` directory until the aggregate report can include the finished entry.

## Previous pilot: fine-tuning the saved PatternBoost policies

The [fine-tuning screen](FINETUNE_PLAN.md) starts from the same saved two-hour
PatternBoost parent for each of three seeds. It compares another 30 minutes of
PatternBoost, current GRPO-style training, Dr. GRPO, REINFORCE with a frozen
greedy-policy baseline, and DPO. Three additional tasks evaluate the unchanged
parents. All 18 endpoints receive 60 seconds on each of the 16 validation strings.

```bash
cd /data/home/ah23335/Thesis-To-Paper &&
git pull --ff-only &&
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/finetune_pilot.sh \
  "$PWD/.cluster-runs/policy_allocation_2h_01" \
  "$PWD/.cluster-runs/policy_finetune_30m_01"
```

This submits **one array with up to 18 concurrent tasks**, each requesting
**4 EHC cores, 16 GiB and 75 minutes**. Training tasks take about 46 minutes
plus process startup and saving; parent-only tasks take about 16 minutes plus
overhead. Queue time can extend campaign duration. END/FAIL email is requested;
set `HP_MAIL_USER` if an explicit recipient is needed. The frozen preparation
checks the source checkpoints, software and data. Use a fresh destination;
append `--dry-run` to prepare without submitting.

```bash
squeue -u "$USER"
sacct -X -j "$(cat .cluster-runs/policy_finetune_30m_01/job_ids.txt)" --format=JobID,State,Elapsed,ExitCode
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/finetune_report.sh "$PWD/.cluster-runs/policy_finetune_30m_01"
```

After completion, retain resource accounting:

```bash
sacct -j "$(cat .cluster-runs/policy_finetune_30m_01/job_ids.txt)" --units=G --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem > .cluster-runs/policy_finetune_30m_01/accounting.txt
```

The new planned allocation is **49.2 CPU-hours** before overhead: 30 for training
and 19.2 for evaluation. The existing pretraining is reused. This screens fixed
configurations with different learning rates; a matched tuning study would be
needed for algorithm-ranking claims. The policy architecture stays the same,
inference uses no solver repair, and neither final test set is evaluated.

## Previous pilot: saved policies versus CP-SAT and greedy

The [inference comparison](INFERENCE_PLAN.md) reuses all four final model
conditions and all three seeds from the completed allocation pilot. It runs all
16 validation strings with 10 minutes per model/sequence/seed, and 17.5 minutes
for fresh CP-SAT and greedy searches. The same search curves support equal
inference budgets and an N=16 comparison charging the models' two-hour training
cost. No retraining or final-test evaluation is performed.

```bash
cd /data/home/ah23335/Thesis-To-Paper &&
git pull --ff-only &&
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/inference_pilot.sh \
  "$PWD/.cluster-runs/policy_allocation_2h_01" \
  "$PWD/.cluster-runs/policy_inference_10m_01"
```

This copies and verifies the source checkpoints before submitting **12 model
tasks and six baseline tasks in two independent arrays**, up to 18 concurrently.
Each requests **4 EHC cores and 16 GiB**. With concurrent starts, allow up to
**4 hours 40 minutes plus startup and queue time**. The scheduler limits are
three hours for model tasks and five hours for baseline tasks. END/FAIL mail is
requested, using the account destination or an explicit `HP_MAIL_USER`.
The planned new search budget is 240 allocated CPU-hours before overhead;
the scheduler reservation ceiling is 264 CPU-hours. Early CP proofs save time.
Use a fresh destination; append `--dry-run` for preparation without submission.

```bash
squeue -u "$USER"
sacct -X -j "$(cat .cluster-runs/policy_inference_10m_01/job_ids.txt)" --format=JobID,State,Elapsed,ExitCode
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/inference_report.sh "$PWD/.cluster-runs/policy_inference_10m_01"
```

After completion, save accounting separately:

```bash
sacct -j "$(cat .cluster-runs/policy_inference_10m_01/job_ids.txt)" --units=G --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem > .cluster-runs/policy_inference_10m_01/accounting.txt
```

## Previous pilot: two-hour training allocation

The [allocation pilot](ALLOCATION_PLAN.md) compares PatternBoost for two hours,
PatternBoost 60 minutes → RL 60 minutes, PatternBoost 90 minutes → RL 30 minutes,
and RL from scratch for two hours, with three seeds. Shared pretraining states
include optimizer, random states and elite pools. Actual pre-RL and post-RL
endpoints are evaluated on all 16 validation strings, separately from the
earlier pilot's selected-checkpoint results.

```bash
cd /data/home/ah23335/Thesis-To-Paper
git pull --ff-only
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/allocation_pilot.sh "$PWD/.cluster-runs/policy_allocation_2h_01"
```

This submits four arrays of three tasks each, with dependencies connecting each
RL continuation to its matching PatternBoost seed. Each task requests **4 EHC
cores and 16 GiB**. Allow approximately **four hours and ten minutes plus queue
time and overhead** for the longest dependency path. END/FAIL email requests
use the account destination, or supply `HP_MAIL_USER` explicitly. No final test
or later campaign runs automatically. A fresh directory is required.

```bash
squeue -u "$USER"
sacct -X -j "$(cat .cluster-runs/policy_allocation_2h_01/job_ids.txt)" --format=JobID,State,Elapsed,ExitCode
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/allocation_report.sh "$PWD/.cluster-runs/policy_allocation_2h_01"
```

The dependent arrays initially show `Dependency`; independent tasks may wait
for resources or priority. The report can be generated before completion and
keeps incomplete tasks visible. Save detailed accounting after completion:

```bash
sacct -j "$(cat .cluster-runs/policy_allocation_2h_01/job_ids.txt)" --units=G --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS,ReqMem > .cluster-runs/policy_allocation_2h_01/accounting.txt
```

The original 20-minute pilot and its collection commands remain below.

## Run the Apocrita pilot

From an Apocrita login shell, using the already installed cluster environment:

```bash
cd /data/home/ah23335/Thesis-To-Paper
git pull --ff-only
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/pilot.sh "$PWD/.cluster-runs/policy_pilot_01"
```

This freezes source, sequence partitions, configuration and dependency metadata
before submitting a **10-element Slurm array**, with **up to 10 concurrent
tasks**, **4 EHC cores and 16 GiB each**, and a **45-minute scheduler limit**.
Each learned condition/seed gets 20 minutes for data generation, training and
checkpoint selection, then four one-minute validation searches. The first two
tasks also run greedy and CP-SAT controls, so they take approximately 32 minutes
plus startup/saving. Actual queue and runtime depend on cluster availability.
The reservation ceiling is 30 allocated CPU-hours. No later campaign starts
automatically.

Email notifications request `END,FAIL`. To specify the destination explicitly,
prefix the submission command with `HP_MAIL_USER="your-address@example.com"`.
Delivery depends on the account's mail configuration and cluster service.

For a preparation-only preview use a separate, new directory:

```bash
bash experiments/folding_policy/pilot.sh "$PWD/.cluster-runs/policy_preview_01" --dry-run
```

Existing output directories are rejected to preserve earlier runs. The dry run
does not execute training. Dependencies are the existing cluster Torch/NumPy/
OR-Tools environment; no new package installation is required on Apocrita.

## Monitor and collect

```bash
squeue -u "$USER"
cat .cluster-runs/policy_pilot_01/job_id.txt
tail -n 30 .cluster-runs/policy_pilot_01/logs/*.out
```

Use the printed array job ID for accounting:

```bash
sacct -X -j JOB_ID --format=JobID,State,Elapsed,ExitCode,AllocCPUS
sacct -j JOB_ID --format=JobID,State,Elapsed,TotalCPU,MaxRSS,ReqMem
```

Generate the validation report (also works while jobs are unfinished):

```bash
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash <<'HP_POLICY_REPORT'
source experiments/cluster/environment.sh
"$HP_VENV_PATH/bin/python" experiments/folding_policy/report.py "$PWD/.cluster-runs/policy_pilot_01"
cat .cluster-runs/policy_pilot_01/report.md
HP_POLICY_REPORT
```

Inspect each task's `launch.json`, `training/result.json`, `training/events.jsonl`,
and evaluation JSON. Best fold coordinates and improvement-time curves are
saved for independent checking. A failure or incomplete validation is visible;
it is not converted to a zero score or silently included as a successful run.
Checkpoint/model/data hashes connect each inference result to its training run.

The report exports `summary.csv` and `curves.csv`. To generate a standard
PGFPlots figure source from independently verified per-run curves:

```bash
"$HP_VENV_PATH/bin/python" experiments/folding_policy/plot_curves.py "$PWD/.cluster-runs/policy_pilot_01"
```

Run that command in a shell where `environment.sh` has been sourced. Copy the
generated `plots/inference_curves.tex` to a machine with TeX/PGFPlots and compile
with `pdflatex inference_curves.tex`. Each line is one seed's cumulative
inference time; it does not pool several runs against one CP-SAT run.

## Local correctness tests

Run in an existing Python environment with Torch and OR-Tools installed:

```bash
python -m unittest discover -s experiments/folding_policy -p 'test_*.py' -v
bash -n experiments/folding_policy/pilot.sh experiments/folding_policy/pilot.sbatch
```

Pure partition/accounting checks also run without Torch; numerical tests require
it. Tiny synthetic tests verify geometry, legal masking, variable lengths,
symmetry transformations, gradients, stage continuity and frozen bundles.
See [VALIDATION.md](VALIDATION.md) for the actual checks performed before release.
These checks are engineering evidence, not measured learning superiority.

`protocol.json` is the pilot's fixed configuration. `dataset.py` creates 160
training, 16 validation, 80 novel test and 8 public benchmark sequence records.
Only validation IDs 000, 005, 010 and 015 enter post-training pilot comparisons;
they span lengths 24/40/60/72 and H fractions 0.25/0.40/0.60/0.75. Training
checkpoint selection uses the separate validation partition. Public benchmark
contact counts are literature context and never training rewards or selection
targets.
