# PatternBoost → RL: standalone variable-length HP folding policy

This package implements the validation pilot in [PLAN.md](PLAN.md). One small
Transformer is reused across pretraining and RL; at inference it generates folds
without solver repair. It is separate from the historical per-sequence DQN
campaign and does not modify its code or results.

The pilot runs all five learned conditions with two seeds: RL from scratch,
PatternBoost only, CP-SAT cloning, PatternBoost → RL and CP-SAT data → RL. It
also runs greedy and CP-SAT validation controls. It does **not** evaluate the
eight public benchmark strings or the new synthetic final test set.

## Next pilot: saved policies versus CP-SAT and greedy

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
