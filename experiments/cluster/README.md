# Apocrita CPU experiment bundle

This bundle runs from a fresh clone. It does not read sibling projects, old
outputs, laptop virtual environments or the dissertation. All numerical work
runs on CPU. Nothing is submitted by cloning or installing dependencies.

## Deadline and compute budget

Apocrita currently recommends requesting **one hour** on `computeshort`, or
**240 hours / ten days** on `compute`. These are scheduler ceilings, not a
requirement to use every hour. [Official runtime documentation](https://docs.hpc.qmul.ac.uk/using/submittingjobs/runtime/).

The supplied defaults are:

| Setting | Pilot | Campaign |
|---|---|---|
| Slurm request | 1 hour | 10 days per allocation |
| Application compute cutoff | 50 minutes; individual cells also bounded | **24 hours per primary run**, configurable |
| Absolute campaign deadline | 10 days from preparation (queue allowance) | **10 days from preparation**, including queue time |
| CPUs / memory per array element | 8 / 32 GB | 8 / 32 GB |
| Concurrent array elements | 1 | 24, configurable |

`--run-hours 12`, `24`, `48`, etc. changes the scientific compute budget without
changing the ten-day Slurm ceiling. Select it after the CPU pilot, then freeze
it for all primary methods. The maximum accepted value is 239.75 hours to leave
room for saving. No new work starts in the final five minutes of the absolute
deadline. A parent guardian signals its worker at the application cutoff, allows
up to three minutes for saving, then kills its own worker process group if needed;
the global ten-day deadline always takes precedence. Native calls may stop with
some latency: record actual elapsed time and ignore post-budget witnesses for
timepoint comparisons. Jobs also receive Slurm's pre-timeout signal. There is no
automatic requeue, extension or follow-up campaign.

There are 120 primary jobs: eight exact strings × five seeds × three methods.
At 24 hours and 24 concurrent jobs this is roughly five waves / five days if all
jobs use their full budgets, before queue delays. Concurrency 24 means up to 192
allocated CPUs across jobs, **eight per job**. EHC's documented 96-core limit is
per job. The global ten-day/concurrency ceiling is approximately 46,080 allocated
CPU-hours; the primary 24-hour jobs themselves total at most 23,040 CPU-hours
before saving overhead. This is not a prediction of actual usage or queue time.
Changing concurrency changes completion feasibility. Late or interrupted runs
are explicitly partial, not silently treated as full-budget outcomes.

## Next step: 24-hour seed-0 feasibility pass

After the resource pilots, run **nine fresh jobs**, rather than the full
120-job campaign: 3d4, 3d6 and the explicitly proposed 3d8 configuration, each
with seed 0 and CP-SAT, RL and fixed RL+CP-SAT. This is a learning/search and
sustained-memory check. It preserves the original configurations, epsilon
schedule, 200,000-episode limit and fixed repair policy. It is not the learned
DecisionBoost controller study.

From the existing Apocrita clone, after updating the repository:

```bash
git pull --ff-only
HP_NODE_CONSTRAINT=ehc HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 \
  bash experiments/cluster/seed0-test.sh "$PWD/.cluster-runs/seed0_24h_01"
```

The `ehc` feature was verified in the supplied `sinfo` output. The command
submits an array with at most nine simultaneous jobs, **eight CPUs and 32 GiB
per job**, using the existing environment. No dependency reinstall is needed.
Each run computes for up to 24 hours, then has up to three minutes to save.
The Slurm ceiling remains ten days; the frozen absolute deadline includes queue
time and is ten days from preparation. All nine jobs together use up to about
**1,728 allocated CPU-hours plus saving overhead** if all reach the compute
cutoff. Nine available allocations allow one approximately 24-hour wave;
scheduling can delay or stagger them. CP-SAT can finish early with a cube proof.
Slurm sends END/FAIL notifications to the registered address for the array;
per-task notification timing follows site Slurm configuration.

Use a fresh output directory. The wrapper freezes only these nine tasks and
enables completion/failure mail. It does not submit any follow-up experiment.
To inspect the submission command without submitting, append `--dry-run` and
use a separate fresh directory, then use another directory for the real run.

Monitor and collect:

```bash
squeue -u "$USER"
bash -c 'source experiments/cluster/environment.sh; "$HP_VENV_PATH/bin/python" experiments/cluster/analyse.py "$PWD/.cluster-runs/seed0_24h_01"'
cat .cluster-runs/seed0_24h_01/report.md
sacct -j "$(cat .cluster-runs/seed0_24h_01/job_id.txt)" \
  --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS,NodeList
```

Review validated best-fold trajectories, hourly greedy evaluations, completed
episodes/epsilon, checkpoints and peak memory. One seed cannot establish a
multi-seed method ranking. Use these results to choose the later five-seed
budget; no full campaign or deadline extension follows automatically.

The reported completion of array `30608441` is recorded in
[APOCRITA_SEED0_24H.md](APOCRITA_SEED0_24H.md). All nine tasks completed at
24 hours; CP-SAT scored highest on the three representatives and the fixed
hybrid improved on RL. The report distinguishes these one-seed best folds
from policy learning and the separate learned DecisionBoost study. The next
decision requires trajectories, episode counts and greedy evaluations.
The full submission command below still creates all 120 tasks; continuing
with the remaining 111 requires explicit exclusions and combined analysis
across the frozen bundles to avoid duplicating these seed-0 observations.

## Methods and scientific scope

- `cp_sat`: one continuous full-model search in the declared cube, with eight
  workers, no heuristic seed in a fresh run. It may finish early if it proves
  optimality **within that cube**. Resume starts a new CP search with the saved
  incumbent; it does not preserve learned solver clauses.
- `rl`: the project reference DQN, fresh weights/replay per sequence/seed, at most
  one update per episode. Evaluation restores training mode. The 200,000-episode
  schedule is fixed. If training finishes before the elapsed budget, the frozen
  policy continues sampling with epsilon 0.05; those samples are labelled.
- `rl_cp_sat`: the same training reference plus a fixed CP repair every 1,000
  episodes, alternating suffix/internal ten-residue windows, at most 30 seconds
  each. Every CP cost is charged to the same elapsed budget. This is a fixed
  hybrid, **not** the learned DecisionBoost controller, and does not inject CP
  folds into the RL replay buffer.

Primary methods share origin/+x anchoring and the cube
`abs(x), abs(y), abs(z) <= floor(n/2)`, matching the existing RL environment.
The existing RL mask additionally uses its reference symmetry and lookahead
rules. Report those explicitly; equal hardware is not identical search behaviour.
The independent checker accepts only complete integer, self-avoiding folds with
unit backbone bonds. Contacts exclude consecutive residues. Failed/incomplete
folds are not solutions. Missing outcomes remain null. Bounds from fixed-window
repairs are not full-fold optimality proofs.

`protocol.json` preserves the saved project configuration rows. The 3d8
architecture remains explicitly **proposed**, because the saved protocol lacked
that row. This is a project-baseline comparison, not a claim of exact reproduction
of every published hyperparameter. Exact primary sequence lengths are
**20, 24, 25, 36, 46, 48, 50, 58**; the length-eight training sequences belonged
to an earlier generated corpus.

## Clone and install

Use a shared persistent work/scratch filesystem visible to compute nodes, not
node-local `$TMPDIR`. Check its quota and retention period; full replay
checkpoints and per-episode traces can consume substantial space across 120 jobs.

```bash
git clone https://github.com/pranav-ramanathan/Thesis-To-Paper.git
cd Thesis-To-Paper
module avail python miniforge
module load python
# Use the same available Python 3.12 module for setup and submissions.
export HP_PYTHON_MODULE=python
bash experiments/cluster/setup.sh
```

Setup installs CPU PyTorch 2.9.0, NumPy 2.3.4 and OR-Tools 9.15.6755 into
`.venv-cluster`; it saves the full dependency listing and checks dependencies.
Workers record actual library versions and reject numerical version mismatches
before scientific execution. Local engineering smoke checks may use existing
test runtimes and are explicitly excluded from scientific results.
Module names/versions must come from the cluster; the documented example
`python/3.12.1-gcc-12.2.0` is not assumed to be installed today.
[Python module documentation](https://docs.hpc.qmul.ac.uk/using/python_distributions/).

## Choose identical CPU nodes

```bash
sinfo -N -p compute -o '%N %f %c'
scontrol show node ACTUAL_EHC_HOSTNAME
```

EHC nodes have AMD EPYC 9965 CPUs. Use a **verified** feature from `sinfo` that
selects only the intended homogeneous CPU architecture:

```bash
export HP_NODE_CONSTRAINT='ACTUAL_VERIFIED_FEATURE'
```

Alternatively, set `HP_NODELIST` to **one actual EHC hostname**, and all jobs
will request that node. Do not supply several hostnames: Slurm's nodelist requests
those nodes together, whereas these scripts request one node per job. A single
node can run multiple independent jobs as resources permit. The scripts refuse
unrestricted submission; they do not invent an `ehc` feature name.
[EHC specifications](https://docs.hpc.qmul.ac.uk/nodes/ehc/),
[Slurm constraints](https://docs.hpc.qmul.ac.uk/using/submittingjobs/constraints/).

Check that your selected nodes are also available in `computeshort` for the pilot.
If not, set `export HP_PILOT_PARTITION=compute` before submitting: the pilot still
requests **one hour**, on the same CPU architecture.

## Pilot, then campaign

```bash
# Substitute a persistent shared path of your choosing.
export HP_RUN_ROOT="$PWD/.cluster-runs"
mkdir -p "$HP_RUN_ROOT"

bash experiments/cluster/cp-resource-test.sh "$HP_RUN_ROOT/pilot_cp_01"
bash experiments/cluster/rl-resource-test.sh "$HP_RUN_ROOT/pilot_rl_01"
# After each separate job finishes:
bash experiments/cluster/resource-report.sh "$HP_RUN_ROOT/pilot_cp_01"
bash experiments/cluster/resource-report.sh "$HP_RUN_ROOT/pilot_rl_01"
# Read tasks/000_pilot_cp_sat_resources_seed0/resource_report.md in the CP directory,
# and tasks/000_pilot_rl_resources_seed0/resource_report.md in the RL directory.
# Detailed timings: resource_report.json and cells/*/measurement.json.
# Scheduler usage: accounting.txt (including job-step MaxRSS and TotalCPU).

# Preview a concrete frozen campaign without submitting it:
bash experiments/cluster/submit.sh campaign "$HP_RUN_ROOT/preview_01" --run-hours 24 --concurrency 24 --dry-run

# Actual submission requires a NEW directory. Its ten-day clock starts now.
bash experiments/cluster/submit.sh campaign "$HP_RUN_ROOT/campaign_01" --run-hours 24 --concurrency 24 --threads 8
```

These are **two independent jobs**, one CP-SAT and one RL, with separate frozen
manifests, logs, measurements and reports. **Each requests eight CPUs, 32 GiB
and one hour**. Each screens 1/2/4/8 threads sequentially inside its allocation:
at most eight allocated CPU-hours per job, sixteen for both. Each stops
computation after 50 minutes, leaving saving/termination time. No campaign is
submitted afterwards. Preview either submission with
`cp-resource-test.sh NEW_DIRECTORY --dry-run` or
`rl-resource-test.sh NEW_DIRECTORY --dry-run`; use fresh directories for actual
submissions. You can also use `resource-test.sh cp_sat|rl NEW_DIRECTORY`.

Each measurement uses a separate child process for meaningful peak resident
memory. The **RL job**, on **3d4/3d6/3d8**, populates configured replay with distinct FP32
arrays (repeated valid transition values), warms up twice, and targets five
complete episode operations at epsilon 0.25. Their updates provide the five
update timings too, for **seven updates total** including warmup. Separate
exploratory/greedy rollout probes measure action/environment costs without
adding optimizer updates or writing replay. The pilot measures exploratory
and greedy action selection, environment stepping, insertion, updates, independent
fold checking, trace writing, evaluation and a full optimizer/replay/RNG checkpoint.
Synthetic buffer population time is reported separately; it does not estimate
how long genuine experience takes to accumulate. Checkpoint files are removed
after their sizes and write times are recorded, keeping scratch use bounded.

The **CP-SAT job** runs five fresh full searches (15 seconds each) and
fixed-window repairs (five seconds each) on each representative at each worker
count. Both jobs check the remaining five exact strings: RL uses their actual
architecture, batch and replay sizes; CP checks full search and repair once.
RL representative caps are 100–400 seconds depending on configuration/thread
count, sized from the first EHC pilot. RL compatibility checks allow 120 seconds;
CP representative caps remain 125 seconds and other CP caps 30 seconds.
Caps include Python/library startup, which used part of the original short
limits. The sum of RL cell caps, kill grace and reporting reserves fits inside
the 50-minute application limit. Pilot jobs email the registered Apocrita
address on completion or failure.
Slow/native calls are terminated and saved as partial, with logs and sampled
peak RSS; missing repetitions do not become successful timings.

The RL report proposes the **fewest cores within 10% of the fastest
aggregate representative RL throughput**, RAM rounded up with 50% headroom,
checkpoint sizes, CPU use, descriptive timing ranges, extrapolated episodes/hour
and 200,000-episode durations for the three representatives. It also shows the
CPU-hour and elapsed-wave costs of several common campaign budgets. The separate
CP report records worker counts, witness return rates, search/repair duration,
CPU use and memory. Eight CP workers remains a tested provisional setting;
these short calls cannot choose the best long-run worker count. Other RL rows
are compatibility checks, not reliable runtime estimates. Short CP tests do not
establish solver scaling or a long-run memory upper bound. The optional
pretrained encoder/controller study needs a separate memory check.

Inspect both reports, timeouts, all eight configuration checks and Slurm accounting
before choosing final resource requests and a common allocation for the paper
comparison. Each report flags incomplete coverage; it does not
automatically freeze a campaign budget. These disposable engineering states
cannot establish learning convergence or a fair method ranking. The primary
bundle records its exact executed sources, protocol, seeds and deadline. Every
launch checks the frozen source hashes.

The first EHC resource reports and deadline diagnostics are transcribed in
[APOCRITA_PILOT_01.md](APOCRITA_PILOT_01.md), with provisional interpretations.
The completed corrected RL pilot is transcribed in
[APOCRITA_PILOT_02.md](APOCRITA_PILOT_02.md): eight threads were fastest among
the tested counts, all eight rows completed their eight-thread checks, and
Slurm reported 10.92G peak memory. One four-thread 3d6 cell remained partial
after recording all five episode timings. This is enough to end broad resource
profiling; the next recommendation is a separate 24-hour seed-0 learning and
memory feasibility pass before freezing the five-seed campaign. No such pass
has been submitted here. The full-run templates still request 32 GiB.
To repeat the corrected **RL pilot only**, keep the completed CP pilot and use
a fresh output directory:

```bash
bash experiments/cluster/rl-resource-test.sh "$HP_RUN_ROOT/pilot_rl_02"
```

## Optional transformer + learned controller + CP-SAT replication

This is a **separate study**, with the previously tested method and a new frozen
corpus excluding all prior generated/paper reversal classes. It trains five
independent pipelines and evaluates 72 fresh sequences per seed. It uses
single-worker solver calls and complete reachable coordinate domains, so it must
not be pooled into the primary cube/equal-elapsed-budget table.

```bash
bash experiments/cluster/setup.sh --decisionboost
bash experiments/cluster/prepare-model.sh "$PWD/.cluster-models/ModernBERT-large"
bash experiments/cluster/submit.sh campaign "$HP_RUN_ROOT/campaign_db_01" \
  --run-hours 24 --concurrency 24 --threads 8 \
  --include-decisionboost --model-dir "$PWD/.cluster-models/ModernBERT-large"
```

This submits **125 jobs total**, including the same 120 primary jobs. Choose
either the primary-only campaign or this combined submission; do not launch
both accidentally. Model preparation downloads the open, pinned ModernBERT-large
revision once (~1.5 GB weights); compute jobs are offline. Its files are hashed.
No API key or paid service is used. Training/teaching, encoder preparation,
proposal inference and solver costs are logged separately. Reserved validation
sequences are encoded but do not choose hyperparameters/checkpoints. The full
pretrained encoder's original training cost is unknown, not zero.

The DecisionBoost workload finishes when its fixed protocol finishes; it does
not train for 24 hours just to fill the allocation. Its per-call-budget endpoint
and actual cost must be reported separately from primary elapsed-budget scores.
Partial training is preserved; no automatic restart/resume is provided for it.

## Monitor, analyse and explicitly resume

If a resource report shows `partial_or_failed`, read its saved exit codes,
timeout flags and individual timings before submitting another pilot:

```bash
bash experiments/cluster/resource-diagnose.sh "$HP_RUN_ROOT/pilot_rl_01"
```

This command only reads existing records and logs, and works with the original
pilot outputs after a repository update. It runs no model or solver. Updated
reports show completed-operation counts from unfinished cells while excluding
those cells from resource selection. They distinguish per-cell deadlines from
exceptions. Future pilots also record their active phase and warmup durations.
`resource-report.sh` uses the current reader, verifies the frozen original
bundle, and records the analyser hash alongside regenerated derived reports.
The original measurements and executed-source snapshot are preserved.

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,TotalCPU,MaxRSS,NodeList
.venv-cluster/bin/python experiments/cluster/analyse.py "$HP_RUN_ROOT/campaign_01"
# To stop the campaign earlier by hand (preserves existing checkpoints/results):
scancel "$(cat "$HP_RUN_ROOT/campaign_01/job_id.txt")"
```

The analyser writes `analysis.json` and `report.md`, verifies complete witnesses
independently, and keeps primary versus DecisionBoost results separate. Primary
timepoints include 1/2/12/24 hours and longer checkpoints if the budget allows.
Short-budget CP missing solutions do not become zero-contact outcomes. Per-seed
SD is descriptive; a final manuscript needs uncertainty and paired comparisons
from completed runs rather than selecting the best seed. Report queue time and
actual allocated CPU-hours from `sacct` separately from scientific compute time.

RL checkpoints contain both networks, optimizer, complete replay/priorities,
update count and Python/NumPy/Torch RNG states. To resume an interrupted primary
task **within its original deadline and remaining compute budget**, explicitly
submit the original frozen template (replace the array index and hardware flag):

```bash
sbatch --constraint="$HP_NODE_CONSTRAINT" --array=TASK_INDEX \
  --chdir="$HP_RUN_ROOT/campaign_01" \
  --output="$HP_RUN_ROOT/campaign_01/logs/resume_%A_%a.out" \
  --error="$HP_RUN_ROOT/campaign_01/logs/resume_%A_%a.err" \
  "$HP_RUN_ROOT/campaign_01/bundle/campaign.sbatch" "$HP_RUN_ROOT/campaign_01" --resume
```

Use `--nodelist="$HP_NODELIST"` instead of `--constraint` if you selected one
hostname. If you changed the CPU count, also pass `--cpus-per-task=YOUR_COUNT`.
Never run a second active instance of the same task; a filesystem lock rejects
that. Resume rolls training back to the last durable checkpoint, retains all
previously validated best folds and charges previously consumed time, including
work since that checkpoint. Lost work is not free. CP resume starts a new search.
No deadline resets. A finished budget is not extended by resubmission.
Queued jobs that have not started by the absolute deadline will skip computation
when scheduled; cancel any remaining pending array elements with `scancel`.

## Local engineering verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-cluster/bin/python -m unittest discover -s experiments/cluster/tests -v
```

The tests exercise real CP witnesses, fixed-window repairs, reference RL updates,
mode restoration, full checkpoint restoration, absolute deadline skipping,
guardian termination, source freezing, missing-result accounting and fake-Slurm
submission. Smoke runs use small engineering settings and are excluded from
scientific summaries. Apocrita scheduling and EPYC throughput still require the
actual cluster pilot.
