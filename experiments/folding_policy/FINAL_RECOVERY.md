# CP-SAT seed 4 recovery

The original final campaign stopped CP-SAT seed 4 after 75 accepted synthetic
test searches. The next search, `test_075`, returned a valid FEASIBLE fold after
599.9213819764555 seconds, 0.0786180235445 seconds short of its 600-second budget.
The worker's strict elapsed-coverage check rejected it and prevented the remaining
test and public-benchmark searches from starting. This was an orchestration
failure triggered by elapsed time, not a failed fold-geometry check.

This recovery is declared after seeing the incomplete report. It keeps the
original budget acceptance rule and original solver implementation. It does not
change a timestamp or introduce a tolerance after observing test outcomes.

## Fixed recovery scope

- Reuse the 75 previously accepted synthetic searches for seed 4.
- Run exactly 13 fresh searches: `test_075`–`test_079`, then the eight public
  benchmarks, using seed 4, four EHC cores and 600 seconds per search.
- Replace the rejected `test_075` attempt regardless of whether its replacement
  scores higher or lower. Never pool the two attempts' best folds.
- Retain the original short attempt, failure records, campaign manifest, frozen
  code and all other results. Write new files in a separate recovery directory.
- Validate the new results with the existing geometry, provenance, proof and
  elapsed-budget checks. No automatic retries or deadline extensions occur.

The recovery manifest freezes the original inputs and exact replacement cells.
The original frozen worker executes each fresh search. The reporting overlay
combines verified original and recovery cells without overwriting source files;
the original report continues to describe the original incomplete execution.

## Cluster execution

From the repository root after pulling the latest code:

```bash
HP_PYTHON_MODULE=python/3.11.7-gcc-12.2.0 bash experiments/folding_policy/final_recovery.sh \
  "$PWD/.cluster-runs/policy_final_01" \
  "$PWD/.cluster-runs/policy_final_cp_seed4_recovery_01"
```

The 13 array elements can run concurrently. Each requests four EHC cores, 16 GiB
and a 20-minute scheduler limit. Planned search is **8.67 allocated CPU-hours**;
the scheduler reservation ceiling is **17.33 CPU-hours**. Search takes at most ten
minutes per cell, plus startup/verification and queue time. Early optimality
proofs can finish sooner. The original campaign's absolute deadline still applies.

Schedule report verification after successful completion of every recovery cell:

```bash
sbatch --dependency="afterok:$(cat .cluster-runs/policy_final_cp_seed4_recovery_01/job_ids.txt)" \
  --kill-on-invalid-dep=yes \
  --output="$PWD/.cluster-runs/policy_final_cp_seed4_recovery_01/report-%j.out" \
  --error="$PWD/.cluster-runs/policy_final_cp_seed4_recovery_01/report-%j.err" \
  experiments/folding_policy/final_report.sbatch \
  "$PWD/.cluster-runs/policy_final_01" \
  --recovery "$PWD/.cluster-runs/policy_final_cp_seed4_recovery_01"
```

Read the merged report at
`.cluster-runs/policy_final_cp_seed4_recovery_01/report/report.md`.
Report generation requests one EHC core, 8 GiB and a two-hour maximum. The first
complete original report took approximately 24 minutes to verify 3.58 million
witness records; recovery reporting checks those records again. Record recovery
and reporting Slurm accounting separately from the original array jobs.

In the paper, disclose the failure and recovery, the discarded original attempt,
the unchanged search settings and the additional physical search cost. The
reused 75 searches are not new cluster consumption and must not be charged twice.
