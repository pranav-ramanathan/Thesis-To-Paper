# Engineering verification — 6 October 2026

These checks validate the bundle. They are not scientific experiment outcomes,
cluster measurements, learning-convergence evidence or a completed seed campaign.

- **14 behavioural tests passed**, both in the workspace and an isolated copy
  containing only the cluster bundle. Tests cover frozen sources/deadline,
  complete five-seed task coverage, new corpus exclusions, missing-result
  accounting, fake-Slurm submission with paths containing spaces, forced process
  termination, real CP full-search/repair witnesses, mode restoration on success
  and failure, exact next-update restoration from a full RL checkpoint, and
  interrupted-log recovery without discarding committed history.
- The extracted DQN matched the preserved corrected reference on CPU for
  observations, action masks, transitions, initial weights, losses, parameter
  updates and replay priorities over three real updates. This comparison used
  **PyTorch 2.9.0 / NumPy 2.3.4**, the target RL versions.
- The launch guardian ran all three primary arms with small engineering
  settings and an eight-second cutoff. All produced complete independently
  checked witnesses; RL/hybrid saved replay/optimiser/RNG checkpoints. The
  resulting smoke trajectories were excluded from scientific means.
- An isolated transformer/controller/CP smoke used the existing pinned local
  encoder, new corpus, two gradient steps and short solver caps. It completed
  two cases; **74 saved CP witnesses** passed the independent geometry/contact
  counter. No model download or API call occurred during verification.
- The combined tests and integrated smoke runs used the available engineering
  runtime **Python 3.12 / PyTorch 2.11.0 / NumPy 2.4.4 / OR-Tools 9.15.6755**.
  They do not claim end-to-end execution under the pinned cluster environment.
  Scientific workers reject version mismatches; the installed CPU cluster
  environment and EPYC throughput still require the actual cluster pilot.
- Every cluster Python file parsed; Bash syntax checks passed for all shell and
  batch templates. The revised manuscript built successfully with LaTeX/BibTeX,
  without undefined citations/references or overflowing boxes.
- Checksums for **all 18 dissertation files** matched the before-work snapshot.
  The prior laptop manuscript was preserved verbatim in the archive.

No real Slurm job was submitted. Scheduler availability, account permissions,
the EHC node selection, quotas and target-cluster performance remain to be
checked on Apocrita. The one-hour pilot is the next actual cluster step.

## Separate resource pilots — subsequent script revision

The CP-SAT and RL resource tests now submit independent one-hour jobs with
separate frozen manifests and reports. No local performance benchmark or real
cluster job was run for this revision. Standard-library checks completed:
**17 tests passed; five existing numerical tests skipped** in the system Python
environment. New checks cover separation of the two workload plans, both Bash
submission wrappers through fake Slurm, reporting without numerical imports,
RAM headroom and core selection, incomplete/failed measurement rejection, and
distinct resident replay arrays. Bash/Python syntax checks passed. All 18
dissertation checksums remain unchanged. Target-cluster execution remains the
purpose of these scripts, rather than a claimed result.

## Partial-pilot diagnostics

After the first Apocrita reports showed incomplete RL cells, a read-only
diagnostic command was added to distinguish recorded deadlines from exceptions
and expose timings already saved by the original pilot. Revised reporting
preserves partial sample counts, excludes them from core selection, and records
its analyser hash. New pilot records additionally identify the active operation
and warmup timings. Standard-library verification: **19 tests passed, five
existing numerical tests skipped**. No new performance run was launched.
