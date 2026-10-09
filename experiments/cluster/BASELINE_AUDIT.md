# Preserved CPU comparison baseline — 9 October 2026

The main comparison asks which method finds better independently validated
folds within a **24-hour cold-start CPU budget**, including RL training time.
It does not compare fully converged policies or claim exact reproduction of
every original-paper hyperparameter. All eight saved configuration rows are
retained, including the explicitly proposed 3d8 row.

Static inspection of the bundled project reference found:

- Action selection uses the environment's legality mask, symmetry rules and
  one-step trap check (`vendor/rl_reference.py`, `select_action`).
- The Double-DQN bootstrap selects a next action with an **unmasked argmax**
  (`update`). This is an inherited limitation and a possible source of target
  bias. It has not been shown to cause the observed weak policy scores.
- The target network is in evaluation mode; the policy is in training mode
  during training. Greedy evaluation temporarily switches policy mode and
  restores it in `finally` (`worker.py`, `evaluate`).
- The cluster runner preserves FP32, model architecture, Adam, replay and
  the 200,000-episode epsilon schedule, with at most one update per episode.
  These settings led to only 2.4–10.0% of training completing in the 24-hour
  representative runs. Actual episode counts and greedy outcomes accompany
  final search scores; a budget stop is not training completion.
- Independent geometry validation checks complete integer self-avoiding folds,
  unit backbone bonds, shared cube bounds and non-backbone HH contacts.
  Restricted repairs do not provide full-search optimality certificates.

No algorithm correction is applied in this continuation: changing the target,
schedule or architecture would prevent pooling the six completed observations
with the remaining runs. A masked-target variant or a learning positive control
would be a separate declared experiment. No new numerical learning audit or
local training run was executed while preparing the submission.

Engineering verification covers the actual submission and analysis paths with
synthetic completed artifacts: exact 80-observation matrix, 74 new/6 reused
tasks, source/configuration/version/device/CPU checks, rejection of failed and
short-budget runs, independent witness validation, raw-result hash changes,
combined five-seed denominators, no duplicate observations, original deadline
inheritance, latest usable greedy evaluations and fake Slurm array submission.
These checks do not establish convergence. Numerical tests requiring PyTorch
or OR-Tools are skipped when those packages are absent from the local runtime.

The feasibility seed-0 outcomes were viewed before the two-method campaign
scope was selected. Report that provenance and avoid describing all 80
observations as an entirely unseen confirmatory sample. The learned
DecisionBoost experiment retains its separate protocol and cost accounting.
