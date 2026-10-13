# HP folding: arXiv manuscript and experiments

The revised paper is [paper.tex](paper.tex); its structure and completed
CPU comparison are described in [arxiv_outline.md](arxiv_outline.md).

The cloneable **CPU-only Apocrita bundle** is in
[experiments/cluster/](experiments/cluster/README.md). It provides setup, a
one-hour pilot, configurable per-run cutoffs (24 hours by default), ten-day Slurm
requests, an absolute ten-day campaign deadline, checkpointing and independent
fold verification. The completed comparison has 80 valid observations: eight
sequences, five seeds and two methods under 24-hour CPU budgets. CP-SAT leads
on six sequences and ties RL on the two shortest. Reported final results and
their provenance are in [experiments/cluster/results/](experiments/cluster/results/README.md).

The new [variable-length policy experiment](experiments/folding_policy/README.md)
tests PatternBoost and CP-SAT pretraining followed by RL on the same Transformer.
Its [prospective plan](experiments/folding_policy/PLAN.md) starts with a bounded
validation-only Apocrita pilot; final benchmark evaluation is disabled in that
pilot. This policy generates folds independently at inference.
The [two-hour allocation pilot](experiments/folding_policy/ALLOCATION_PLAN.md)
uses shared pretraining to compare 60:60 and 90:30 PatternBoost/RL allocations
with two-hour single-stage controls, over three seeds.
The next [saved-policy comparison](experiments/folding_policy/INFERENCE_PLAN.md)
reuses those checkpoints and adds matched CP-SAT and greedy validation searches,
reporting both inference-only budgets and training costs shared over 16 sequences.

The earlier laptop manuscript is preserved in
[archive/paper_laptop_draft_20261006.tex](archive/paper_laptop_draft_20261006.tex).
The `thesis/` directory is the dissertation and is excluded from this work.

Other local exploratory experiments/results are separate from this portable
bundle; the cluster scripts do not depend on them or on neighbouring projects.
