# Peer Review (Simulated)

**Artifact reviewed:** `paper.tex` (3D HP folding: CP-SAT vs attention-DQN baseline)  
**Reviewer stance:** Tough but constructive; evidence-first  
**Date:** 2026-05-06

## 1) Summary

### Observations (from manuscript)
- The paper studies 3D HP lattice protein folding as an energy minimization problem (maximize non-consecutive H-H contacts).  
- It presents a CP-SAT formulation (OR-Tools v9.11) and compares results on standard 3d1–3d8 sequences against an attention-based DQN baseline (Liu & Iba, 2025).  
- Reported results show CP-SAT reaches best-known energies with OPTIMAL status on 3d1–3d4, and FEASIBLE but sub-best-known results on 3d5–3d8.  
- The RL side is mixed provenance: local reruns for 3d1–3d3 (200k episodes, single seed) and published numbers for later sequences.

### Inference
- The manuscript is strongest as a careful **computational baseline and methodology report**, not yet as a definitive cross-paradigm superiority claim.

---

## 2) Strengths

### Observations
1. **Clear optimization formulation:** Objective and constraints are mathematically explicit and implementation-oriented.  
2. **Solver-status transparency:** CP-SAT distinguishes OPTIMAL vs FEASIBLE rather than implying proof for all results.  
3. **Disclosure of mixed RL sourcing:** Table indicates local vs published RL values, which is honest and important.  
4. **Reasonable symmetry handling:** Fixing first residues to reduce equivalent searches is standard and well-motivated.  
5. **Practical systems focus:** Includes runtime budgets, worker counts, and hardware context.

### Inference
- The paper has good engineering transparency for the CP component and avoids overclaiming on formal optimality where not proven.

---

## 3) Major Concerns

1. **Comparison fairness is incomplete (major).**  
   - **Observation:** CP values are locally generated; RL values are partly local and partly imported from prior work.  
   - **Why this matters:** Differences may reflect environment/config variance rather than method quality.

2. **RL evaluation lacks statistical treatment (major).**  
   - **Observation:** Local RL appears single-seed for reproduced runs; no variance bands/CIs.  
   - **Why this matters:** RL performance can vary substantially by seed; point estimates are fragile.

3. **Compute-budget equivalence is not established (major).**  
   - **Observation:** CP is wall-clock-budgeted; RL described by episodes/steps.  
   - **Why this matters:** Without normalized compute (time, FLOPs, or energy), claims about relative effectiveness are underdetermined.

4. **CP tuning protocol under-specified (major).**  
   - **Observation:** Mentions tuning phases/seeds but does not fully list tuned parameters and per-instance chosen settings.  
   - **Why this matters:** Reproducibility and attribution of gains (modeling vs tuning) are unclear.

5. **Scaling claims remain preliminary (major).**  
   - **Observation:** Both methods miss best-known energies on longer instances under tested budgets.  
   - **Why this matters:** The current evidence supports "hardness increases" but not strong relative-scaling conclusions.

---

## 4) Minor Concerns

1. Add per-instance CP diagnostics (vars/constraints/conflicts/branches/memory).  
2. Clarify whether any RL hyperparameters were retuned for local reruns.  
3. Tighten terminology consistency around baseline naming history.  
4. Improve framing of CPSP results (helpful reference, not headline comparator).  
5. Include exact command lines/config files in appendix or repository.

---

## 5) Questions for Authors

1. For each sequence, what exact CP-SAT parameters were selected after tuning?  
2. Were RL local reruns done with identical code commit and preprocessing as published setup?  
3. Can you provide multi-seed RL outcomes (mean ± std / best / median) for at least 3d1–3d4?  
4. Can you provide energy-vs-wall-clock trajectories for both CP and RL on the same machine?  
5. How sensitive are CP outcomes to lattice bounding choices and symmetry constraints?

---

## 6) Recommended Experiments / Ablations (highest value first)

1. **Compute-normalized comparison:** same wall-clock budgets for RL training+inference vs CP solving.  
2. **RL multi-seed study:** ≥5 seeds per short instance; report dispersion and best-of-k.  
3. **CP ablation:** remove/add symmetry breaking, vary workers, vary grid bounds, report effect sizes.  
4. **Anytime curves:** objective vs time for both methods on 3d5–3d8.  
5. **Unified local rerun set:** produce local RL results for all 3d1–3d8 to remove mixed-source ambiguity.

---

## 7) Reproducibility Checklist

### Present
- [x] Problem formulation and objective definition  
- [x] Benchmark instances identified  
- [x] Hardware platform stated  
- [x] CP high-level budgeting described  
- [x] Distinction between local vs published RL numbers

### Missing / needs strengthening
- [ ] Full CP parameter table per instance  
- [ ] Full RL run protocol for local reproductions (all seeds, stopping rules, checkpoints)  
- [ ] Commit hashes / environment manifests / deterministic scripts  
- [ ] Statistical uncertainty reporting for RL  
- [ ] Compute-normalized comparison protocol

---

## 8) Ethics / Risk Notes

### Observations
- This is a methodological/computational benchmark paper with low direct human-subject or privacy risk.

### Inference
- Main risk is scientific: overinterpreting partially comparable baselines. This is fixable with stronger protocol alignment and uncertainty reporting.

---

## 9) Overall Recommendation

**Score:** **6.5 / 10** (weak accept for workshop/technical report; borderline for a selective conference in current form)  
**Confidence:** **0.72**

### Rationale
- **Positive:** Clear formulation, transparent CP status reporting, practical benchmarking.  
- **Blocking issues:** Incomplete apples-to-apples comparison and insufficient RL statistical rigor.

If the authors add compute-normalized comparisons, multi-seed RL statistics, and full local parity runs, this could become a substantially stronger submission.

---

## Sources

1. Local manuscript: `paper.tex` (workspace file reviewed directly).  
2. Local bibliography: `references.bib` (workspace file reviewed directly).  
3. Liu, P. & Iba, H. (2025). *Enhancing Reinforcement Learning in 3-Dimensional Hydrophobic-Polar Protein Folding Model with Attention-based Layers*. arXiv:2504.15634. https://arxiv.org/abs/2504.15634  
4. Google OR-Tools documentation: https://developers.google.com/optimization  
5. Perron, L., Didier, F., & Furnon, V. (2023). *The CP-SAT-LP Solver* (CP 2023). https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.CP.2023.3
