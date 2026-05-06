# Venue Recommendation Memo

**Project:** 3D HP lattice protein folding paper comparing CP-SAT and reinforcement learning  
**Date:** 2026-05-06

## Executive recommendation

**Recommended first target:** **Physical Biology**  
**Recommended backup:** **Computer Physics Communications**  
**Third option:** **Journal of Biological Physics**  
**Not recommended as first target in current form:** **Journal of Computational Physics**

## Why this recommendation

The current manuscript reads primarily as:
- an **algorithm/methods paper**,
- on a **simplified biophysical model** (3D HP lattice protein folding),
- with a **computational comparison** (CP-SAT vs RL),
- but **not yet a strong biological-discovery paper**, and
- **not primarily a new numerical PDE / continuum computational physics paper**.

That makes the best venue more likely to be a **physics-adjacent or computational-biophysics journal** rather than a top-tier general computational physics journal.

## Ranked venue shortlist

### 1) Physical Biology
**Why it fits**
- Official scope explicitly covers quantitative theoretical/modeling work bridging biology with physics.
- It names **molecular biophysics** and **multiscale modeling and simulation approaches**.
- The paper can be framed as a coarse-grained/statistical-physics treatment of protein folding with algorithmic comparison.

**Why it may work better than JCP**
- Editors and reviewers are more likely to accept a careful **toy-model biophysics** paper.
- The manuscript does not need to be sold as a mainstream numerical physics paper.

**Main risk**
- The manuscript should emphasize **biophysical significance**, not only solver engineering.

**Verdict**
- Best current fit if the goal is a **physics journal**.

### 2) Computer Physics Communications
**Why it fits**
- Official scope emphasizes **computational methods and techniques and their implementation** for substantive physics problems.
- It explicitly values implementation and performance details.

**Why it may be a strong backup**
- More natural than JCP for a **method + implementation + benchmark** paper.
- Good fit if code, scripts, and reproducibility are packaged well.

**Main risk**
- The manuscript still needs to justify the **physics relevance** and not read only as an operations-research benchmark.

**Verdict**
- Strong candidate if the paper is positioned as a **reproducible computational physics / biophysics software-method paper**.

### 3) Journal of Biological Physics
**Why it fits**
- Official scope welcomes physics-based approaches to biological problems.
- It explicitly includes **Biological Physics of Nucleic Acids and Proteins**.

**Why it could work**
- More tolerant of coarse-grained conceptual biological-physics models than general computational physics journals.

**Main risk**
- Reviewers may want more physical interpretation if the manuscript reads too much like a benchmark paper.

**Verdict**
- Good physics-facing option if the biophysics framing is strengthened.

### 4) Journal of Molecular Graphics and Modelling
**Why it fits**
- Scope includes computer-based theoretical investigation of molecular structure and molecular modeling.

**Main risk**
- Some reviewers may expect a stronger tie to realistic molecular modeling than the HP lattice model provides.

**Verdict**
- Reasonable fallback option.

### 5) Computational and Mathematical Biophysics
**Why it fits**
- Scope strongly matches the paper: computational and mathematical methods, numerical algorithms, optimization, machine learning, biomolecular structure.

**Main risk**
- Lower visibility/prestige than the leading options above.

**Verdict**
- Strong scope-fit safety option.

## Why Journal of Computational Physics is not the best first target

JCP’s official aims and scope emphasize innovative computational methods for physical problems, including robustness, computational complexity, comparison to prior approaches, and reproducibility.

However, in practice, JCP is closely associated with:
- advanced numerical modeling,
- PDE and continuum methods,
- scientific machine learning for physical simulation,
- and broadly applicable computational-physics methodology.

The current paper is instead a:
- discrete combinatorial optimization study,
- on a simplified protein-folding model,
- with a benchmark-style comparison.

That does not make JCP impossible, but it creates a **high editorial mismatch risk** and therefore a **higher desk-reject risk** than the better-matched venues.

## What would improve fit by venue

### For Physical Biology / Journal of Biological Physics
Emphasize:
- HP model as a **coarse-grained statistical-physics / biophysics testbed**,
- what the comparison teaches about **energy landscapes and search difficulty**,
- why exact vs approximate search matters scientifically,
- biological or physical interpretation of instance difficulty.

### For Computer Physics Communications
Emphasize:
- implementation details,
- reproducibility,
- benchmark protocol,
- performance diagnostics,
- code availability,
- reusable software artifact.

### For Journal of Computational Physics
Would likely require:
- stronger methodological novelty,
- broader computational-physics relevance,
- deeper complexity and performance analysis,
- cleaner apples-to-apples baseline study,
- more ambitious framing than the current manuscript appears to support.

## Final recommendation

If choosing today:
1. **Physical Biology**
2. **Computer Physics Communications**
3. **Journal of Biological Physics**

## Sources

1. Journal of Computational Physics — official aims & scope  
   https://www.sciencedirect.com/journal/journal-of-computational-physics
2. Computer Physics Communications — official aims & scope  
   https://www.sciencedirect.com/journal/computer-physics-communications
3. Physical Biology — official scope and key information  
   https://iopscience.iop.org/journal/1478-3975/page/Scope-and-key-information
4. Journal of Biological Physics — official aims and scope  
   https://link.springer.com/journal/10867/aims-and-scope
5. Journal of Molecular Graphics and Modelling — official aims & scope  
   https://www.sciencedirect.com/journal/journal-of-molecular-graphics-and-modelling
6. Computational and Mathematical Biophysics — official aims and scope  
   https://www.degruyterbrill.com/journal/key/cmb/html?lang=de
7. PLOS Computational Biology — official journal information and scope  
   https://journals.plos.org/ploscompbiol/s/journal-information
