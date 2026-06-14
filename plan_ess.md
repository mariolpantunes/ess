# ESS Modernization and Refactoring Plan (Updated)

## Overview
This document outlines the actionable steps to modernize the Empty Space Strategy (ESS) library. It focuses on code quality, performance stability in high dimensions, and initial distribution generation. **Crucially, do not proceed to subsequent implementation steps without comparing results against the baseline.**

## Phase 0: Benchmarking and Baseline Generation
**Goal:** Establish a quantifiable baseline for execution time and coverage before any algorithmic changes.
1. **Create `benchmark.py`:** Write a script in the `examples/` directory.
   - Use `argparse` to allow dimension overriding (default: `[2, 4, 8, 10, 20]`).
   - For each dimension, initialize a fixed number of obstacles/bounds.
   - Run the standard `esa()` / `ess()` generation.
   - Record execution time.
   - Calculate spatial coverage metrics using `ess.utils` (e.g., Clark-Evans, Maximin).
2. **Generate Baseline:** Run `python3 examples/benchmark.py --out baseline.json`. This file must be kept as the ground-truth reference to detect regressions or quantify improvements.

## Phase 1: Code Quality & Infrastructure Improvements
**Goal:** Modernize the codebase structure, fix existing nearest-neighbor bugs, and enforce quality standards.
1. **Create a WIP Branch:** Create a branch named `wip/modernization` to isolate these changes.
2. **Packaging Migration:** 
   - Move all configuration from `setup.cfg` to `pyproject.toml`.
   - Delete `setup.cfg`.
3. **Metadata Cleanup:** 
   - Remove legacy module-level metadata (`__author__`, `__version__`, etc.) from all files in `src/ess/` and `test/`.
4. **Internal Imports:** Refactor absolute internal imports in `ess.py` and `legacy.py` to relative imports (e.g., `from . import nn`).
5. **CI/CD pipeline:** Update GitHub Actions workflows for Python 3.10, 3.12, 3.14. Include pdoc documentation generation and PyPI publishing.
6. **Bug Fixes:**
   - Fix collision detection in `FaissBaseNN.range_search` by removing the `dists_sq_active > 1e-9` clause.
   - Fix index clipping in `_compute_knn_forces` to properly mask missing neighbors rather than clipping to arbitrary indices.

## Phase 2: Feature Implementations
**Goal:** Implement algorithm enhancements individually on the WIP branch and measure their isolated impact against `baseline.json`.

### Step 2.1: Log-Space Force Formulation (Option A: Capped Restoration)
**Goal:** Prevent floating-point underflow/overflow in high dimensions while preserving relative dynamics.
- Modify the force calculation functions to operate in log-space:
  $$\log F_{ij} = \log \alpha - (D - 1) \log d_{ij}$$
- **Summation and Offset:** Offset forces per particle by the maximum log-force $M_i = \max_k(\log F_{ik})$ to ensure safe exponentiation:
  $$\vec{\tilde{F}}_i = \sum_j \exp(\log F_{ij} - M_i) \cdot \frac{\vec{r}_{ij}}{d_{ij}}$$
- **Scale Restoration with Cap:** Restore the scale factor with a cap to prevent giant unstable steps:
  $$\vec{F}_i = \exp(\min(M_i, \log(\text{force\_cap}))) \cdot \vec{\tilde{F}}_i$$
- *Validation:* Run `benchmark.py`. Compare 10D and 20D coverage and time against the baseline.

### Step 2.2: Latin Hypercube Sampling (LHS) Integration
**Goal:** Maximize initial distribution efficiency to reduce optimization epochs.
- Implement `LHCSampler` (LHS) in pure NumPy within `ess/utils.py` (or a new `ess/samplers.py`).
- **Condition 1 (No initial samples given):** If no initial samples are given, use LHS to initialize the first population directly.
- **Condition 2 (Initial samples are given):** If initial samples are given, use LHS strictly to generate the candidate coordinate pool within `_smart_init`, replacing the standard random uniform candidate generation.
- *Validation:* Run `benchmark.py` specifically using LHS. Compare epoch convergence and coverage against the baseline.

## Phase 3: Unified Validation
**Goal:** Combine all improvements and validate overall library enhancement.
1. Ensure both the Log-Space formulation and LHS sampling are merged together in the WIP branch.
2. Run `benchmark.py`.
3. Generate a `final_report.json` to prove the combined improvements in high-dimensional stability, code cleanliness, and convergence speed over the initial baseline.
