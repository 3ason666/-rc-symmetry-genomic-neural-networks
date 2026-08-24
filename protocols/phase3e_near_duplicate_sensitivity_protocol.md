# Phase 3E cross-phase near-duplicate sensitivity protocol

Protocol revision: `phase3e_near_duplicate_sensitivity_v1`  
Frozen on: 2026-08-23  
Status: post-primary sensitivity only

## Purpose

Phase 3D identified P3 development sequences that were at least 90% identical
to a P2F sealed-test sequence over at least 90% of both sequences, allowing the
reverse-complement orientation. This protocol implements the sensitivity rerun
that was prespecified in the frozen Phase 3A SOP.

## Fixed exclusion

- The only exclusion source is the locked Phase 3D
  `high_similarity_pairs.csv.gz` artifact.
- For each task and fold, a confirmed P3 sample is removed only when its frozen
  fold assignment is `train`.
- Validation rows are retained unchanged. P3 test rows are retained unchanged.
- Exact-sequence and coordinate findings do not add any further deletion rule:
  exact overlap remained below the frozen 1% threshold and coordinate overlap
  is reported as a locus stratum rather than treated as contamination.

## Fixed rerun

- Refit CNN-Raw, CNN-Aug and CNN-RCPS for folds 1-3 and seeds 42, 123 and 2026.
- Reuse the Phase 3B architecture, optimization settings, early stopping rule,
  augmentation probability and validation-only checkpoint choice without
  modification.
- Reuse the Phase 3C test population, prediction endpoints, Exact ISM cohorts,
  bootstrap settings, hypothesis thresholds and interpretability gates without
  modification.
- No hyperparameter tuning, model selection, threshold revision or cohort
  revision is permitted after observing the primary P3 result.

## Interpretation

This is a robustness analysis. It cannot replace or retroactively redefine the
locked Phase 3C primary analysis. Agreement supports robustness to cross-phase
near duplicates; disagreement must be reported as sensitivity of the primary
claim to those training sequences.

All generated checkpoints, exclusions, row counts, hashes, and decisions are
recorded under an independent Phase 3E output directory. A completed evaluation
cannot be force-rerun.

