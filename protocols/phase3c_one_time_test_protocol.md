# Phase 3C one-time sealed-test protocol

Protocol revision: `phase3c_one_time_test_v1`  
Authorization date: 2026-08-23  
Status: analysis specification frozen before semantic access to either Phase 3 test table

## Scope and hierarchy

This protocol evaluates the 54 checkpoints frozen after Phase 3B. Fetal GATA1 is
the primary confirmatory task. GM12878 CTCF is supportive cross-TF generalization
and is interpreted separately. CTCF non-support does not negate the Phase 2 result.

Each task has one permanent chromosome-held-out test set shared by all three
development folds, three seeds, and three model types. Test rows are used once and
never for early stopping, model selection, threshold selection, debugging, or
post-test tuning.

## Frozen prediction analysis

For every checkpoint, predict both `S` and `RC(S)` for every test sequence. Report
AUROC, AUPRC, mean/median/P95 absolute probability difference, Pearson, Spearman,
and symmetry flip rate at threshold 0.5. The model-level ensemble is the arithmetic
mean across the nine fold/seed fits of the same model type.

## Frozen attribution cohorts

Exact ISM in logit space is the primary attribution method.

The base attribution cohort is selected before model predictions by a SHA-256 rank
within each frozen test chromosome and label. It contains 10 samples per chromosome
per label: 60 fetal-GATA1 samples and 80 GM12878-CTCF samples. All 27 checkpoints
per task are evaluated on this cohort.

H1b uses a separate, prespecified prediction-consistent cohort. After all test
predictions have been generated, average `p(S)` and `p(RC(S))` over the nine
CNN-Raw fits for each sample. Eligible samples have absolute difference at most
0.01. Select exactly 100 eligible samples by the frozen SHA-256 rank, with
deterministic chromosome/label round-robin balancing. Only CNN-Raw Exact ISM is
run for this H1b cohort. If fewer than 100 eligible samples exist, H1b is reported
as not estimable; the threshold, cohort size, and selection rule are not changed.

## Frozen hypotheses

- H1a: fetal-GATA1 CNN-Raw mean absolute prediction difference has 95% CI lower
  bound greater than 0.01.
- H1b: the H1b cohort's CNN-Raw Exact-ISM normalized L1 asymmetry has 95% CI lower
  bound greater than 0.05.
- H2: the paired CNN-Aug minus CNN-Raw prediction-difference contrast has 95% CI
  upper bound below zero.
- H3a: CNN-Aug base-cohort Exact-ISM normalized L1 has 95% CI lower bound greater
  than 0.05.
- H3b: CNN-RCPS maximum prediction residual is at most 1e-6 and maximum RC-aligned
  Exact-ISM matrix residual is at most 1e-5.
- H4: biological correctness remains a separate diagnostic. For strong PWM hits,
  report motif localization versus circular shifts and equal-mutation-count motif
  disruption versus matched-flank disruption. H4 is not part of the composite
  H1-H3 replication decision.

Use 10,000 hierarchical paired bootstrap replicates over model fit and sample.
Fold direction agreement is descriptive robustness, not a biological replicate
vote. Report each task separately before the primary-task conclusion.

## Interpretability and boundary gates

CTCF non-support may be discussed as a possible functional boundary only if all
data-quality gates remain passed, median development AUROC is at least 0.80, test
ensemble AUROC is at least 0.75, the H1b eligible pool contains at least 100
samples, and every primary confidence interval has width at most 0.15. Otherwise
CTCF non-support is labeled inconclusive.

## One-time execution lock

Before unsealing, freeze SHA-256 values for this protocol, its YAML configuration,
the runner, both PWM resources, the P3A/P3B manifests, all 54 checkpoints, and all
54 training histories. The input manifest is immutable. A ledger is written before
loading test rows. A completed run cannot be executed again; `--resume` is allowed
only for the identical frozen manifest after interruption. No force-rerun option is
provided.
