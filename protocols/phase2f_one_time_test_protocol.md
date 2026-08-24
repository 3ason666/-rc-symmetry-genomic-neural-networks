# Phase 2F one-time sealed-test protocol

Protocol revision: `phase2f_one_time_test_v1`  
Authorization date: 2026-08-22  
Status at freeze: authorized, endpoints not observed

## Purpose

Phase 2F is the single final evaluation on the previously sealed chromosome test split.
It tests whether the confirmatory Phase 2D reverse-complement (RC) findings and the
Phase 2E biological-correctness distinction reproduce without further model selection,
hyperparameter tuning, threshold changes, or hypothesis rewriting.

## Frozen evidence entering Phase 2F

- Phase 2D completed all 27 planned CNN runs and all numerical gates.
- Phase 2E completed all motif, disruption, summit, and conservative-IDR gates.
- The original validation split has been observed during earlier pilots and is not an
  independent confirmatory resource.
- The test split has had zero model predictions, zero model attributions, and zero model
  metrics before this authorization.

## Frozen model rule

No model is retrained. Phase 2F uses exactly the 27 `best_checkpoint.pt` files produced
by Phase 2D:

- `CNN-Raw`, `CNN-Aug`, and `CNN-RCPS`;
- training folds `fold_a`, `fold_b`, and `fold_c`;
- seeds 42, 123, and 2026.

Every checkpoint is evaluated on the same test samples. Comparisons are paired by
training fold, seed, and test sample. A SHA-256 manifest of every checkpoint is created
before the first test row is loaded and is immutable thereafter.

## Frozen test population

- Dataset: `data/phase2/processed/phase2_matched_dataset.csv`.
- Dataset SHA-256: `54928c8be54d310a43a7e4daaf8516ab054775c043a5cabac76fbe60ad034f88`.
- Test chromosomes: chr3, chr8, chr14, and chr18.
- Expected test rows: 3,496, balanced as 1,748 positives and 1,748 negatives.
- All test rows are used for prediction, classification, and RC-consistency endpoints.

The attribution cohort is selected before any prediction is computed. Within each test
chromosome and label, rows are ordered by SHA-256 of
`phase2f_v1|chromosome|label|sample_id|canonical_key`; the first 10 are retained. This
produces 80 sequences: 40 positives and 40 negatives, balanced across four chromosomes.
All 27 checkpoints use the identical cohort.

## Frozen prediction endpoints

For every checkpoint and every test sample, evaluate the original sequence S and
RC(S), using threshold 0.5.

Primary endpoints:

1. AUROC and AUPRC on S.
2. Mean per-sample absolute probability difference `|p(S)-p(RC(S))|`.
3. Symmetry flip rate at threshold 0.5.
4. Paired contrasts `CNN-Aug - CNN-Raw` and `CNN-RCPS - CNN-Aug` for probability
   difference. Negative values favor the later model.

The mean prediction across the nine checkpoints of each model is reported as a
secondary ensemble description. It is not used to select a winner or hide any run.

## Frozen attribution endpoints

Exact ISM in logit space is the only test attribution method. For each frozen sequence,
compute forward and RC attribution matrices, align the RC result back to the S
coordinate system, and report:

1. absolute-position normalized L1 difference (primary);
2. Pearson correlation and Top-8 overlap (descriptive);
3. full 4-by-L matrix maximum residual for the RCPS numerical gate;
4. CNN-Raw asymmetry among samples with prediction difference at most 0.01;
5. residual CNN-Aug asymmetry;
6. paired attribution contrasts matching the prediction contrasts.

The RCPS prediction tolerance is 1e-6 and the aligned Exact ISM matrix tolerance is
1e-5.

## Frozen hypothesis decision rules

- H1a reproduces if the lower 95% confidence limit for CNN-Raw mean probability
  difference is greater than 0.01.
- H1b reproduces if the lower 95% confidence limit for CNN-Raw Exact ISM normalized
  L1 among prediction-consistent samples is greater than 0.05.
- H2 reproduces if the upper 95% confidence limit for `CNN-Aug - CNN-Raw` prediction
  difference is below zero.
- H3a reproduces if the lower 95% confidence limit for CNN-Aug Exact ISM normalized
  L1 is greater than 0.05.
- H3b reproduces if CNN-RCPS maximum prediction difference is at most 1e-6 and its
  maximum aligned Exact ISM matrix residual is at most 1e-5.
- H3c is not retested and remains exploratory.
- H4 reproduces only as a conceptual distinction under the diagnostic rule specified
  below; it never declares all flagged samples biologically wrong.

## Frozen biological endpoints and H4

Use the already frozen JASPAR human GATA1 MA0035.5 PWM resource, scan both strands,
and use relative score 0.80 as the primary strong-hit threshold. On positive attribution
samples with a strong hit:

- report Exact ISM absolute mass within the best motif;
- report Top-7 motif recall;
- compare localization with the same 64 non-zero circular shifts used in Phase 2E;
- disrupt the motif with the frozen minimum-PWM-base rule;
- compare the motif logit drop with an equal-mutation-count flank edit.

The Phase 2F H4 diagnostic is unchanged: within CNN-RCPS sample-runs, flag
`Top-7 motif recall < 0.5 OR motif-minus-flank logit specificity <= 0`.

H4 is supported only as a distinction between concepts if the bootstrap lower 95%
confidence limit for the diagnostic rate is above zero. This does not label every flagged
sample biologically wrong and does not prove that RCPS causes a biological error.

## Statistics

- Use 10,000 hierarchical paired bootstrap replicates.
- Hierarchy for checkpoint-level endpoints: training fold -> seed -> test sample.
- Comparisons use the same fold, seed, and sample on both models.
- Fixed bootstrap seed: 20260822, with deterministic offsets per endpoint.
- Report effect estimates and percentile 95% confidence intervals.
- All prespecified confirmatory p-values, if generated, receive BH-FDR correction.
- Preserve zero effects, unfavorable effects, failures, NaNs, and constant-attribution
  diagnostics; do not silently filter them.

## One-time unsealing and restart policy

The runner writes an authorization/checksum ledger before loading the first test row.
There is no force-rerun option. If an operating-system or software interruption occurs,
the same frozen run may resume only when protocol, config, dataset, PWM, and all
checkpoint checksums are unchanged. Completed checkpoint stages are not recomputed.
After the completion marker is written, any second execution must abort before loading
test data.

No decision may be changed after test endpoints are observed. Test results cannot be
used to tune, retrain, replace, or selectively report models.
