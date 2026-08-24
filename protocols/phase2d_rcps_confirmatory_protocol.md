# Phase 2D confirmatory RC-symmetry protocol

Protocol revision: `phase2d_rcps_confirmatory_v1`  
Frozen date: 2026-08-22  
Status: frozen before any Phase 2D model output is generated

## Scope and evidence hierarchy

Phase 2D is the confirmatory architecture-family comparison. Its fixed conditions are
`CNN-Raw`, `CNN-Aug`, and the parameter-tied, strictly reverse-complement (RC)
equivariant/invariant `CNN-RCPS`. All CNN-versus-Transformer results from Phase 2C are
exploratory and cannot support a causal architecture or positional-encoding claim.
No further Transformer performance-matching remedy, hyperparameter, or architecture
search is permitted after the frozen Phase 2C.3 screen completes.

Phase 2D tests whether progressively stronger RC handling changes (1) prediction
symmetry and (2) RC-aligned attribution symmetry. Classification performance is always
reported, but is not a tuning trigger and is not an eligibility gate for the symmetry
endpoints. Biological validity is evaluated separately: better symmetry is not assumed
to imply better biology.

## Data boundary and leakage controls

- Dataset: the already frozen Phase 2 matched dataset with SHA-256
  `54928c8be54d310a43a7e4daaf8516ab054775c043a5cabac76fbe60ad034f88`.
- Phase 2D modeling uses only rows whose original split is `train`.
- The original validation chromosomes have been repeatedly inspected and are not an
  independent confirmation resource. They are excluded from Phase 2D model fitting,
  checkpoint selection, prediction, attribution, and biological endpoints.
- The test chromosomes remain sealed. This v1 protocol does **not** authorize test
  unsealing.
- Use the frozen P2C.3 three chromosome folds. Every train chromosome appears in exactly
  one holdout fold. Matched pairs and RC-canonical groups may not cross a fold boundary.
- All three model conditions use identical fold assignments, seeds, optimization budget,
  validation-loss checkpoint selection, and deterministic attribution samples.

This is confirmatory for the previously unseen Phase 2D RCPS contrasts, but it is not a
substitute for a new external dataset because the broader project has already used the
training chromosomes.

## Fixed model and training conditions

- Conditions in fixed order: `CNN-Raw`, `CNN-Aug`, `CNN-RCPS`.
- Seeds: 42, 123, 2026.
- CNN widths/kernels: 32 channels at kernel 15, then 24 channels at kernel 7.
- RCPS uses the same realized feature widths. Its smaller number of independent
  parameters is an inherent consequence of weight tying and is reported, not repaired by
  width search.
- Adam, learning rate 0.001, weight decay 0.0001, batch size 128, at most 30 epochs,
  patience 5, selection by inner-holdout loss.
- RC augmentation is stochastic replacement with probability 0.5 on training rows only.
- RCPS is trained without augmentation; augmentation is not combined with RCPS in the
  primary design.
- No architecture, optimizer, scheduler, width, kernel, learning-rate, patience, or
  augmentation-probability change is allowed in response to Phase 2D results.

## Endpoints

Primary prediction endpoint: per-sample `|p(S)-p(RC(S))|`, summarized by the mean for
each fold-seed run. Prediction flip rate, P95 difference, and probability correlation are
secondary.

Primary attribution endpoint: Exact ISM, logit difference, absolute-position
RC-aligned normalized L1 distance. Absolute Pearson, top-8 overlap, normalized L2, signed
position metrics, and the full nucleotide matrix are retained. Integrated Gradients,
DeepLIFT, and GradientSHAP use the already frozen RC-compatible baselines and are
secondary method-robustness checks.

The two fixed contrasts are:

1. `CNN-Aug - CNN-Raw` (effect of augmentation).
2. `CNN-RCPS - CNN-Aug` (effect of strict equivariance after augmentation).

Negative differences favor the later condition for both primary asymmetry endpoints.
All effects are paired by fold, seed, and sample. Report estimates and a hierarchical
bootstrap 95% interval; retain unfavorable and null results. RCPS symmetry does not
require performance matching. AUROC/AUPRC and their paired differences are reported to
separate symmetry from predictive utility.

## Attribution sample freeze

Within every holdout fold, select 40 positive and 40 negative samples using a fixed random
seed before model predictions are inspected. The same 80 IDs are used by all three models,
all three training seeds, both orientations, and all attribution methods. No hard-negative
or confidence-based sampling is allowed in the primary Phase 2D analysis.

## Numerical and reproducibility gates

- Reload every saved checkpoint into a newly constructed model and require maximum
  probability error no greater than `1e-6`.
- For RCPS require maximum forward-versus-RC probability error no greater than `1e-6`,
  feature-map equivariance error no greater than `1e-5`, and RC-aligned Exact ISM matrix
  error no greater than `1e-5`. The feature tolerance reflects deterministic float32
  accumulation error observed in the already implemented mathematical RCPS construction;
  it is frozen before the multi-seed Phase 2D run.
- Verify double RC is exact, expected row counts are complete, no forbidden split is
  accessed, and all finite-required endpoint rows are present.
- Any failed numerical/reload/completeness gate is a software/audit failure to diagnose,
  not permission to tune model performance.

## Biological-validity staging

Phase 2D records a resource-free summit-localization endpoint on positive examples because
positive windows are centered on the frozen ChIP-seq summit: absolute attribution mass in
the central 16 bp and attribution-weighted distance to the center. These are secondary,
not evidence that a specific motif was learned.

GATA1 motif overlap, motif disruption, conservative-peak sensitivity, and any external
annotation endpoint belong to Phase 2E. Before Phase 2E runs, its motif/annotation resources,
checksums, motif thresholds, disruption rule, sample set, multiplicity handling, and test
policy must be frozen. Phase 2D v1 does not authorize test access.

## Decision rule

Complete and report Phase 2D regardless of effect direction. A failed strict-invariance or
checkpoint-reload gate blocks scientific interpretation until the implementation/audit is
fixed. A predictive-performance shortfall limits biological interpretation but does not
trigger further CNN-versus-Transformer or RCPS hyperparameter search. Test remains sealed
after Phase 2D unless a later, separately frozen protocol explicitly authorizes a single
final evaluation.
