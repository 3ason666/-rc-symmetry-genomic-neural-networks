# Phase 2E biological-correctness protocol

Protocol revision: `phase2e_biological_correctness_v1`  
Frozen date: 2026-08-22  
Status: frozen before any Phase 2E endpoint is computed

## Question and evidence boundary

Phase 2E asks whether reverse-complement (RC) attribution symmetry is distinct from
biological correctness. It reuses the completed Phase 2D `CNN-Raw`, `CNN-Aug`, and
`CNN-RCPS` checkpoints without retraining, reselection, or hyperparameter changes.
The frozen GATA1 motif, ChIP-seq summit, motif-disruption, and conservative-peak
endpoints are evaluated independently of model symmetry.

This phase can show that exact RC symmetry does not by itself determine motif or
functional localization. It cannot establish the complete in-vivo regulatory mechanism,
and a null difference between models is not interpreted as proof of equivalence.

## Data and leakage boundary

- Primary attribution cohort: the 240 Phase 2D frozen attribution samples, including
  120 positive and 120 matched negative samples across the three chromosome folds.
- Primary motif and disruption endpoints use the 120 positive rows. Matched negative
  rows are a biological null/control and never replace the positive endpoint.
- Conservative sensitivity cohort: only rows with original split `train` from the
  frozen conservative-IDR matched dataset. Each chromosome is evaluated only by its
  Phase 2D fold-holdout model.
- Original validation chromosomes and test chromosomes remain forbidden. Phase 2E
  does not authorize test unsealing.
- All Phase 2D checkpoints, seeds, fold assignments, sample IDs, and Exact ISM matrices
  are immutable inputs. Any missing or changed checksum is an audit failure.

## Frozen external motif resource

- Resource: JASPAR CORE 2026, human GATA1 `MA0035.5`, ChIP-seq profile.
- Local resource: `metadata/jaspar_MA0035.5_GATA1_2026.json`.
- Local SHA-256: `a180d7d4c7cbb159f1c9d25eb2a6b11774ca07467bb9bbee2e55a3e47413e9ee`.
- The 7-position PFM is converted to a log2-odds PWM using a pseudocount of 0.5 and
  a uniform 0.25 genomic background. Both strands are scanned.
- Relative score is `(score - theoretical minimum) / (theoretical maximum - minimum)`.
- Primary strong-hit threshold: 0.80. Thresholds 0.85 and 0.90 are sensitivity analyses.
- The highest-scoring hit across both strands is the primary hit. Ties are resolved by
  lower zero-based start, then forward strand.

No motif version, PWM transformation, threshold, or tie rule may be changed after
endpoint computation begins.

## Primary endpoints

### A. Attribution localization to GATA1 motif

On strong-hit positive samples, use forward-orientation Exact ISM absolute positional
attribution from Phase 2D.

1. Motif mass fraction: attribution mass inside the seven-base best hit divided by total
   attribution mass.
2. Top-7 motif recall: fraction of the seven motif positions present among the seven
   highest-attribution sequence positions.
3. Distance: attribution-weighted distance to the best-hit center.

The primary localization endpoint is motif mass fraction. Circularly shift each
attribution vector by 64 deterministic non-zero offsets and compute the observed/control
mass ratio. This preserves the attribution distribution while breaking its alignment to
the motif.

### B. Motif-disruption specificity

For every strong-hit positive sequence, replace each best-hit base with the minimum-PWM
base at that motif column; if the original already equals the minimum, use the
second-lowest base. A matched control edit applies the same number of deterministic base
changes to a non-overlapping flank window selected to mirror the motif's distance from the
sequence center whenever possible.

For every frozen model checkpoint report:

- original logit minus motif-disrupted logit;
- original probability minus motif-disrupted probability;
- motif logit drop minus matched-flank logit drop (primary disruption endpoint).

Positive specificity means motif disruption damages the model output more than an edit
with the same mutation count away from the motif.

### C. Summit localization

Retain the Phase 2D center-16-bp mass fraction and attribution-weighted distance to the
fixed summit index 128. Summit localization is complementary evidence and is never used
as a surrogate for a specific GATA1 motif.

### D. Conservative-IDR sensitivity

Evaluate prediction AUROC, AUPRC, mean RC probability difference, and flip rate on all
12,552 conservative-IDR train-split rows using the corresponding chromosome-holdout
checkpoint. This analysis checks whether conclusions are specific to the optimal-IDR peak
definition. It is a sensitivity endpoint, not an external validation set.

## Controls and robustness

1. Matched negative sequences for motif prevalence and score enrichment.
2. Sixty-four deterministic non-zero circular attribution shifts per sample-run.
3. Matched flank edits with the same number of substitutions as motif disruption.
4. Relative motif thresholds 0.85 and 0.90.
5. Optimal-IDR primary samples versus conservative-IDR matched sensitivity data.
6. Forward and RC orientation agreement is retained as an audit, not reused as biological
   evidence.

## Statistics and multiplicity

- Unit: fold, seed, sample. Comparisons are paired on all three identifiers.
- Fixed model contrasts: `CNN-Aug - CNN-Raw` and `CNN-RCPS - CNN-Aug`.
- Use 10,000 fold -> seed -> sample hierarchical bootstrap replicates with seed 20260822.
- Report estimates and 95% percentile intervals. Never erase unfavorable or null results.
- Apply Benjamini-Hochberg FDR within the frozen confirmatory family: two model contrasts
  for motif mass fraction, two for disruption specificity, and the three per-model tests
  of disruption specificity against zero.
- Threshold sensitivity, top-7 recall, distance, summit metrics, negative controls, and
  conservative-IDR performance are secondary/descriptive unless explicitly stated above.

## H4 interpretation rule

H4 is **empirically supported as a distinction** when the Phase 2D RCPS symmetry gates
remain passed and biologically relevant quantities still vary across RCPS samples. The
predefined diagnostic is the proportion of strong-hit RCPS sample-runs with either
top-7 motif recall below 0.5 or motif-versus-flank logit specificity no greater than zero.
A bootstrap 95% lower bound above zero supports the claim that exact symmetry does not
guarantee uniformly correct motif localization/use.

This rule does not label every diagnostic failure as a biologically wrong prediction and
does not claim that augmentation or RCPS caused the failure. If the lower bound includes
zero, H4 is reported as inconclusive on these endpoints. Between-model null results alone
cannot support equivalence.

## Gates and stopping rule

- Require all 27 Phase 2D checkpoints and all 27 Exact ISM NPZ files.
- Require Phase 2D scientific gates and test seal to remain passed.
- Require resource and input SHA-256 checksums, expected sample counts, finite endpoints,
  exact fold routing, and zero original-validation/test rows.
- Complete and report P2E regardless of direction. No model, motif, threshold, endpoint,
  or control is tuned in response to P2E results.
- Phase 2F remains unauthorized until a separate one-time test protocol is frozen.

