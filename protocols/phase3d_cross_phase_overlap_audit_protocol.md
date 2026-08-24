# Phase 3D post-primary P2F/P3 overlap audit protocol

Protocol revision: `phase3d_cross_phase_overlap_audit_v1`

## Purpose and timing

This is the post-primary audit frozen in the Phase 3A SOP. It starts only after
P3B checkpoints and P3C primary results are completed and locked. It does not
change, tune, or replace the P3 primary analysis.

P2F remains read-only. The runner may access only P2F test identifiers, GRCh38
intervals, labels, and sequence representations needed for the three frozen
checks. It must not read P2F predictions, attributions, model outputs, or report
statistics. Raw P2F or P3 sequences must not be written to audit outputs.

## P3 training denominator

The primary P3-training denominator is the unique union of development rows used
as `train` in at least one of the three frozen folds. Fold-specific denominators
are also reported. A trigger is conservative: it fires when either the union or
any frozen fold exceeds its threshold.

## Three checks

1. Coordinate overlap: at least one base-pair overlap between half-open GRCh38
   intervals on the same chromosome.
2. Exact sequence overlap: equality of SHA-256 hashes of
   `min(S, reverse_complement(S))`.
3. High similarity: RC-aware canonical 15-mer bottom-hash MinHash screening,
   followed by exact confirmation at at least 90% sequence identity and at least
   90% coverage of both 256-bp sequences. MinHash is only a candidate screen; a
   pair is never called high-similarity without alignment confirmation.

The frozen MinHash operationalization is: seed 20260823, sketch size 128, at
least two shared sketch hashes, and exclusion of P2F sketch hashes occurring in
more than 256 P2F test sequences. These implementation details are frozen before
the first P2F test-sequence extraction.

## Decisions

- Exact overlap above 1% of P3 train: remove affected P3 training rows and rerun
  the core P3 analysis as a sensitivity analysis.
- Coordinate overlap above 5% of P3 train: report shared-locus and P3-specific
  strata. Do not delete biologically conserved loci solely for coordinate overlap.
- Any confirmed 90%/90% near-duplicate finding: remove the affected P3 training
  rows and rerun the core P3 analysis as a sensitivity analysis.

All fractions are reported against both P3 train and P2F test. Sensitivity
results never replace the locked primary P3 result. Cross-stage overlap is not
automatically called leakage because P2 and P3 models were fitted independently.

## Outputs

The audit writes input hashes, aggregate summaries, and pair tables containing
only identifiers, coordinates, hashes, and similarity statistics. It writes no
raw sequences, predictions, or attributions. Once `completion.json` exists, the
same output directory cannot be executed again.

