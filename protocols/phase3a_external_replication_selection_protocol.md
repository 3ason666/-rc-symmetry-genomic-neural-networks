# Phase 3A external-replication selection and freeze protocol

Protocol revision: `phase3a_external_replication_selection_v2`  
Freeze date: 2026-08-23  
Status at freeze: external resources selected; no Phase 3 model output observed

## Purpose

Phase 3 tests whether the Phase 2 conclusions reproduce outside the K562 GATA1
system. Phase 3A freezes the external tasks, raw resources, sample construction,
leakage controls, models, primary endpoints, and decision rules before external
model training or prediction.

Phase 2F is completed and locked. Its K562 test rows, predictions, attributions,
and checkpoints are not development inputs for Phase 3. Phase 3 does not reopen,
rerun, improve, or reinterpret the Phase 2F test.

## Frozen confirmatory tasks

### Task P3-GATA1-Fetal

- Question: does the same GATA1 conclusion reproduce after replacing the K562
  leukemia cell line with primary human fetal erythroblasts?
- Positive assay: ENCODE `ENCSR000EXR`, GATA1 TF ChIP-seq in human embryo
  (16-19 weeks) erythroblasts, two isogenic replicates.
- Primary peak file: `ENCFF802VPT`, released ENCODE4 v1.6.1 GRCh38 IDR-thresholded
  narrowPeak file.
- Peak sensitivity file: `ENCFF305RPQ`, conservative IDR-thresholded narrowPeak.
- Accessibility universe: the union of FDR 0.01 DNase peaks from two independent
  fetal erythroblast experiments, `ENCSR059YWJ/ENCFF253NOJ` and
  `ENCSR514POY/ENCFF404PZM`.
- Important limitation: the two DNase experiments are independent, unreplicated
  fetal erythroblast profiles and are not the same physical biosample as the
  ChIP-seq experiment. They define a cell-state-matched accessibility universe;
  they are not treated as ChIP biological replicates.
- Biology resource: the already frozen JASPAR 2026 GATA1 `MA0035.5` PWM.

### Task P3-CTCF-GM12878

- Question: do the conclusions reproduce for a different transcription factor
  and a different cell line?
- Positive assay: ENCODE `ENCSR000DZN`, CTCF TF ChIP-seq in human GM12878, two
  isogenic replicates.
- Primary peak file: `ENCFF138REW`, released ENCODE4 v1.5.1 GRCh38 IDR-thresholded
  narrowPeak file.
- Peak sensitivity file: `ENCFF796WRU`, conservative IDR-thresholded narrowPeak.
- Accessibility universe: `ENCSR637XSC/ENCFF687HBT`, released ENCODE4 v1.9.1
  GRCh38 replicated ATAC-seq peaks from three isogenic replicates.
- Biology resource: JASPAR 2026 CTCF `MA0139.2` PWM, to be downloaded and
  checksummed before model training.

## Confirmatory hierarchy

The tasks have different evidential roles and are not combined by majority vote:

1. `P3-GATA1-Fetal` is the primary confirmatory task: same TF, different cellular environment;
2. `P3-CTCF-GM12878` is supportive cross-TF generalization evidence: different TF and different cellular environment.

CTCF support strengthens external validity. CTCF non-support does not invalidate
the Phase 2 K562 conclusion and is not automatically labeled a TF functional
boundary. Boundary language is allowed only if the data-quality gates pass and a
performance/power-precision manifest frozen before test unsealing establishes
that the negative result is interpretable. Otherwise it is reported as
inconclusive.

Mouse erythroblast GATA1 remains an optional cross-species stress test and is not
part of the confirmatory Phase 3 family. GEO datasets requiring hg18/hg19 lift-over
remain documented alternatives, not silent replacements.

## Frozen sample construction

- Assembly: GRCh38 no-alt primary chromosomes chr1-chr22.
- Sequence length: 256 bp.
- Positive center: the valid narrowPeak summit.
- A primary positive must overlap the task's frozen accessibility universe.
- If positive windows overlap, retain deterministically by decreasing peak signal,
  then chromosome, start, end, and source row.
- Exclude windows outside chromosome bounds, containing non-ACGT bases,
  overlapping the frozen ENCODE GRCh38 blacklist, or using non-primary contigs.
- Negative candidates are centered on accessible peaks, must not overlap any
  primary or conservative target peak, and must not fall within 128 bp of a target
  peak boundary. This means "not called as a target peak"; it is not proof of zero
  ChIP signal.
- Match negatives 1:1 without replacement on chromosome, 0.02-wide GC bin,
  accessibility-signal quantile, and sequence length.
- For the fetal GATA1 task, accessibility is the maximum peak signal among either
  frozen DNase resource. For the CTCF task it is the overlapping replicated ATAC
  peak signal.
- Repeat fetal-GATA1 sensitivity analysis with each DNase source alone and with
  random genomic negatives. A sensitivity set that misses its declared sample gate
  is reported as underpowered rather than repaired after results.

The large accessibility bigWig files are not required. Peak-level signal already
present in the frozen BED records is used for matching. This reduces transfer and
storage without changing the class definition.

## Frozen data-quality gates

- Retain at least 1,500 primary positives per task after all coordinate, sequence,
  accessibility, blacklist, and duplicate filters.
- Preserve exact 1:1 class balance. Require at least 1,000 examples per class in
  train and at least 200 per class in validation and permanent test.
- The H1b prediction-consistent test subgroup must contain at least 100 samples.
- At least 90% of eligible positives must obtain an accessible matched negative
  for the confirmatory analysis. Rates from 80% to below 90% are sensitivity-only;
  below 80% fails the task gate.
- Match GC with 0.02 bins and require positive-versus-negative KS D at most 0.10
  within every split.
- Compare accessibility within task after `log2(signal + 1)` and require KS D at
  most 0.10 within every split.
- The permanent test must contain at least three chromosomes, and no single
  chromosome may contribute more than 50% of its retained samples.

These attainable thresholds replace any generic 5,000-train/1,000-test proposal:
the fetal GATA1 primary peak file has only 2,499 autosomal rows before downstream
filtering, so augmentation cannot be counted as new independent samples.

## Split and leakage policy

Whole chromosomes are the split unit. Exact chromosome lists are assigned only
after coordinate and sequence QC establishes retained counts, and before any model
is trained. The deterministic allocation targets 70% train, 15% validation, and
15% one-time test by positive count, minimizes class-fraction deviations, and uses
lexicographic tie breaking.

Each task has one permanent test set shared by all folds and seeds. Three
development folds rotate train and validation chromosomes only. The test set is
forbidden for early stopping, checkpoint/epoch selection, threshold selection,
hyperparameter selection, and code debugging. It is unsealed once after all nine
checkpoints, cohorts, endpoints, analysis code, and checksums are frozen.

After assignment:

- reject base-pair overlap across splits;
- reject reuse of the same source peak across splits;
- reject identical sequences across splits;
- reject canonical-key collisions where
  `canonical_key = min(S, reverse_complement(S))`;
- use RC-aware 90% sequence identity with at least 90% coverage of both 256-bp
  sequences as the primary near-duplicate rule; report 80%/80% as sensitivity;
- if a near-duplicate cluster crosses splits, preserve the higher-protection split
  using the frozen priority `test > validation > train` and remove the
  lower-priority member deterministically;
- report near-duplicate and low-complexity rates without silently deleting failures.

No Phase 3 task may be compared against or deduplicated by reading Phase 2F test
sequences during P3 development. Independence is established by dataset provenance
and a separate Phase 3 split, not by reopening the locked test.

## Frozen model and analysis family

- Models: `CNN-Raw`, `CNN-Aug`, and `CNN-RCPS` only.
- Architecture and training defaults are inherited unchanged from Phase 2D.
- Seeds: 42, 123, and 2026.
- Training: one permanent test plus three train/validation chromosome folds per
  task; three seeds per fold; maximum 30 epochs; patience 5.
- No Transformer optimization or new architecture search.
- Primary attribution: Exact ISM in logit space.
- Gradient attribution may be reported only as a prespecified sensitivity analysis.
- Biological checks: task-specific PWM localization, circular-shift localization
  control, motif disruption, and equal-mutation-count matched flank control.

## Frozen primary endpoints and decisions

Each task is first reported separately. Fold and seed fits are not independent
biological replications and are not used as a 2-of-3 success vote. The primary
task-level estimate uses hierarchical paired bootstrap over model fit and sample.
Direction agreement in at least two of the three development folds is a robustness
descriptor only.

- H1a external replication: lower 95% confidence limit for CNN-Raw mean
  `abs(p(S)-p(RC(S)))` is greater than 0.01.
- H1b external replication: among CNN-Raw samples with prediction difference at
  most 0.01, the lower 95% confidence limit for Exact ISM normalized L1 is greater
  than 0.05.
- H2 external replication: upper 95% confidence limit for the paired
  `CNN-Aug - CNN-Raw` prediction-difference contrast is below zero.
- H3a external replication: lower 95% confidence limit for CNN-Aug Exact ISM
  normalized L1 is greater than 0.05.
- H3b numerical gate: CNN-RCPS maximum prediction difference is at most 1e-6 and
  maximum RC-aligned Exact ISM matrix residual is at most 1e-5.
- H4 is tested as conceptual separability, not as a claim that every flagged sample
  is biologically wrong. Its task-specific diagnostic is frozen before test unsealing.

Full fetal-GATA1 replication requires H1a, H1b, H2, and H3a to pass their frozen
confidence-limit decisions and H3b to pass both numerical gates. Otherwise report
partial replication hypothesis by hypothesis. H4 remains a separate mechanistic
diagnostic. Transformer H3c is exploratory and excluded from confirmatory success.

Use 10,000 hierarchical paired bootstrap replicates over model fit and sample.
Report task-specific estimates before any cross-task summary. Preserve null,
unfavorable, failed, NaN, and constant-attribution results.

## Post-primary P2F/P3 overlap audit

P2F remains sealed throughout P3 development and primary analysis. After P3
checkpoints, code, manifests, and primary results are frozen, an isolated read-only
audit may access only the minimum P2F coordinates and sequence representations
needed for three distinct checks:

- GRCh38 intervals for coordinate overlap;
- canonical-sequence SHA-256 for exact sequence overlap;
- 15-mer MinHash sketches for high-similarity screening.

Report overlap as both a fraction of P3 train and a fraction of P2F test. If exact
overlap exceeds 1% of P3 train, remove those P3 training rows and rerun the core P3
analysis as sensitivity. If coordinate overlap exceeds 5% of P3 train, report
shared-locus and P3-specific strata rather than deleting biologically conserved
sites. High-similarity findings trigger a 90%/90% near-duplicate sensitivity rerun.
These analyses never replace the original primary result and cross-stage overlap is
not automatically labeled training leakage because Phase 2 and Phase 3 models are
fit independently.

## Gates before training

Training remains blocked until all selected files are downloaded, MD5 verified,
non-empty, GRCh38, and QC-counted; both matched datasets pass the frozen sample,
balance, near-duplicate, leakage, and fixed-test gates; chromosome splits are
frozen; the CTCF PWM is checksummed; the near-duplicate implementation and CTCF
performance/power-precision interpretation manifest are frozen; and a Phase 3B
execution manifest is written. No gate may be relaxed after model output is observed.
