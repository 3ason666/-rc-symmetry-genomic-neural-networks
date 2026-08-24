# Supplementary Information

**Prediction Consistency Is Not Explanation Consistency: Reverse-Complement Symmetry in Genomic Neural Networks**

This Supplementary Information was assembled from frozen configurations, machine-readable outputs, executed protocols, source code, and the verified Methods extraction. It introduces no new statistical inference and does not modify any frozen value, confidence interval, sample selection, threshold, or hypothesis decision.

# Supplementary Methods

## S1. Synthetic task construction

The baseline synthetic task used random 100-bp DNA sequences. Positive sequences contained the motif `TGATTTAT` at a uniformly sampled position and in a randomly sampled forward or reverse-complement (RC) orientation. Negative sequences were rejected if they contained either motif orientation. The frozen generation seed was 31415, with 8,000 training, 1,000 validation, and 1,000 test sequences.

Three 160-bp Phase 1.5 tasks tested more difficult or failure-prone settings. The hard multi-motif task used `TGATTTAT` as the causal consensus together with three one-base-related decoys (`TGACTTAT`, `TGATCTAT`, and `TAATTTAT`), variable motif count and strength, and 3% label noise in training only. It contained 5,000/1,000/1,200 train/validation/test sequences and used seed 915150. The shortcut-shift task inserted an edge-localized `GGGGGG` feature within a 24-bp edge region. Development shortcut probabilities were 0.95 for positives and 0.05 for negatives, whereas both classes used probability 0.50 at test; class-specific GC targets similarly shifted from 0.58/0.42 in development to 0.50/0.50 at test. This task contained 4,000/800/1,000 sequences and used seed 915151. The motif-grammar task combined `TGATTTAT` and `CAGGTG`; positives used the frozen RC-invariant convergent grammar with gap 2-6, whereas negatives used gap 18-32 or invalid orientation. It contained 5,000/1,000/1,200 sequences and used seed 915152. Generators rejected RC-canonical duplicates across splits.

## S2. Dataset construction and accession details

All genomic tasks used GRCh38 and 256-bp autosomal windows. K562 GATA1 positives came from ENCODE ENCSR000EFT/ENCFF148JKK. A positive was centered on a valid ChIP-seq summit, required overlap with K562 ATAC resource ENCSR868FGK/ENCFF333TAT, and was excluded for blacklist overlap, non-ACGT sequence, invalid/out-of-bounds coordinates, non-primary chromosome, or overlap with a higher-priority retained positive. Negative candidates were centered on K562 ATAC summits and excluded if they overlapped a GATA1 interval expanded by 128 bp, the blacklist ENCSR636HFF/ENCFF356LFX, an invalid sequence, or a chromosome boundary. Positives and negatives were matched 1:1 without replacement by chromosome, 0.02-wide GC bin, and chromosome-specific accessibility decile, using matching seed 20260821.

Fetal erythroid GATA1 positives came from ENCSR000EXR/ENCFF802VPT, with ENCFF305RPQ reserved for conservative sensitivity. Accessibility was the union of fetal erythroblast DNase files ENCSR059YWJ/ENCFF253NOJ and ENCSR514POY/ENCFF404PZM. DNase and ChIP represented a similar cellular state but were not the same physical biosample. GM12878 CTCF positives came from ENCSR000DZN/ENCFF138REW, with ENCFF796WRU reserved for conservative sensitivity; accessible negatives came from GM12878 ATAC ENCSR637XSC/ENCFF687HBT.

External candidates used the same boundary, blacklist, alphabet, and 128-bp exclusion rules. Exact and RC-equivalent sequences were deduplicated using `min(S,RC(S))`. Matching was 1:1 without replacement by chromosome, 0.02 GC bin, and accessibility decile, minimizing absolute log2(accessibility+1) difference subject to a maximum difference of 1.0. The configured base seed was 20260823. The implementation added the zero-based task index, yielding effective matching seeds 20260823 for fetal GATA1 and 20260824 for CTCF.

## S3. Full chromosome partition tables

K562 used train chromosomes chr1, chr4, chr5, chr6, chr7, chr9, chr10, chr12, chr13, chr15, chr16, chr17, chr19, chr20, and chr21; reserved validation chromosomes chr2, chr11, and chr22; and permanent test chromosomes chr3, chr8, chr14, and chr18. Phase 2D used only the original train partition. Its three inner validation folds were chr1/13/16/17/21, chr4/6/12/15/20, and chr5/7/9/10/19, respectively.

The fetal GATA1 permanent test used chr11, chr12, and chr14. Its development-fold validation chromosomes were chr1/16, chr2/4/16, and chr2/9/16. The GM12878 CTCF permanent test used chr11, chr16, chr19, and chr22. Its development-fold validation chromosomes were chr9/15/17/20, chr5/13/14/15, and chr8/9/12/21. A task’s permanent test was shared across folds and seeds and was not used for early stopping or tuning.

## S4. CNN-Raw, CNN-Aug, and CNN-RCPS architecture

Sequences were float32 one-hot tensors of shape batch x 4 x length with A,C,G,T channel order. CNN-Raw and CNN-Aug used Conv1d(4,32,kernel=15,stride=1,same padding), ReLU, Conv1d(32,24,kernel=7,stride=1,same padding), ReLU, adaptive global maximum pooling, and Linear(24,1). No normalization or dropout was present. The architecture returned one logit and contained 7,377 trainable parameters for the 256-bp genomic runs.

CNN-RCPS learned only one half of each convolutional filter pair. At the input layer, the partner response was obtained by reversing position and permuting A,C,G,T channels to T,G,C,A. At deeper layers, the RC transform reversed position and swapped the first and second feature-channel halves. Original- and transformed-orientation responses were concatenated. After the two RCPS convolutions, ReLU, and global max pooling, paired pooled features were averaged before Linear(12,1). Odd kernels were enforced for exact same-padding equivariance. The 256-bp model contained 3,689 trainable parameters. Learned half-filter weights used Kaiming-uniform initialization with `a=sqrt(5)` and zero half-biases; other layers used the recorded PyTorch constructor defaults.

## S5. Training and deterministic settings

Training minimized `BCEWithLogitsLoss` with Adam, learning rate 0.001, weight decay 0.0001, batch size 128, at most 30 epochs, and patience 5. No scheduler was used. The checkpoint with minimum validation loss was retained, with a minimum improvement of 1e-8. Model seeds were 42, 123, and 2026, giving three seeds per fold and nine fits per architecture for each three-fold genomic analysis.

CNN-Aug replaced a training example by RC(S) with probability 0.5 each time the dataset item was retrieved; labels were unchanged. Python `random`, NumPy, and PyTorch were seeded per fit. PyTorch deterministic algorithms were enabled, the DataLoader shuffle generator used the fit seed, and worker count was zero. Training and inference used CPU. The K562 and external workflows used the same model/training hyperparameters; their data and chromosome fold sizes differed.

## S6. Attribution methods

Exact in silico mutagenesis (ISM) was the confirmatory attribution method. Integrated Gradients, DeepLIFT, and GradientSHAP were secondary. Captum 0.8.0 implemented the gradient-based methods. Integrated Gradients used a zero baseline, 32 Gauss-Legendre steps, and a capped internal batch. DeepLIFT used the zero baseline. GradientSHAP used an RC-compatible two-point baseline distribution containing an all-zero tensor and a uniform 0.25 tensor, with 16 samples and zero added noise.

Phase 2D deterministically selected 40 positive and 40 negative sequences within each chromosome holdout fold, producing 80 sequences per fold and 240 sequence-fold entries. The K562 permanent-test attribution cohort used 10 samples per test chromosome per label, ranked by SHA-256 with salt `phase2f_v1`, producing 80 unique sequences evaluated across 27 checkpoints. For one architecture this yielded 720 sample-run observations.

## S7. Exact ISM formula and RC alignment

For sequence x and reference logit f(x), each position was changed to each of its three alternative nucleotides. For mutant x(i->b), the matrix entry was

`A[i,b] = f(x) - f(x(i->b))`.

The observed-nucleotide entry was zero, giving an L x 4 matrix in A,C,G,T order. Signed position attribution was the mean of the three alternative-base effects; absolute position attribution was the mean absolute effect. ISM used logits, not probabilities.

RC matrices were aligned to S by reversing the position axis and permuting channels with indices [3,2,1,0]. For forward absolute position attribution a and aligned-RC attribution b, explanation asymmetry was

`normalized L1 = sum_i |a_i-b_i| / (sum_i |a_i| + sum_i |b_i|)`.

No epsilon was added. A zero denominator produced a missing value. Figure 3B was selected among frozen CNN-Raw observations with prediction difference <=0.01. Of 63 eligible observations, the one closest to the eligible median normalized L1 was selected, with ties ordered by sample ID, fold, and seed. This fixed ENCFF148JKK:12229, fold_a, seed 123.

## S8. Motif-based biological plausibility

JASPAR CORE 2026 matrices MA0035.5 (GATA1) and MA0139.2 (CTCF) were converted from counts to probabilities with pseudocount 0.5 and to log2 odds against a uniform 0.25 background. A project scanner evaluated both strands. Relative PWM score was normalized by the theoretical minimum and maximum and clipped to [0,1]. The best hit maximized raw score, with lower start and then forward strand as tie-breaks. A strong hit required relative score >=0.80; 0.85 and 0.90 were sensitivity thresholds.

Motif attribution mass was `sum_(i in motif)|m_i| / sum_i|m_i|`. Top-7 recall was the fraction of motif positions among the seven largest absolute attribution positions. Circular enrichment used 64 nonzero circular shifts.

For in silico perturbation, every best-hit motif position was changed to the lowest-scoring base in the strand-oriented PWM column, or the next-lowest base when necessary to force a change. Motif logit drop was original minus mutated logit. The flank control was a non-overlapping same-width interval nearest the position mirrored around sequence center, with lower coordinate as tie-break. It received the same number and relative offsets of deterministic cyclic A->C->G->T->A changes. Specificity was motif logit drop minus flank logit drop. These analyses measure motif-based biological plausibility and are not wet-laboratory validation.

## S9. Bootstrap and multiple testing

Confirmatory intervals used 10,000 hierarchical paired bootstrap replicates. Contrasts were first paired by fold, seed/model fit, and sample. Each replicate sampled folds with replacement, sampled seed-specific fits within each selected fold, and sampled sequences within each fold-seed cell. Cell means were averaged equally within fold and then across folds. The observed estimate was the equal-weight mean of observed fold-seed cell means. The 95% confidence interval used the 2.5th and 97.5th percentiles. Two-sided bootstrap P values used twice the smaller plus-one-corrected tail probability.

Benjamini-Hochberg adjustment used stable ordering and reverse cumulative minima within each generated confirmatory table. For external Phase 3C/3E, the five bootstrap rows H1a, H1b, H2, H3a, and H4 were corrected separately within each task; a non-estimable H1b P value was set to 1 for correction. H3b used deterministic numerical tolerances rather than bootstrap significance. Fold-seed fits were repeated technical/model-fitting units, not independent biological replicates.

## S10. External replication protocol

Fetal erythroid GATA1 was the primary same-factor/different-context setting. GM12878 CTCF was a parallel supportive different-factor/different-context setting. Phase 3B trained 27 checkpoints per task from development rows only. Phase 3C performed one frozen evaluation of each permanent test. Each checkpoint predicted S and RC(S), and architecture ensembles averaged nine fold-seed fits.

External H1b used CNN-Raw ensemble probabilities averaged across nine fits. A unique sequence was eligible when the absolute difference between ensemble p(S) and p(RC(S)) was <=0.01. The target/minimum cohort was 100 SHA-ranked unique sequences with round-robin chromosome-label coverage. If fewer than 100 were eligible, H1b was NOT ESTIMABLE without threshold relaxation. Fetal GATA1 had 76 eligible sequences and remained NOT ESTIMABLE.

## S11. Cross-stage overlap audit

Phase 3D accessed the K562 Phase 2F test sequences in isolated read-only mode and did not access their predictions or attributions. It compared the union of sequences used for Phase 3 training in any fold against the frozen K562 test. Coordinate overlap used GRCh38 half-open intervals and any overlap >=1 bp. Exact overlap used SHA-256 of `min(S,RC(S))`.

High-similarity screening used RC-canonical 15-mers. Each k-mer was 2-bit encoded and transformed with SplitMix64 after XOR with seed 20260823. Per-sequence hashes were deduplicated and represented by the 128 smallest values. Hashes occurring in more than 256 K562 test sequences were excluded; candidates required at least two shared hashes. Both sequence orientations and every shift allowing at least 90% overlap were examined with RapidFuzz Levenshtein distance. Identity was `1-distance/max(aligned lengths)` and coverage was `overlap/max(full lengths)`. Confirmed near duplicates required identity >=0.90 and coverage >=0.90.

## S12. Near-duplicate sensitivity

Phase 3E was a sensitivity analysis only. A confirmed near-duplicate Phase 3 sequence was removed only if it belonged to the current fold’s training partition. Validation and permanent-test rows were unchanged. Architectures, hyperparameters, seeds, folds, test populations, attribution cohorts, thresholds, and statistical procedures were reused. Fifty-four sensitivity fits were completed. Phase 3E did not replace Phase 3C and no additional CTCF H2 tuning was performed.

## S13. Exploratory Transformer analysis

Transformer/H3c remained exploratory. The C4 model divided DNA into non-overlapping 4-bp patches using Conv1d patch embedding and used width 32, four attention heads, three pre-norm self-attention blocks, feed-forward width 64, dropout 0.1, mean token pooling, and a linear output. Variants used no positional term, learned absolute position, or learned signed relative-position attention bias. RC augmentation used the same stochastic 0.5 replacement as CNN-Aug.

The three-seed validation comparison was inconclusive under its performance-matching gate. A subsequent train-only remediation was intentionally terminated after the project reclassified CNN-versus-Transformer comparison as exploratory. Missing fold-seed cells were not imputed, no test predictions were generated, and H3c was excluded from Phase 2D/2F and Phase 3 confirmatory decisions.

# Supplementary Results

## S1. Baseline synthetic results

The simple implanted-motif task was nearly saturated. Across three seeds, mean S/RC prediction difference was 0.000769 for CNN-Raw, 0.000550 for CNN-Aug, and 0.000280 for CNN-Pair. Mean absolute attribution Pearson similarity was 0.99798, 0.99850, and 0.99827, respectively. These ceiling-level results motivated the harder Phase 1.5 tasks.

## S2. Hard synthetic results

All principal models retained high test discrimination. Mean test AUROC was 0.94721 for CNN-Raw, 0.94924 for CNN-Aug, 0.95239 for CNN-Pair, 0.95012 for post-hoc conjoining, and 0.94833 for CNN-RCPS. Relative to CNN-Raw, CNN-Aug reduced mean prediction difference by -0.01096 (95% CI -0.01328 to -0.00853). Mean causal attribution mass was 0.66685 for CNN-Raw and 0.69080 for CNN-Aug; the paired difference was 0.02395 (95% CI 0.01661 to 0.03146). CNN-RCPS achieved near-unit RC-aligned attribution Pearson similarity, while its causal-mass contrast with CNN-Raw was uncertain (0.02958; 95% CI -0.03288 to 0.07803).

## S3. Shortcut-shift results

The shortcut-shift test intentionally broke the development correlation. Validation AUROC was approximately 0.989-0.995, while test AUROC was approximately 0.485-0.492 across models, demonstrating severe shortcut-induced distribution-shift failure. CNN-Raw mean shortcut attribution mass was 0.54938 versus 0.31623 for CNN-Aug and 0.30433 for CNN-RCPS. RC interventions strongly reduced prediction asymmetry, but did not restore test discrimination, illustrating that RC consistency alone does not guarantee task-valid generalization.

## S4. Motif-grammar results

Mean test AUROC was 0.97182 for CNN-Raw, 0.97197 for CNN-Aug, 0.98369 for CNN-Pair, 0.97426 for post-hoc conjoining, and 0.97145 for CNN-RCPS. CNN-Aug minus CNN-Raw prediction difference was -0.18481 (95% CI -0.21546 to -0.16248). CNN-Aug increased causal attribution mass by 0.07864 (95% CI 0.02106 to 0.14235). CNN-RCPS achieved numerical attribution symmetry and a causal-mass difference of 0.12615 (95% CI 0.03033 to 0.19414) relative to CNN-Raw in this controlled grammar task.

## S5. Secondary attribution metrics

In Phase 2D, mean RC-aligned normalized L1 across nine runs was 0.33853/0.31048/4.13e-9 for Exact ISM in CNN-Raw/CNN-Aug/CNN-RCPS. Corresponding secondary values were 0.45215/0.42887/6.68e-5 for Integrated Gradients, 0.46910/0.44702/0.000248 for DeepLIFT, and 0.42769/0.40794/0.09555 for GradientSHAP. The nonzero RCPS GradientSHAP residual was retained as a method/baseline diagnostic; it did not supersede Exact ISM as the frozen endpoint.

## S6. Motif threshold sensitivity

At strong-hit thresholds 0.80, 0.85, and 0.90, eligible sample-run counts per model were 324, 306, and 138. Mean motif mass increased with the stricter threshold. At 0.80, means were 0.18913, 0.19666, and 0.20239 for CNN-Raw, CNN-Aug, and CNN-RCPS. At 0.90, they were 0.21741, 0.22614, and 0.23212. Mean motif-minus-flank specificity remained positive for all models at all three thresholds. These descriptive sensitivity summaries did not establish a model-superiority ordering.

## S7. Conservative-IDR analyses

Using conservative GATA1 positives, mean AUROC across nine runs was 0.87926 for CNN-Raw, 0.88754 for CNN-Aug, and 0.89041 for CNN-RCPS. Mean prediction difference was 0.10059, 0.08567, and 0, respectively; mean flip rate was 0.11135, 0.09267, and 0. The analysis supported the same prediction-symmetry direction without replacing the primary optimal-IDR dataset.

## S8. Transformer exploratory results

The three-seed Phase 2C.2 comparison produced a Transformer-minus-CNN Exact ISM normalized-L1 estimate of -0.07646 (95% CI -0.08863 to -0.06521), but the frozen performance-matching criterion did not pass for every seed. The status was therefore inconclusive in multiseed validation. Phase 2C.3 ended by researcher-directed scope change after 28 completed train-only runs; it generated neither validation predictions nor test predictions. No causal or confirmatory H3c claim was made.

## S9. Phase 3 overlap audit details

For fetal GATA1, the union training population contained 3,316 rows. Coordinate overlap involved 275 unique Phase 3 rows (8.29%), exact RC-canonical overlap involved 1 row (0.030%), and confirmed high similarity involved 181 rows (5.46%). For CTCF, 55,712 union training rows included 882 coordinate-overlap rows (1.58%), 15 exact-overlap rows (0.0269%), and 668 high-similarity rows (1.20%). Exact-overlap thresholds did not trigger reruns; high similarity triggered the predefined Phase 3E sensitivity.

## S10. Near-duplicate sensitivity details

Fetal GATA1 H2 remained supported in Phase 3E: estimate -0.02396, 95% CI -0.03313 to -0.01561, compared with primary -0.02528, 95% CI -0.03604 to -0.01668. CTCF H2 was supported in primary Phase 3C at -0.01035 (95% CI -0.01600 to -0.00520), while the Phase 3E estimate was -0.00639 (95% CI -0.01248 to 0.00256). The latter interval crosses zero. This qualifies robustness but does not replace or retrospectively fail the primary analysis.

# Supplementary Tables

## Table S1. Datasets and accessions

| Dataset | Positive/binding source | Accessibility/negative source | Final N | Permanent test |
|---|---|---|---:|---|
| K562 GATA1 | ENCSR000EFT / ENCFF148JKK | ENCSR868FGK / ENCFF333TAT | 23,310 | 3,496 |
| Fetal erythroid GATA1 | ENCSR000EXR / ENCFF802VPT | ENCSR059YWJ / ENCFF253NOJ and ENCSR514POY / ENCFF404PZM | 4,064 | 610 |
| GM12878 CTCF | ENCSR000DZN / ENCFF138REW | ENCSR637XSC / ENCFF687HBT | 65,558 | 9,846 |

## Table S2. Chromosome splits

| Dataset/role | Chromosomes |
|---|---|
| K562 train | chr1,4,5,6,7,9,10,12,13,15,16,17,19,20,21 |
| K562 reserved validation | chr2,11,22 |
| K562 frozen test | chr3,8,14,18 |
| K562 Phase 2D fold A train / validation | chr4,5,6,7,9,10,12,15,19,20 / chr1,13,16,17,21 |
| K562 Phase 2D fold B train / validation | chr1,5,7,9,10,13,16,17,19,21 / chr4,6,12,15,20 |
| K562 Phase 2D fold C train / validation | chr1,4,6,12,13,15,16,17,20,21 / chr5,7,9,10,19 |
| Fetal GATA1 frozen test | chr11,12,14 |
| Fetal fold 1 train / validation | chr2,3,4,5,6,7,8,9,10,13,15,17,18,19,20,21,22 / chr1,16 |
| Fetal fold 2 train / validation | chr1,3,5,6,7,8,9,10,13,15,17,18,19,20,21,22 / chr2,4,16 |
| Fetal fold 3 train / validation | chr1,3,4,5,6,7,8,10,13,15,17,18,19,20,21,22 / chr2,9,16 |
| CTCF frozen test | chr11,16,19,22 |
| CTCF fold 1 train / validation | chr1,2,3,4,5,6,7,8,10,12,13,14,18,21 / chr9,15,17,20 |
| CTCF fold 2 train / validation | chr1,2,3,4,6,7,8,9,10,12,17,18,20,21 / chr5,13,14,15 |
| CTCF fold 3 train / validation | chr1,2,3,4,5,6,7,10,13,14,15,17,18,20 / chr8,9,12,21 |

## Table S3. Architecture

| Model | Convolutions | Pool/classifier | Parameters | RC mechanism |
|---|---|---|---:|---|
| CNN-Raw | 4->32, k15; 32->24, k7; ReLU | global max; 24->1 | 7,377 | none |
| CNN-Aug | same as Raw | same as Raw | 7,377 | stochastic data augmentation |
| CNN-RCPS | paired half-filter RCPS convolutions, k15/k7 | paired mean; 12->1 | 3,689 | exact parameter sharing and feature-half transform |

## Table S4. Training hyperparameters

| Parameter | Value |
|---|---|
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam |
| Learning rate / weight decay | 0.001 / 0.0001 |
| Batch size | 128 |
| Epoch limit / patience | 30 / 5 |
| Checkpoint criterion | minimum validation loss |
| Scheduler | none |
| Seeds | 42, 123, 2026 |
| CNN-Aug probability | 0.5 RC replacement per retrieval |

## Table S5. Frozen hypothesis thresholds

| Hypothesis | Frozen rule |
|---|---|
| H1a | CNN-Raw mean prediction-difference 95% CI lower bound >0.01 |
| H1b | eligible delta_p<=0.01; normalized-L1 95% CI lower bound >0.05; external target/minimum 100 |
| H2 | paired CNN-Aug-minus-CNN-Raw delta_p 95% CI upper bound <0 |
| H3a | CNN-Aug normalized-L1 95% CI lower bound >0.05 |
| H3b | maximum RCPS prediction error <=1e-6 and aligned Exact ISM matrix error <=1e-5 |
| H3c | exploratory; no confirmatory decision |
| H4 | RCPS diagnostic-failure-rate 95% CI lower bound >0; conceptual diagnostic only |

## Table S6. K562 frozen hypothesis results

| Analysis | Endpoint/contrast | Estimate | 95% CI | Frozen interpretation |
|---|---|---:|---|---|
| Phase 2D H1a | CNN-Raw delta_p | 0.10151 | 0.09245 to 0.11268 | supported |
| Phase 2D H1b | Raw Exact ISM L1, consistent subset | 0.33062 | 0.31190 to 0.35034 | supported |
| Phase 2D H2 | Aug minus Raw delta_p | -0.01470 | -0.02260 to -0.00386 | supported |
| Phase 2D H3a | Aug Exact ISM L1 | 0.31048 | 0.28315 to 0.33568 | supported |
| Phase 2D H3b | RCPS Exact ISM L1 | 4.13e-9 | 3.35e-9 to 5.02e-9 | numerical zero; tolerance audit passed |
| Phase 2F H1b | Raw Exact ISM L1, consistent subset | 0.31171 | 0.28686 to 0.33485 | reproduced |
| Phase 2F H2 | Aug minus Raw delta_p | -0.01515 | -0.02569 to -0.00294 | reproduced |

## Table S7. External primary hypothesis results

| Task | Hypothesis | Estimate | 95% CI / evidence | Status |
|---|---|---:|---|---|
| Fetal GATA1 | H1a | 0.10401 | 0.09357 to 0.11496 | reproduced |
| Fetal GATA1 | H1b | NE | eligible pool=76 | NOT ESTIMABLE |
| Fetal GATA1 | H2 | -0.02528 | -0.03604 to -0.01668 | reproduced |
| Fetal GATA1 | H3a | 0.32227 | 0.30614 to 0.34108 | reproduced |
| CTCF | H1a | 0.06971 | 0.06317 to 0.07665 | reproduced |
| CTCF | H1b | 0.29580 | 0.28091 to 0.31397 | reproduced |
| CTCF | H2 | -0.01035 | -0.01600 to -0.00520 | primary supported |
| CTCF | H3a | 0.28835 | 0.27550 to 0.29880 | reproduced |

## Table S8. Phase 3D overlap audit

| Task | Method | Unique Phase 3 rows | Fraction of union train | Triggered sensitivity? |
|---|---|---:|---:|---|
| Fetal GATA1 | coordinate | 275 | 0.08293 | stratification trigger |
| Fetal GATA1 | exact RC-canonical | 1 | 0.000302 | no |
| Fetal GATA1 | high similarity | 181 | 0.05458 | yes |
| CTCF | coordinate | 882 | 0.01583 | no |
| CTCF | exact RC-canonical | 15 | 0.000269 | no |
| CTCF | high similarity | 668 | 0.01199 | yes |

## Table S9. Phase 3E sensitivity analysis

| Task/Hypothesis | Primary estimate (95% CI) | Sensitivity estimate (95% CI) | Status |
|---|---|---|---|
| Fetal GATA1 H2 | -0.02528 (-0.03604, -0.01668) | -0.02396 (-0.03313, -0.01561) | stable support |
| CTCF H2 | -0.01035 (-0.01600, -0.00520) | -0.00639 (-0.01248, 0.00256) | sensitivity CI crosses zero; primary unchanged |

## Table S10. Software and reproducibility metadata

| Item | Recorded value/status |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.13.0+cpu |
| Captum | 0.8.0 |
| RapidFuzz | 3.14.1 |
| Device | CPU; CUDA/ROCm not used |
| Other Python package versions | exact historical versions not persisted |
| Historical OS/CPU/RAM | not authoritatively recorded |
| Figure regeneration | PASS; PNGs bitwise unchanged |
| Public GitHub/DOI | pending |
| Checkpoints in GitHub package | excluded; identities retained in manifests |

# Supplementary provenance

Primary sources for this document are `methods_metadata_extraction_v1.md`, `publication_methods_draft_v1.md`, frozen configs and protocols, Phase 1/1.5 summaries, Phase 2D-2F machine-readable outputs, Phase 3A-3E manifests and summaries, and the Main Figure validation records. Numerical precision shown here follows the source outputs; rounding is for presentation only.
