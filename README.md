# Prediction Consistency Is Not Explanation Consistency

This repository contains the code, frozen configurations, selected machine-readable outputs, and figure-generation inputs for the study **“Prediction Consistency Is Not Explanation Consistency: Reverse-Complement Symmetry in Genomic Neural Networks.”** The publication repository is hosted at [3ason666/-rc-symmetry-genomic-neural-networks](https://github.com/3ason666/-rc-symmetry-genomic-neural-networks). A release tag and archival DOI will be added when issued.

## Project overview

The study separates three questions that are often conflated in genomic machine learning:

1. Does a model make similar predictions for a DNA sequence, S, and its reverse complement, RC(S)?
2. Does it assign similar RC-aligned nucleotide attributions to S and RC(S)?
3. Are the learned features compatible with motif-based biological plausibility checks?

The principal comparison is among a conventional CNN trained on the supplied orientation (CNN-Raw), the same architecture trained with stochastic reverse-complement augmentation (CNN-Aug), and a reverse-complement parameter-sharing CNN (CNN-RCPS). Transformer/H3c analyses are exploratory and are not part of the confirmatory claim set.

## Main scientific question

The central question is whether prediction consistency is sufficient evidence for explanation consistency. The frozen analyses show that these properties must be measured separately: near-identical S/RC(S) predictions can coexist with appreciable RC-aligned Exact ISM asymmetry, whereas the RCPS architecture provides numerical symmetry by construction.

## Repository structure

- `configs/`: executed and historical YAML configurations.
- `protocols/`: frozen analysis and test-unsealing protocols.
- `src/`: model, sequence, attribution, metric, matching, and training implementations.
- `scripts/`: dataset construction, quality control, training, evaluation, and audit entry points.
- `metadata/`: dataset, accession, checkpoint, and freeze manifests.
- `results/`: selected frozen summary tables, hypothesis decisions, overlap audit, and sensitivity results.
- `data/phase2/processed/`: the K562 table required only to reconstruct Figure 2A.
- `paper_figures/`: publication figure script, the minimum frozen plotting inputs, captions, final outputs, and validation records.
- `supplementary/`: Supplementary Information and release notes.

Large reference genomes, source BED files, model checkpoints, caches, and nonessential sample-level intermediate outputs are deliberately excluded.

## Data sources and accessions

All genomic coordinates use GRCh38. The principal sources are:

| Task | Role | Experiment / file |
|---|---|---|
| K562 GATA1 | ChIP-seq positives | ENCSR000EFT / ENCFF148JKK |
| K562 GATA1 | conservative-IDR sensitivity | ENCFF875JHB |
| K562 | ATAC accessible negative pool | ENCSR868FGK / ENCFF333TAT |
| Exclusion regions | ENCODE blacklist | ENCSR636HFF / ENCFF356LFX |
| Reference | GRCh38 no-alt | ENCSR425FOI / GCA_000001405.15 |
| Fetal erythroid GATA1 | ChIP-seq positives | ENCSR000EXR / ENCFF802VPT |
| Fetal erythroid GATA1 | conservative sensitivity | ENCFF305RPQ |
| Fetal erythroid accessibility | DNase | ENCSR059YWJ / ENCFF253NOJ; ENCSR514POY / ENCFF404PZM |
| GM12878 CTCF | ChIP-seq positives | ENCSR000DZN / ENCFF138REW |
| GM12878 CTCF | conservative sensitivity | ENCFF796WRU |
| GM12878 accessibility | ATAC | ENCSR637XSC / ENCFF687HBT |

The motif resources are JASPAR CORE 2026 MA0035.5 (GATA1) and MA0139.2 (CTCF). URLs, retrieval metadata, and checksums are retained under `metadata/accession_registry/` and the dataset manifests.

## Environment

Historical run metadata recorded Python 3.12.13, PyTorch 2.13.0+cpu, CPU execution, Captum 0.8.0, and RapidFuzz 3.14.1. Other Python dependencies were not historically pinned. See `environment_notes.md` for the reproducibility boundary and `requirements.txt` for the reconstructed environment specification.

## Reproducing dataset construction

Dataset construction requires downloading public ENCODE resources and the GRCh38 reference; those large files are not bundled. The relevant order is:

1. Review `configs/phase2_dataset_manifest.yaml` and `configs/phase3a_external_replication.yaml`.
2. Download sources using the phase-specific download scripts.
3. Run BED/reference QC.
4. Build candidate windows and perform matching.
5. Run the split, RC-aware deduplication, leakage, and near-duplicate gates.
6. Compare generated hashes and counts with `metadata/dataset_manifests/`.

The scripts and manifests make these steps traceable, but an end-to-end clean-machine reconstruction was not run during this submission audit. Network retrieval and storage of the multi-gigabyte reference are therefore a documented manual prerequisite.

## Reproducing frozen analyses

The executed configs, protocols, source code, random seeds, completion locks, hypothesis tables, and checksums are included. Full model-level recomputation is not self-contained because trained checkpoints and most large sample-level prediction tables are intentionally excluded. The release must not be described as a one-command retraining bundle.

The included results can be inspected without recomputation. If checkpoints are archived separately, their expected file identities are listed in `metadata/frozen_manifests/`. Re-running any analysis should use the frozen configs and must not change sample selection, thresholds, seeds, or hypothesis rules.

## Regenerating Figures 1–5

The main figures can be rebuilt from the bundled frozen plotting inputs without checkpoints or retraining:

```bash
python paper_figures/scripts/generate_main_figures.py
```

The script writes SVG, PDF, and 600-dpi PNG files under `paper_figures/main/`, plus validation and provenance records. The submission audit successfully executed this command. PNG output was bitwise identical to the pre-run figures; vector output retained the same quantitative content but was not byte-identical because PDF/SVG serialization metadata changed.

## Frozen analysis policy

The permanent test partitions were fixed before model selection and were excluded from training, validation, early stopping, hyperparameter selection, and debugging. Frozen values, confidence intervals, sample selections, thresholds, hypothesis decisions, and Figure 1–5 quantitative content must not be changed after release.

Phase 3C is the primary external analysis. Phase 3E is a training-only near-duplicate sensitivity analysis and does not replace Phase 3C.

- Fetal erythroid GATA1 H1b is **NOT ESTIMABLE** because the eligible prediction-consistent pool is 76, below the frozen target/minimum of 100.
- GM12878 CTCF H2 is supported in the primary Phase 3C analysis. In Phase 3E, the point estimate remains negative but the 95% confidence interval crosses zero; this is an explicit robustness boundary, not a reclassification of the primary result.
- Transformer/H3c is exploratory and was not retested as a confirmatory external hypothesis.

## Citation

Citation metadata, including the author, ORCID identifier, and repository URL, are provided in `CITATION.cff`. The release tag and archival DOI will be added after the archival release is created.

## AI-use disclosure

OpenAI Codex was used for code and documentation assistance, audit support, and preparation of submission materials. Scientific design choices, frozen decisions, source verification, interpretation, and final responsibility remain with the authors. The final disclosure should be reconciled with the journal’s current policy before submission.

## License

The original contents of this repository are released under the MIT License; see `LICENSE`. Third-party resources remain subject to their original terms and are not relicensed by this repository.
