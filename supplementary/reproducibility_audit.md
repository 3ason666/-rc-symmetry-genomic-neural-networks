# Reproducibility audit

Audit date: 2026-08-24  
Scope: frozen artifacts and figure regeneration only; no model retraining or attribution recomputation.

## Summary

| Requirement | Status | Evidence and limitation |
|---|---|---|
| Frozen configs exist | PASS | Phase 1-3 YAML files are present and copied into `github_release/configs/` |
| Dataset accessions are traceable | PASS | ENCODE/JASPAR accessions, URLs, retrieval metadata, and QC are present in configs and metadata registries |
| Random seeds are traceable | PASS | Model seeds 42, 123, and 2026 and phase-specific selection/bootstrap seeds are recorded in configs and outputs; fetal matching uses 20260823 and CTCF uses the task-offset value 20260824 |
| Model architecture is traceable | PASS | `src/models.py`, input encoding, parameter counts in run summaries, and Methods metadata agree |
| Exact ISM definition is traceable | PASS | `src/interpret.py`, `src/dna_utils.py`, configs, and frozen protocols record logit-based 3L substitutions and RC channel/position alignment |
| Bootstrap procedure is traceable | PASS | Hierarchical paired resampling code, 10,000-replicate configs, percentile CI, and BH correction are present |
| Frozen manifests/checksums are traceable | PASS | Dataset, checkpoint, input-audit, completion, unseal, and Phase 3D-3E freeze records are present |
| Figure script executes | PASS | `generate_main_figures.py` completed with exit code 0 using frozen outputs |
| Figure 1-5 quantitative reconstruction | PASS | Build manifest preserved 720 Figure 3A observations, 80 sequences, the same Figure 3B sample, fetal H1b=NE, and CTCF sensitivity CI crossing zero |
| Bitwise reproducibility of all figure formats | PARTIAL | All 600-dpi PNG hashes were unchanged; regenerated SVG/PDF hashes changed because vector serialization metadata changed, while values and visible content were preserved |
| Dataset reconstruction on a clean machine | PARTIAL | Code and manifests exist, but the multi-gigabyte reference and public-source downloads are excluded and a new end-to-end download/build was not run |
| Full frozen model evaluation from public package alone | PARTIAL | Checkpoints and large non-plotting intermediates are excluded; manifests identify them, but a separate checkpoint archive would be required |
| One-command full retraining | FAIL / NOT CLAIMED | Not tested and intentionally not claimed |
| Historical environment lock | PARTIAL | Python, PyTorch, Captum, RapidFuzz, and CPU execution are known; several package versions and hardware/OS details were not persisted |

## Figure regeneration record

Executed command from the project workspace:

```text
python paper_figures/scripts/generate_main_figures.py
```

The script completed normally. It regenerated SVG, PDF, and 600-dpi PNG outputs and refreshed provenance. The validation manifest reported:

- Figure 3A: 720 sample-run observations and 80 unique sequences.
- Figure 3B: ENCFF148JKK:12229, fold_a, seed 123; eligible n=63; delta_p=0.00600654; Exact ISM normalized L1=0.312214.
- Fetal erythroid GATA1 H1b: NE / NOT ESTIMABLE.
- CTCF Phase 3E H2: effect -0.00639364, 95% CI [-0.01247998, 0.00255896], crossing zero.
- No new quantitative discrepancy.

The self-contained release copy was also executed successfully using only bundled relative plotting inputs. Its exit code was 0, all five PNG hashes remained unchanged, and the build manifest retained the same frozen values and Figure 3B sample.

## Frozen-analysis safeguards

No checkpoint was opened for this audit. No model was trained, no attribution was recomputed, no sample was reselected, and no statistical inference was added. Figure regeneration was the only executed analysis-like workflow and used existing frozen source outputs.

## Release assessment

The package is sufficient to inspect code, configs, frozen decisions, selected summary evidence, and to regenerate Figures 1-5. It is not yet a complete archival release because the author metadata, public repository URL, release tag, DOI, final license, complete environment lock, and checkpoint distribution decision remain unresolved.
