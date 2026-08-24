# Paper figure source validation report

## Scope and safeguards

- Scope: Phase 2D, Phase 2E, Phase 2F, fetal erythroid GATA1, GM12878 CTCF, Phase 3D overlap audit, and Phase 3E near-duplicate sensitivity outputs.
- No checkpoint was opened; no model was trained; no frozen output was modified; no final figure was created.
- Prompt-provided reference numbers were used only as validation targets. Plotting must use the source files listed in `source_inventory.csv`.
- Validation/test and Phase 3C/Phase 3E branches remain explicitly separated.

## Key source groups found

- Phase 2D: run-level performance, 27 Exact ISM matrix archives, sample-level prediction pairs and attribution metrics, bootstrap tables, frozen report.
- Phase 2E: biological-validity bootstrap, motif localization/disruption, conservative-IDR summaries and sample-level evidence, frozen report.
- Phase 2F: frozen test prediction/attribution tables, 27 Exact ISM matrix archives, frozen sequence manifest, test bootstrap/decisions, frozen report.
- Phase 3C: separate fetal GATA1 and GM12878 CTCF test summaries, sample-level prediction/attribution tables, per-run Exact ISM matrices, frozen report.
- Phase 3D: coordinate, exact-sequence and 90%/90% high-similarity pair files plus audit summary and trigger decisions.
- Phase 3E: exclusion summary, primary-vs-sensitivity plotting tables, separate sensitivity sample-level outputs and Exact ISM matrices, frozen report.

## Frozen reference-value validation

| Reference item | Observed from source | Target | Status | Source |
|---|---:|---:|---|---|
| K562 confirmatory AUROC CNN-Raw | 0.859061500264 | ≈0.8591 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/training_run_summary.csv` |
| K562 confirmatory AUROC CNN-Aug | 0.869500725993 | ≈0.8695 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/training_run_summary.csv` |
| K562 confirmatory AUROC CNN-RCPS | 0.871613657444 | ≈0.8716 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/training_run_summary.csv` |
| K562 mean prediction asymmetry CNN-Raw | 0.101505244212 | ≈0.1015 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/training_run_summary.csv` |
| K562 mean prediction asymmetry CNN-Aug | 0.0868083433001 | ≈0.0868 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/training_run_summary.csv` |
| K562 mean prediction asymmetry CNN-RCPS | 0 | numerical zero | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/training_run_summary.csv` |
| K562 Aug-Raw prediction asymmetry | -0.0146969009122 | ≈-0.0147 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/hierarchical_paired_bootstrap.csv` |
| K562 Aug-Raw CI low | -0.0226046577563 | ≈-0.0226 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/hierarchical_paired_bootstrap.csv` |
| K562 Aug-Raw CI high | -0.00385601788188 | ≈-0.0039 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/hierarchical_paired_bootstrap.csv` |
| K562 Exact ISM L1 CNN-Raw | 0.338533021984 | ≈0.3385 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/attribution_run_summary.csv` |
| K562 Exact ISM L1 CNN-Aug | 0.310475897959 | ≈0.3105 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/attribution_run_summary.csv` |
| K562 Exact ISM L1 CNN-RCPS | 4.13258516518e-09 | ≈4.13e-9 | VERIFIED | `rc_attribution_project/results/phase2d_rcps_confirmatory/attribution_run_summary.csv` |
| Frozen K562 test N | 3496 | 3496 | VERIFIED | `rc_attribution_project/results/phase2f_one_time_test/test_prediction_results.csv` |
| Frozen K562 chromosomes | chr14,chr18,chr3,chr8 | chr3,chr8,chr14,chr18 | VERIFIED | `rc_attribution_project/results/phase2f_one_time_test/test_prediction_results.csv` |
| Prediction-consistent threshold | 0.01 | delta_p <= 0.01 | VERIFIED | `rc_attribution_project/results/phase2f_one_time_test/resolved_config.yaml` |
| Frozen K562 Raw consistent-subset Exact ISM L1 | 0.311712522827 | ≈0.3117 | VERIFIED | `rc_attribution_project/results/phase2f_one_time_test/test_confirmatory_bootstrap.csv` |
| Frozen K562 Raw consistent-subset CI low | 0.286864319187 | ≈0.2869 | VERIFIED | `rc_attribution_project/results/phase2f_one_time_test/test_confirmatory_bootstrap.csv` |
| Frozen K562 Raw consistent-subset CI high | 0.334847455994 | ≈0.3348 | VERIFIED | `rc_attribution_project/results/phase2f_one_time_test/test_confirmatory_bootstrap.csv` |
| Fetal GATA1 H1b status | not_estimable | NOT ESTIMABLE | VERIFIED | `rc_attribution_project/results/phase3c_one_time_test/p3_gata1_fetal/test_hypothesis_decisions.json` |
| Fetal GATA1 H1b eligible pool | eligible pool=76 | ≈76 | VERIFIED | `rc_attribution_project/results/phase3c_one_time_test/p3_gata1_fetal/test_hypothesis_decisions.json` |
| CTCF H2 near-duplicate effect | -0.00639364428696 | ≈-0.0064 | VERIFIED | `rc_attribution_project/results/phase3e_near_duplicate_sensitivity/evaluation/p3_ctcf_gm12878/test_confirmatory_bootstrap.csv` |
| CTCF H2 near-duplicate CI low | -0.0124799754727 | ≈-0.0125 | VERIFIED | `rc_attribution_project/results/phase3e_near_duplicate_sensitivity/evaluation/p3_ctcf_gm12878/test_confirmatory_bootstrap.csv` |
| CTCF H2 near-duplicate CI high | 0.00255895509213 | ≈0.0026 | VERIFIED | `rc_attribution_project/results/phase3e_near_duplicate_sensitivity/evaluation/p3_ctcf_gm12878/test_confirmatory_bootstrap.csv` |

## Values not verified

- None among the supplied frozen reference values; all were independently located in existing source outputs.

## Figure 3 data readiness

### Figure 3A: per-sample delta_p vs Exact ISM normalized L1

- READY for the frozen Phase 2F Exact-ISM attribution cohort: `rc_attribution_project/results/phase2f_one_time_test/test_attribution_results.csv` already contains `fold_id`, `model_type`, `seed`, `sample_id`, `prediction_difference`, and `absolute_normalized_l1`.
- Independent join to `rc_attribution_project/results/phase2f_one_time_test/test_prediction_results.csv` matched 2160/2160 rows; missing rows=0; maximum delta_p disagreement=0.
- Limitation: Exact ISM was computed only for the frozen attribution cohort (80 samples per fold/model/seed run), not for every one of the 3,496 test sequences. A Figure 3A scatter is therefore valid for that prespecified attribution cohort, not as an all-test-sample scatter.

### Figure 3B: example-sequence Exact ISM attribution map

- READY: 27 Phase 2F NPZ archives were found; total run-sample matrices checked=2160; observed forward-matrix shapes=(80, 256, 4).
- NPZ `sample_ids` link to `rc_attribution_project/results/phase2f_one_time_test/frozen_test_attribution_samples.csv` (`sample_id`, `sequence`) and to the matching per-run attribution CSV. Each checked sequence length matches matrix length and each matrix has four nucleotide channels.
- Matrix validation failures: 0.
- Safe fields for mapping: `forward_matrix` and `aligned_rc_matrix` for the full L×4 map; `forward_absolute` and `aligned_rc_absolute` for position-level absolute summaries.
- The example must be selected by a predeclared rule from the frozen cohort; it must not be chosen by looking for the most visually dramatic sample after plotting.

## Discrepancy assessment

- No discrepancy detected for the supplied reference values when comparing the raw summary/decision outputs with the corresponding frozen DOCX reports. Differences are rounding only.
- This statement is limited to the values and source relationships checked here; it is not a claim that every number in every report was exhaustively re-derived.

## Safe inputs for the next stage

1. Aggregate K562 confirmatory panels: `rc_attribution_project/results/phase2d_rcps_confirmatory/training_run_summary.csv`, `hierarchical_paired_bootstrap.csv`, `hypothesis_endpoint_bootstrap.csv`, and `attribution_run_summary.csv` in the same directory.
2. Frozen held-out K562 panels: `rc_attribution_project/results/phase2f_one_time_test/test_prediction_ensemble_summary.csv`, `test_confirmatory_bootstrap.csv`, and `test_hypothesis_decisions.json` in the same directory.
3. Figure 3A: `rc_attribution_project/results/phase2f_one_time_test/test_attribution_results.csv`, optionally cross-checked/joined to `test_prediction_results.csv` using `fold_id + model_type + seed + sample_id`.
4. Figure 3B: matching `rc_attribution_project/results/phase2f_one_time_test/runs/<run>/test_exact_ism_attributions.npz` plus `rc_attribution_project/results/phase2f_one_time_test/frozen_test_attribution_samples.csv`; keep run/fold/model/seed fixed.
5. External replication panels: task-specific Phase 3C `test_confirmatory_bootstrap.csv`, `test_prediction_ensemble_summary.csv`, and `test_hypothesis_decisions.json` under `rc_attribution_project/results/phase3c_one_time_test/p3_gata1_fetal/` or `p3_ctcf_gm12878/`.
6. Overlap/sensitivity panels: `rc_attribution_project/results/phase3d_cross_phase_overlap_audit/audit_summary.json` and pair files; `rc_attribution_project/results/phase3e_near_duplicate_sensitivity/report_tables/*.csv` and task-specific sensitivity bootstrap tables. Phase 3E must be labeled sensitivity analysis and must not replace Phase 3C primary results.
7. Phase 2E biological-validity panels: use files under `rc_attribution_project/results/phase2e_biological_correctness/` directly; do not reconstruct values from the DOCX report.

## Unsafe or prohibited uses

- Do not use DOCX prose or prompt reference numbers as plotting data.
- Do not mix Phase 2D validation/holdout rows with Phase 2F frozen test rows in a single unlabeled distribution.
- Do not merge Phase 3C primary and Phase 3E sensitivity rows as if they were independent samples.
- Do not infer uncomputed all-test Exact ISM values from aggregate summaries.
- Do not read checkpoints or rerun attribution/training during the plotting stages.

## Inventory completeness

- Inventory rows: 90.
- Missing listed sources: 0.
- Report/source textual checks: {"phase2d_reference_values_present": true, "phase2f_h1b_values_present": true, "phase3c_gata_h1b_present": true, "phase3e_ctcf_h2_present": true}.
