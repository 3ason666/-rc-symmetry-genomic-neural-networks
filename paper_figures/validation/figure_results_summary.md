# Main Figure results summary

All quantitative values below were read or aggregated from frozen source files at figure-generation time; prompt reference values were not used as plotting data.

## Figure 2

- CNN-Raw: mean AUROC=0.859062; mean prediction asymmetry=0.101505; mean flip rate=0.114568.
- CNN-Aug: mean AUROC=0.869501; mean prediction asymmetry=0.086808; mean flip rate=0.096688.
- CNN-RCPS: mean AUROC=0.871614; mean prediction asymmetry=0.000000; mean flip rate=0.000000.
- P2D CNN-Aug minus CNN-Raw prediction asymmetry=-0.014697, 95% CI [-0.022605, -0.003856].

## Figure 3

- Figure 3A rows=720; unique sample IDs=80.
- Figure 3B eligible subset rows=63; eligible L1 median=0.312214325011.
- Selected sample_id=ENCFF148JKK:12229; fold=fold_a; seed=123; p(S)=0.95366335; p(RC(S))=0.9476568; delta_p=0.00600653886795; Exact ISM L1=0.312214325011.
- Selected NPZ source: `paper_figures/data/phase2f_one_time_test/runs/fold_a__cnn_raw__seed_123/test_exact_ism_attributions.npz`.
- P2F CNN-Raw prediction-consistent Exact ISM L1=0.311712522827, 95% CI [0.286864319187, 0.334847455994].

## Figure 5

- Fetal GATA1 H1b is displayed as NOT ESTIMABLE (NE), not failed.
- CTCF H2 sensitivity effect=-0.00639364428696, 95% CI [-0.0124799754727, 0.00255895509213]; the interval crosses zero.
- Phase 3C primary external replication and Phase 3E sensitivity are kept separate.

## Discrepancy check

- No new discrepancy detected while assembling Main Figures 1-5.
