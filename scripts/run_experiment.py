from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.analyze_results import analyze
from src.data import generate_synthetic_data, save_dataset
from src.dna_utils import align_rc_full_attribution, align_rc_position_attribution, reverse_complement
from src.interpret import run_ism_for_sequences
from src.metrics import attribution_consistency, motif_localization, prediction_consistency_metrics, safe_similarity
from src.plotting import create_summary_plots, plot_representative_samples
from src.training import predict_sequences, train_model
from src.models import build_model, parameter_count


REQUIRED_SAMPLE_COLUMNS = [
    "model_type", "seed", "sample_id", "label", "motif_start", "motif_end", "motif_orientation",
    "p_forward", "p_rc", "prediction_difference", "prediction_consistency", "prediction_flip",
    "attribution_pearson_signed", "attribution_pearson_absolute", "attribution_spearman_signed",
    "attribution_spearman_absolute", "attribution_cosine_signed", "attribution_cosine_absolute",
    "attribution_top8_overlap", "motif_top8_overlap_signed", "motif_top8_overlap_absolute",
    "motif_mass_fraction_signed", "motif_mass_fraction_absolute", "motif_vs_background_signed",
    "motif_vs_background_absolute", "position_auprc_signed", "position_auprc_absolute",
]


def _json_dump(value, path: Path):
    path.write_text(json.dumps(value, indent=2, allow_nan=True), encoding="utf-8")


def run(config_path: Path, output_dir: Path | None = None):
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("device") != "cpu": raise ValueError("This project requires device: cpu")
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    output_dir = output_dir or ROOT / "results" / config["project_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Python={platform.python_version()} | PyTorch={torch.__version__} | device=cpu")
    print(f"Config={config_path.resolve()} | seeds={config['training']['seeds']} | data_seed={config['data']['data_seed']}")
    shutil.copy2(config_path, output_dir / "resolved_config.yaml")
    data, leakage_report = generate_synthetic_data(config["data"])
    save_dataset(data, output_dir / "synthetic_dataset.csv")
    _json_dump({"data_config": config["data"], "leakage_report": leakage_report}, output_dir / "data_generation_report.json")
    test = data[data.split == "test"].reset_index(drop=True)
    positives = test[test.label == 1]
    n_ism = min(int(config["interpretation"]["positive_samples"]), len(positives))
    explained = positives.sample(n=n_ism, random_state=int(config["interpretation"]["sample_seed"])).sort_values("sample_id").reset_index(drop=True)
    explained[["sample_id"]].to_csv(output_dir / "ism_sample_ids.csv", index=False, encoding="utf-8")
    sample_rows, prediction_rows, run_rows, anomaly_rows, plot_arrays = [], [], [], [], {}
    training_seconds, ism_seconds = [], []
    for seed in config["training"]["seeds"]:
        for model_type in config["training"]["model_types"]:
            print(f"Training {model_type}, seed={seed}")
            run_dir = output_dir / "runs" / f"{model_type.lower().replace('-', '_')}_seed_{seed}"
            model, metadata = train_model(model_type, int(seed), config, data, run_dir)
            training_seconds.append(metadata["training_seconds"])
            pf = predict_sequences(model, test.sequence.tolist(), int(config["training"]["batch_size"]))
            rc_sequences = [reverse_complement(s) for s in test.sequence]
            pr = predict_sequences(model, rc_sequences, int(config["training"]["batch_size"]))
            pred_summary, pred_each = prediction_consistency_metrics(pf, pr, float(config["training"]["threshold"]))
            for i, row in test.iterrows():
                prediction_rows.append({"model_type": model_type, "seed": seed, "sample_id": row.sample_id,
                                        "label": row.label, "p_forward": pf[i], "p_rc": pr[i],
                                        **{key: value[i] for key, value in pred_each.items()}})
            sequences = explained.sequence.tolist()
            rc_exp = [reverse_complement(s) for s in sequences]
            batch_size = int(config["interpretation"]["batch_size"]); difference = config["interpretation"]["difference"]
            f_matrix, f_signed, f_absolute, f_time = run_ism_for_sequences(model, sequences, batch_size, difference, f"{model_type} forward ISM")
            r_matrix, r_signed, r_absolute, r_time = run_ism_for_sequences(model, rc_exp, batch_size, difference, f"{model_type} RC ISM")
            current_ism_seconds = f_time + r_time; ism_seconds.append(current_ism_seconds)
            aligned_matrix = align_rc_full_attribution(r_matrix)
            aligned_signed = align_rc_position_attribution(r_signed)
            aligned_absolute = align_rc_position_attribution(r_absolute)
            np.savez_compressed(run_dir / "ism_attributions.npz", sample_ids=explained.sample_id.astype(str).to_numpy(dtype=str),
                                forward_matrix=f_matrix, rc_matrix=r_matrix, aligned_rc_matrix=aligned_matrix,
                                forward_signed=f_signed, aligned_rc_signed=aligned_signed,
                                forward_absolute=f_absolute, aligned_rc_absolute=aligned_absolute)
            explained_indices = [test.index[test.sample_id == sid][0] for sid in explained.sample_id]
            for j, (_, row) in enumerate(explained.iterrows()):
                signed_cons, signed_issues = attribution_consistency(f_signed[j], aligned_signed[j], int(config["interpretation"]["top_k"]))
                absolute_cons, absolute_issues = attribution_consistency(f_absolute[j], aligned_absolute[j], int(config["interpretation"]["top_k"]))
                full_pearson, full_issue = safe_similarity(f_matrix[j], aligned_matrix[j], "pearson")
                for issue in signed_issues: anomaly_rows.append({"model_type": model_type, "seed": seed, "sample_id": row.sample_id, "version": "signed", "issue": issue})
                for issue in absolute_issues: anomaly_rows.append({"model_type": model_type, "seed": seed, "sample_id": row.sample_id, "version": "absolute", "issue": issue})
                if full_issue: anomaly_rows.append({"model_type": model_type, "seed": seed, "sample_id": row.sample_id, "version": "full_matrix", "issue": full_issue})
                motif_signed = motif_localization(f_signed[j], int(row.motif_start), int(row.motif_end), int(config["interpretation"]["top_k"]))
                motif_absolute = motif_localization(f_absolute[j], int(row.motif_start), int(row.motif_end), int(config["interpretation"]["top_k"]))
                ix = explained_indices[j]
                sample_rows.append({
                    "model_type": model_type, "seed": seed, "sample_id": row.sample_id, "label": row.label,
                    "sequence": row.sequence,
                    "motif_start": row.motif_start, "motif_end": row.motif_end, "motif_orientation": row.motif_orientation,
                    "p_forward": pf[ix], "p_rc": pr[ix], **{key: value[ix] for key, value in pred_each.items()},
                    "attribution_pearson_signed": signed_cons["pearson"], "attribution_pearson_absolute": absolute_cons["pearson"],
                    "attribution_spearman_signed": signed_cons["spearman"], "attribution_spearman_absolute": absolute_cons["spearman"],
                    "attribution_cosine_signed": signed_cons["cosine"], "attribution_cosine_absolute": absolute_cons["cosine"],
                    "attribution_top8_overlap": absolute_cons["top8_overlap"], "attribution_full_matrix_pearson": full_pearson,
                    "attribution_std_absolute": float(np.std(f_absolute[j])),
                    **{f"motif_{key}_signed": value for key, value in motif_signed.items() if key != "position_auprc"},
                    **{f"motif_{key}_absolute": value for key, value in motif_absolute.items() if key != "position_auprc"},
                    "position_auprc_signed": motif_signed["position_auprc"], "position_auprc_absolute": motif_absolute["position_auprc"],
                })
                plot_arrays[(model_type, int(seed), row.sample_id)] = (f_absolute[j], aligned_absolute[j])
            run_rows.append({"model_type": model_type, "seed": seed, "best_epoch": metadata["best_epoch"],
                             "parameter_count": metadata["parameter_count"], "training_seconds": metadata["training_seconds"],
                             "ism_seconds": current_ism_seconds, **{f"validation_{k}": v for k, v in metadata["validation_metrics"].items()},
                             **{f"test_{k}": v for k, v in metadata["test_metrics"].items()}, **pred_summary})
    sample_df = pd.DataFrame(sample_rows)
    for column in REQUIRED_SAMPLE_COLUMNS:
        if column not in sample_df: sample_df[column] = np.nan
    sample_df.to_csv(output_dir / "per_sample_results.csv", index=False, encoding="utf-8")
    pd.DataFrame(prediction_rows).to_csv(output_dir / "all_test_prediction_pairs.csv", index=False, encoding="utf-8")
    run_df = pd.DataFrame(run_rows); run_df.to_csv(output_dir / "run_summary.csv", index=False, encoding="utf-8")
    pd.DataFrame(anomaly_rows, columns=["model_type", "seed", "sample_id", "version", "issue"]).to_csv(output_dir / "attribution_anomalies.csv", index=False, encoding="utf-8")
    analyze(output_dir)
    create_summary_plots(sample_df, run_df, output_dir / "figures")
    plot_representative_samples(sample_df, plot_arrays, output_dir / "figures")
    elapsed = time.perf_counter() - started
    smoke_train_mean = float(np.mean(training_seconds)); ism_per_pair = float(np.sum(ism_seconds) / (len(ism_seconds) * n_ism))
    full_config = yaml.safe_load((ROOT / "configs" / "full.yaml").read_text(encoding="utf-8"))
    data_ratio = full_config["data"]["train_size"] / config["data"]["train_size"]
    epoch_ratio = full_config["training"]["max_epochs"] / config["training"]["max_epochs"]
    seed_ratio = len(full_config["training"]["seeds"]) / len(config["training"]["seeds"])
    model_ratio = parameter_count(build_model(full_config["model"])) / parameter_count(build_model(config["model"]))
    sample_ratio = full_config["interpretation"]["positive_samples"] / n_ism
    full_training = float(np.sum(training_seconds) * data_ratio * epoch_ratio * seed_ratio * model_ratio)
    full_ism = float(np.sum(ism_seconds) * sample_ratio * seed_ratio * model_ratio)
    full_total = full_training + full_ism
    estimate = {"actual_total_seconds": elapsed, "mean_training_seconds_per_run": smoke_train_mean,
                "mean_forward_plus_rc_ism_seconds_per_sample": ism_per_pair,
                "estimated_full_training_seconds": full_training, "estimated_full_ism_seconds": full_ism,
                "estimated_full_total_seconds": full_total,
                "estimated_full_plausible_range_seconds": [0.6 * full_total, 2.0 * full_total],
                "scaling_factors": {"data": data_ratio, "max_epochs": epoch_ratio, "seeds": seed_ratio,
                                    "trainable_parameters_proxy": model_ratio, "ism_samples": sample_ratio},
                "estimation_note": "Smoke extrapolation uses dataset size, configured max epochs, seeds, trainable-parameter ratio as a compute proxy, and ISM sample count. Early stopping and CPU kernel scaling can materially change runtime; range is 0.6x-2.0x."}
    _json_dump(estimate, output_dir / "runtime_estimate.json")
    print(f"Completed experiment pipeline in {elapsed:.2f}s; outputs: {output_dir.resolve()}")
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(); run(args.config, args.output_dir)


if __name__ == "__main__":
    main()
