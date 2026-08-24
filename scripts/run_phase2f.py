from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.run_phase2a import attribution_metrics
from scripts.run_phase2d import MODEL_COLORS, MODEL_ORDER, load_checkpoint, run_dir_name
from scripts.run_phase2e import (
    benjamini_hochberg,
    disrupt_best_motif,
    edit_matched_flank,
    hierarchical_bootstrap,
    load_pwm,
    localization_metrics,
    predict_logits,
    scan_pwm,
    select_matched_flank_start,
)
from src.dna_utils import (
    align_rc_full_attribution,
    align_rc_position_attribution,
    reverse_complement,
)
from src.interpret import run_ism_for_sequences
from src.metrics import prediction_consistency_metrics, safe_similarity
from src.training import DEVICE, predict_sequences


PROTOCOL_REVISION = "phase2f_one_time_test_v1"
CONTRASTS = [
    ("CNN-Aug", "CNN-Raw", "CNN-Aug_minus_CNN-Raw"),
    ("CNN-RCPS", "CNN-Aug", "CNN-RCPS_minus_CNN-Aug"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_frozen_config(config: dict) -> None:
    if config.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("P2F protocol revision changed")
    if config.get("device") != "cpu":
        raise ValueError("P2F v1 is frozen to CPU")
    dataset = config["dataset"]
    if dataset.get("split") != "test":
        raise ValueError("P2F must evaluate exactly the test split")
    if dataset.get("chromosomes") != ["chr3", "chr8", "chr14", "chr18"]:
        raise ValueError("P2F test chromosomes changed")
    if int(dataset.get("expected_rows")) != 3496 or int(dataset.get("expected_per_label")) != 1748:
        raise ValueError("P2F expected test population changed")
    checkpoints = config["checkpoints"]
    if checkpoints.get("model_types") != MODEL_ORDER:
        raise ValueError("P2F model order changed")
    if checkpoints.get("folds") != ["fold_a", "fold_b", "fold_c"]:
        raise ValueError("P2F training fold set changed")
    if checkpoints.get("seeds") != [42, 123, 2026]:
        raise ValueError("P2F seed set changed")
    if int(checkpoints.get("expected_count")) != 27 or checkpoints.get("retraining_allowed"):
        raise ValueError("P2F checkpoint or retraining rule changed")
    attribution = config["attribution"]
    if attribution.get("method") != "exact_ism" or attribution.get("difference") != "logit":
        raise ValueError("P2F primary attribution method changed")
    if int(attribution.get("per_chromosome_per_label")) != 10:
        raise ValueError("P2F attribution sampling rule changed")
    if int(attribution.get("expected_samples")) != 80:
        raise ValueError("P2F attribution cohort size changed")
    if float(config["biological"].get("relative_score_threshold")) != 0.80:
        raise ValueError("P2F motif threshold changed")
    decision = config["decision"]
    if not decision.get("test_unseal_authorized"):
        raise ValueError("P2F test unseal is not authorized")
    if decision.get("change_protocol_after_unseal") or decision.get("tune_after_test"):
        raise ValueError("P2F permits forbidden post-test changes")
    if decision.get("select_models_after_test"):
        raise ValueError("P2F permits post-test model selection")
    if config["execution"].get("allow_force_rerun"):
        raise ValueError("P2F must not expose a force-rerun option")


def checkpoint_records(config: dict) -> list[dict]:
    source = ROOT / config["checkpoints"]["source_results"] / "runs"
    records = []
    for summary_path in sorted(source.glob("*/run_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = summary_path.parent / "best_checkpoint.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        records.append(
            {
                "run_name": summary_path.parent.name,
                "fold_id": str(summary["fold_id"]),
                "model_type": str(summary["model_type"]),
                "seed": int(summary["seed"]),
                "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
                "checkpoint_size": int(checkpoint.stat().st_size),
                "checkpoint_sha256": sha256_file(checkpoint),
            }
        )
    records.sort(key=lambda row: (row["fold_id"], row["seed"], row["model_type"]))
    expected = {
        (fold, int(seed), model)
        for fold in config["checkpoints"]["folds"]
        for seed in config["checkpoints"]["seeds"]
        for model in MODEL_ORDER
    }
    observed = {(row["fold_id"], row["seed"], row["model_type"]) for row in records}
    if observed != expected or len(records) != int(config["checkpoints"]["expected_count"]):
        raise ValueError("P2F checkpoint inventory is incomplete")
    return records


def build_input_manifest(config_path: Path, config: dict) -> dict:
    protocol_path = ROOT / config["protocol_path"]
    dataset_path = ROOT / config["dataset"]["path"]
    pwm_path = ROOT / config["biological"]["pwm_resource"]
    p2d_gate_path = ROOT / config["checkpoints"]["source_results"] / "phase2d_gate_assessment.json"
    p2e_gate_path = ROOT / "results" / "phase2e_biological_correctness" / "phase2e_gate_assessment.json"
    runner_path = Path(__file__).resolve()
    p2d_gate = json.loads(p2d_gate_path.read_text(encoding="utf-8"))
    p2e_gate = json.loads(p2e_gate_path.read_text(encoding="utf-8"))
    if not p2d_gate.get("test_seal_intact") or int(p2d_gate.get("test_rows_evaluated", -1)) != 0:
        raise ValueError("P2D test seal is not intact")
    if not p2e_gate.get("test_seal_intact") or int(p2e_gate.get("test_model_rows", -1)) != 0:
        raise ValueError("P2E test seal is not intact")
    dataset_sha = sha256_file(dataset_path)
    if dataset_sha.lower() != str(config["dataset"]["sha256"]).lower():
        raise ValueError("Frozen dataset checksum mismatch")
    manifest = {
        "protocol_revision": PROTOCOL_REVISION,
        "created_before_test_rows_loaded": True,
        "authorization_date": str(config["authorization_date"]),
        "resources": {
            "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(config_path)},
            "protocol": {"path": protocol_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(protocol_path)},
            "runner": {"path": runner_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(runner_path)},
            "dataset": {"path": dataset_path.relative_to(ROOT).as_posix(), "sha256": dataset_sha},
            "pwm": {"path": pwm_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(pwm_path)},
            "phase2d_gate": {"path": p2d_gate_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(p2d_gate_path)},
            "phase2e_gate": {"path": p2e_gate_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(p2e_gate_path)},
        },
        "prior_test_access": {
            "phase2d_test_rows": int(p2d_gate["test_rows_evaluated"]),
            "phase2e_test_rows": int(p2e_gate["test_model_rows"]),
        },
        "checkpoints": checkpoint_records(config),
    }
    manifest_path = ROOT / config["checkpoints"]["manifest_path"]
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing P2F input manifest differs from current frozen inputs")
    else:
        write_json(manifest_path, manifest)
    return manifest


def verify_input_manifest(config_path: Path, config: dict) -> dict:
    manifest_path = ROOT / config["checkpoints"]["manifest_path"]
    if not manifest_path.exists():
        raise FileNotFoundError("P2F input manifest must be frozen before test unsealing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("P2F input manifest protocol changed")
    checks = []
    for name, record in manifest["resources"].items():
        path = ROOT / record["path"]
        observed = sha256_file(path)
        checks.append({"resource": name, "expected": record["sha256"], "observed": observed, "passed": observed == record["sha256"]})
    for record in manifest["checkpoints"]:
        path = ROOT / record["checkpoint_path"]
        observed = sha256_file(path)
        checks.append({"resource": record["run_name"], "expected": record["checkpoint_sha256"], "observed": observed, "passed": observed == record["checkpoint_sha256"]})
    if not all(row["passed"] for row in checks):
        raise ValueError("P2F frozen input checksum audit failed")
    if sha256_file(config_path) != manifest["resources"]["config"]["sha256"]:
        raise ValueError("P2F config changed after freeze")
    return {"manifest_path": manifest_path.relative_to(ROOT).as_posix(), "manifest_sha256": sha256_file(manifest_path), "checks": checks, "all_passed": True}


def start_unseal_ledger(config_path: Path, config: dict, audit: dict, resume: bool) -> dict:
    output = ROOT / config["execution"]["output_dir"]
    completion = ROOT / config["execution"]["completion_path"]
    ledger_path = ROOT / config["execution"]["ledger_path"]
    if completion.exists():
        raise RuntimeError("P2F is already complete; a second test execution is forbidden")
    fingerprints = {
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": audit["manifest_sha256"],
        "protocol_revision": PROTOCOL_REVISION,
    }
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError("P2F has already been unsealed; use --resume only for the identical interrupted run")
        if ledger.get("fingerprints") != fingerprints:
            raise ValueError("P2F resume fingerprints changed")
        ledger["resume_count"] = int(ledger.get("resume_count", 0)) + 1
        ledger["last_resumed_at_utc"] = utc_now()
        ledger["status"] = "resuming_same_frozen_run"
    else:
        if resume:
            raise RuntimeError("No interrupted P2F ledger exists to resume")
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("P2F output directory is non-empty without an unseal ledger")
        output.mkdir(parents=True, exist_ok=True)
        ledger = {
            "protocol_revision": PROTOCOL_REVISION,
            "authorized": True,
            "started_at_utc": utc_now(),
            "status": "unsealed_running",
            "resume_count": 0,
            "fingerprints": fingerprints,
            "test_rows_loaded": False,
            "test_predictions_generated": False,
            "test_attributions_generated": False,
        }
    write_json(ledger_path, ledger)
    return ledger


def validate_test_population(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    test = frame[frame.split.eq("test")].copy()
    expected_chromosomes = set(config["dataset"]["chromosomes"])
    if len(test) != int(config["dataset"]["expected_rows"]):
        raise ValueError("P2F test row count changed")
    if set(test.chromosome.unique()) != expected_chromosomes:
        raise ValueError("P2F test chromosome membership changed")
    counts = test.groupby("label").size().to_dict()
    if counts != {0: int(config["dataset"]["expected_per_label"]), 1: int(config["dataset"]["expected_per_label"])}:
        raise ValueError("P2F test label balance changed")
    if test.sample_id.duplicated().any() or test.canonical_key.duplicated().any():
        raise ValueError("P2F test identifiers are not unique")
    return test.sort_values(["chromosome", "label", "sample_id"], kind="mergesort").reset_index(drop=True)


def selection_digest(row: pd.Series, salt: str) -> str:
    payload = f"{salt}|{row.chromosome}|{int(row.label)}|{row.sample_id}|{row.canonical_key}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_attribution_cohort(test: pd.DataFrame, config: dict, output: Path) -> pd.DataFrame:
    path = output / "frozen_test_attribution_samples.csv"
    n = int(config["attribution"]["per_chromosome_per_label"])
    salt = str(config["attribution"]["selection_salt"])
    rows = []
    for chromosome in config["dataset"]["chromosomes"]:
        for label in (0, 1):
            pool = test[test.chromosome.eq(chromosome) & test.label.eq(label)].copy()
            pool["selection_digest"] = pool.apply(selection_digest, axis=1, salt=salt)
            rows.append(pool.sort_values("selection_digest", kind="mergesort").head(n))
    frozen = pd.concat(rows, ignore_index=True).sort_values(["chromosome", "label", "selection_digest"], kind="mergesort")
    columns = ["sample_id", "pair_id", "canonical_key", "chromosome", "label", "sequence", "selection_digest"]
    frozen = frozen[columns].reset_index(drop=True)
    if len(frozen) != int(config["attribution"]["expected_samples"]):
        raise ValueError("P2F attribution cohort size changed")
    expected = {(chromosome, label): n for chromosome in config["dataset"]["chromosomes"] for label in (0, 1)}
    if frozen.groupby(["chromosome", "label"]).size().to_dict() != expected:
        raise ValueError("P2F attribution cohort is not balanced")
    if path.exists():
        existing = pd.read_csv(path)
        pd.testing.assert_frame_equal(existing, frozen, check_dtype=False)
    else:
        frozen.to_csv(path, index=False, encoding="utf-8")
    return frozen


def update_ledger(config: dict, **updates) -> None:
    path = ROOT / config["execution"]["ledger_path"]
    ledger = json.loads(path.read_text(encoding="utf-8"))
    ledger.update(updates)
    write_json(path, ledger)


def prediction_stage(test: pd.DataFrame, config: dict, manifest: dict, output: Path) -> None:
    sequences = test.sequence.astype(str).tolist()
    rc_sequences = [reverse_complement(sequence) for sequence in sequences]
    labels = test.label.to_numpy(dtype=int)
    threshold = float(config["prediction"]["threshold"])
    batch_size = int(config["prediction"]["batch_size"])
    for record in manifest["checkpoints"]:
        run_output = output / "runs" / record["run_name"]
        prediction_path = run_output / "test_prediction_pairs.csv"
        summary_path = run_output / "test_prediction_summary.json"
        if prediction_path.exists() and summary_path.exists():
            continue
        run_output.mkdir(parents=True, exist_ok=True)
        print(f"P2F prediction {record['run_name']} n={len(test)}", flush=True)
        model, _ = load_checkpoint(ROOT / record["checkpoint_path"])
        forward = predict_sequences(model, sequences, batch_size)
        reverse = predict_sequences(model, rc_sequences, batch_size)
        consistency, each = prediction_consistency_metrics(forward, reverse, threshold)
        predictions = pd.DataFrame(
            {
                "fold_id": record["fold_id"],
                "model_type": record["model_type"],
                "seed": int(record["seed"]),
                "sample_id": test.sample_id,
                "pair_id": test.pair_id,
                "canonical_key": test.canonical_key,
                "chromosome": test.chromosome,
                "label": labels,
                "p_forward": forward,
                "p_rc": reverse,
                "prediction_difference": each["prediction_difference"],
                "prediction_flip": each["prediction_flip"],
            }
        )
        predictions.to_csv(prediction_path, index=False, encoding="utf-8")
        summary = {
            "fold_id": record["fold_id"],
            "model_type": record["model_type"],
            "seed": int(record["seed"]),
            "n": int(len(test)),
            "auroc": float(roc_auc_score(labels, forward)),
            "auprc": float(average_precision_score(labels, forward)),
            **consistency,
            "rcps_prediction_max_abs_error": float(np.max(np.abs(forward - reverse))) if record["model_type"] == "CNN-RCPS" else None,
        }
        write_json(summary_path, summary)
    update_ledger(config, test_predictions_generated=True, prediction_stage_completed_at_utc=utc_now())


def aggregate_predictions(output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = pd.concat([pd.read_csv(path) for path in sorted((output / "runs").glob("*/test_prediction_pairs.csv"))], ignore_index=True)
    summaries = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "runs").glob("*/test_prediction_summary.json"))])
    detail.sort_values(["fold_id", "seed", "model_type", "sample_id"]).to_csv(output / "test_prediction_results.csv", index=False)
    summaries.sort_values(["fold_id", "seed", "model_type"]).to_csv(output / "test_prediction_run_summary.csv", index=False)
    return detail, summaries


def scan_cohort(frozen: pd.DataFrame, config: dict, output: Path):
    resource_path = ROOT / config["biological"]["pwm_resource"]
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    pseudocount = float(resource.get("conversion", {}).get("pseudocount", 0.5))
    background = resource.get("conversion", {}).get("background", {base: 0.25 for base in "ACGT"})
    _, pwm, minimum, maximum = load_pwm(resource_path, pseudocount, background)
    rows = []
    threshold = float(config["biological"]["relative_score_threshold"])
    for row in frozen.to_dict("records"):
        hit = scan_pwm(str(row["sequence"]), pwm, minimum, maximum)
        rows.append({"sample_id": row["sample_id"], "chromosome": row["chromosome"], "label": int(row["label"]), **hit, "strong_hit_080": bool(hit["motif_relative_score"] >= threshold)})
    scans = pd.DataFrame(rows).sort_values(["chromosome", "label", "sample_id"])
    scans.to_csv(output / "test_motif_scan_samples.csv", index=False)
    return scans, pwm


def prepare_disruptions(frozen: pd.DataFrame, scans: pd.DataFrame, pwm: np.ndarray) -> dict[str, dict]:
    positives = frozen[frozen.label.eq(1)].merge(scans, on=["sample_id", "chromosome", "label"])
    positives = positives[positives.strong_hit_080].copy()
    prepared = {}
    width = int(pwm.shape[1])
    for row in positives.to_dict("records"):
        hit = {key: row[key] for key in ("motif_start", "motif_end", "motif_strand")}
        motif_sequence, motif_changes = disrupt_best_motif(row["sequence"], hit, pwm)
        flank_start = select_matched_flank_start(len(row["sequence"]), int(row["motif_start"]), width)
        flank_sequence, flank_changes = edit_matched_flank(row["sequence"], flank_start, motif_changes, int(row["motif_start"]), width)
        if len(motif_changes) != len(flank_changes):
            raise ValueError("P2F motif and flank mutation counts differ")
        prepared[str(row["sample_id"])] = {**row, "motif_disrupted_sequence": motif_sequence, "flank_disrupted_sequence": flank_sequence, "mutation_count": len(motif_changes), "flank_start": int(flank_start)}
    return prepared


def attribution_stage(frozen: pd.DataFrame, config: dict, manifest: dict, output: Path) -> None:
    sequences = frozen.sequence.astype(str).tolist()
    rc_sequences = [reverse_complement(sequence) for sequence in sequences]
    scans, pwm = scan_cohort(frozen, config, output)
    scan_index = scans.set_index("sample_id")
    rng = np.random.default_rng(int(config["biological"]["circular_shift_seed"]))
    shifts = np.sort(rng.choice(np.arange(1, 256), size=int(config["biological"]["circular_shift_count"]), replace=False))
    prepared = prepare_disruptions(frozen, scans, pwm)
    for record in manifest["checkpoints"]:
        run_output = output / "runs" / record["run_name"]
        complete_path = run_output / "test_attribution_complete.json"
        result_path = run_output / "test_attribution_results.csv"
        disruption_path = run_output / "test_motif_disruption_results.csv"
        npz_path = run_output / "test_exact_ism_attributions.npz"
        if complete_path.exists() and result_path.exists() and disruption_path.exists() and npz_path.exists():
            continue
        print(f"P2F Exact ISM {record['run_name']} n={len(frozen)}", flush=True)
        model, _ = load_checkpoint(ROOT / record["checkpoint_path"])
        forward_matrix, forward_signed, forward_absolute, forward_seconds = run_ism_for_sequences(
            model, sequences, int(config["attribution"]["ism_batch_size"]), "logit", f"{record['run_name']} P2F forward ISM"
        )
        rc_matrix, rc_signed, rc_absolute, rc_seconds = run_ism_for_sequences(
            model, rc_sequences, int(config["attribution"]["ism_batch_size"]), "logit", f"{record['run_name']} P2F RC ISM"
        )
        aligned_matrix = align_rc_full_attribution(rc_matrix)
        aligned_signed = align_rc_position_attribution(rc_signed)
        aligned_absolute = align_rc_position_attribution(rc_absolute)
        np.savez_compressed(
            npz_path,
            sample_ids=frozen.sample_id.astype(str).to_numpy(dtype=str),
            forward_matrix=forward_matrix,
            aligned_rc_matrix=aligned_matrix,
            forward_signed=forward_signed,
            aligned_rc_signed=aligned_signed,
            forward_absolute=forward_absolute,
            aligned_rc_absolute=aligned_absolute,
        )
        prediction_index = pd.read_csv(run_output / "test_prediction_pairs.csv").set_index("sample_id")
        rows = []
        for index, sample in frozen.iterrows():
            absolute = attribution_metrics(forward_absolute[index], aligned_absolute[index], int(config["attribution"]["top_k"]))
            signed = attribution_metrics(forward_signed[index], aligned_signed[index], int(config["attribution"]["top_k"]))
            matrix_pearson, matrix_issue = safe_similarity(forward_matrix[index], aligned_matrix[index], "pearson")
            prediction = prediction_index.loc[sample.sample_id]
            biological = {
                "motif_relative_score": np.nan,
                "strong_hit_080": False,
                "motif_mass_fraction": np.nan,
                "top7_motif_recall": np.nan,
                "motif_mass_enrichment_vs_circular": np.nan,
            }
            if int(sample.label) == 1:
                hit = scan_index.loc[sample.sample_id]
                local = localization_metrics(forward_absolute[index], hit, int(config["biological"]["top_k"]), shifts)
                biological = {"motif_relative_score": float(hit.motif_relative_score), "strong_hit_080": bool(hit.strong_hit_080), **local}
            rows.append(
                {
                    "fold_id": record["fold_id"],
                    "model_type": record["model_type"],
                    "seed": int(record["seed"]),
                    "sample_id": sample.sample_id,
                    "chromosome": sample.chromosome,
                    "label": int(sample.label),
                    "prediction_difference": float(prediction.prediction_difference),
                    "prediction_consistent": bool(float(prediction.prediction_difference) <= float(config["attribution"]["prediction_consistent_max_abs_difference"])),
                    **{f"absolute_{key}": value for key, value in absolute.items()},
                    **{f"signed_{key}": value for key, value in signed.items()},
                    "full_matrix_pearson": matrix_pearson,
                    "full_matrix_issue": matrix_issue or "",
                    **biological,
                }
            )
        pd.DataFrame(rows).to_csv(result_path, index=False)
        disruption_rows = []
        if prepared:
            sample_ids = sorted(prepared)
            original = [prepared[sample_id]["sequence"] for sample_id in sample_ids]
            motif_sequences = [prepared[sample_id]["motif_disrupted_sequence"] for sample_id in sample_ids]
            flank_sequences = [prepared[sample_id]["flank_disrupted_sequence"] for sample_id in sample_ids]
            original_logits = predict_logits(model, original, int(config["prediction"]["batch_size"]))
            motif_logits = predict_logits(model, motif_sequences, int(config["prediction"]["batch_size"]))
            flank_logits = predict_logits(model, flank_sequences, int(config["prediction"]["batch_size"]))
            for index, sample_id in enumerate(sample_ids):
                meta = prepared[sample_id]
                motif_drop = float(original_logits[index] - motif_logits[index])
                flank_drop = float(original_logits[index] - flank_logits[index])
                disruption_rows.append(
                    {
                        "fold_id": record["fold_id"],
                        "model_type": record["model_type"],
                        "seed": int(record["seed"]),
                        "sample_id": sample_id,
                        "chromosome": meta["chromosome"],
                        "motif_relative_score": float(meta["motif_relative_score"]),
                        "motif_mutation_count": int(meta["mutation_count"]),
                        "motif_logit_drop": motif_drop,
                        "flank_logit_drop": flank_drop,
                        "motif_minus_flank_logit_drop": motif_drop - flank_drop,
                    }
                )
        pd.DataFrame(disruption_rows).to_csv(disruption_path, index=False)
        matrix_max = float(np.max(np.abs(forward_matrix - aligned_matrix)))
        write_json(
            complete_path,
            {
                "fold_id": record["fold_id"],
                "model_type": record["model_type"],
                "seed": int(record["seed"]),
                "samples": int(len(frozen)),
                "strong_motif_positive_samples": int(len(prepared)),
                "exact_ism_aligned_matrix_max_abs_error": matrix_max,
                "forward_seconds": float(forward_seconds),
                "rc_seconds": float(rc_seconds),
                "rcps_exact_ism_gate_passed": True if record["model_type"] != "CNN-RCPS" else matrix_max <= float(config["invariance_gates"]["exact_ism_aligned_max_abs_tolerance"]),
            },
        )
    update_ledger(config, test_attributions_generated=True, attribution_stage_completed_at_utc=utc_now())


def aggregate_attributions(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attributions = pd.concat([pd.read_csv(path) for path in sorted((output / "runs").glob("*/test_attribution_results.csv"))], ignore_index=True)
    disruptions = pd.concat([pd.read_csv(path) for path in sorted((output / "runs").glob("*/test_motif_disruption_results.csv"))], ignore_index=True)
    audits = pd.DataFrame([json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "runs").glob("*/test_attribution_complete.json"))])
    attributions.sort_values(["fold_id", "seed", "model_type", "sample_id"]).to_csv(output / "test_attribution_results.csv", index=False)
    disruptions.sort_values(["fold_id", "seed", "model_type", "sample_id"]).to_csv(output / "test_motif_disruption_results.csv", index=False)
    audits.sort_values(["fold_id", "seed", "model_type"]).to_csv(output / "test_exact_ism_invariance_audit.csv", index=False)
    return attributions, disruptions, audits


def paired_difference(frame: pd.DataFrame, endpoint: str, later: str, earlier: str) -> pd.DataFrame:
    keys = ["fold_id", "seed", "sample_id"]
    left = frame[frame.model_type.eq(later)][keys + [endpoint]]
    right = frame[frame.model_type.eq(earlier)][keys + [endpoint]]
    paired = left.merge(right, on=keys, suffixes=("_later", "_earlier"), validate="one_to_one")
    paired["difference"] = paired[f"{endpoint}_later"] - paired[f"{endpoint}_earlier"]
    return paired


def bootstrap_endpoint(frame: pd.DataFrame, column: str, config: dict, seed_offset: int) -> dict:
    source = frame[["fold_id", "seed", "sample_id", column]].rename(columns={column: "value"})
    return hierarchical_bootstrap(source, "value", int(config["statistics"]["hierarchical_bootstrap_replicates"]), int(config["statistics"]["hierarchical_bootstrap_seed"]) + seed_offset)


def create_figures(run_summary: pd.DataFrame, attr: pd.DataFrame, disruption: pd.DataFrame, output: Path) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, title, filename in (
        ("auroc", "Test AUROC", "P2F sealed-test classification performance", "p2f_test_auroc.png"),
        ("prediction_mean_absolute_difference", "Mean |p(S)-p(RC(S))|", "P2F sealed-test prediction asymmetry", "p2f_prediction_asymmetry.png"),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
        for index, model in enumerate(MODEL_ORDER):
            values = run_summary.loc[run_summary.model_type.eq(model), metric].to_numpy(float)
            ax.scatter(np.full(len(values), index), values, color=MODEL_COLORS[model], s=42)
            ax.hlines(values.mean(), index - 0.24, index + 0.24, color="black", linewidth=2)
        ax.set_xticks(range(3), MODEL_ORDER)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(figures / filename, dpi=180)
        fig.savefig(figures / filename.replace(".png", ".pdf"))
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
    data = [attr.loc[attr.model_type.eq(model), "absolute_normalized_l1"] for model in MODEL_ORDER]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, model in zip(parts["bodies"], MODEL_ORDER):
        body.set_facecolor(MODEL_COLORS[model]); body.set_alpha(0.55)
    ax.set_xticks(range(1, 4), MODEL_ORDER)
    ax.set_ylabel("RC-aligned Exact ISM normalized L1")
    ax.set_title("P2F sealed-test attribution asymmetry")
    ax.axhline(0, color="black", linewidth=1)
    fig.savefig(figures / "p2f_exact_ism_asymmetry.png", dpi=180)
    fig.savefig(figures / "p2f_exact_ism_asymmetry.pdf")
    plt.close(fig)
    if len(disruption):
        fig, ax = plt.subplots(figsize=(7.4, 4.5), constrained_layout=True)
        data = [disruption.loc[disruption.model_type.eq(model), "motif_minus_flank_logit_drop"] for model in MODEL_ORDER]
        parts = ax.violinplot(data, showmeans=True, showextrema=False)
        for body, model in zip(parts["bodies"], MODEL_ORDER):
            body.set_facecolor(MODEL_COLORS[model]); body.set_alpha(0.55)
        ax.set_xticks(range(1, 4), MODEL_ORDER)
        ax.set_ylabel("Motif drop minus flank drop (logit)")
        ax.set_title("P2F GATA1 motif-disruption specificity")
        ax.axhline(0, color="black", linewidth=1)
        fig.savefig(figures / "p2f_motif_disruption.png", dpi=180)
        fig.savefig(figures / "p2f_motif_disruption.pdf")
        plt.close(fig)


def finalize(config: dict, manifest: dict, audit: dict, output: Path) -> None:
    predictions, run_summary = aggregate_predictions(output)
    attributions, disruptions, invariance = aggregate_attributions(output)
    expected_prediction_rows = int(config["dataset"]["expected_rows"]) * int(config["checkpoints"]["expected_count"])
    expected_attribution_rows = int(config["attribution"]["expected_samples"]) * int(config["checkpoints"]["expected_count"])
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(f"P2F prediction rows incomplete: {len(predictions)}/{expected_prediction_rows}")
    if len(attributions) != expected_attribution_rows:
        raise RuntimeError(f"P2F attribution rows incomplete: {len(attributions)}/{expected_attribution_rows}")
    if len(run_summary) != int(config["checkpoints"]["expected_count"]):
        raise RuntimeError("P2F run summary count incomplete")

    ensemble_rows = []
    ensemble_detail = predictions.groupby(["model_type", "sample_id", "chromosome", "label"], observed=True)[["p_forward", "p_rc"]].mean().reset_index()
    for model in MODEL_ORDER:
        group = ensemble_detail[ensemble_detail.model_type.eq(model)].copy()
        consistency, each = prediction_consistency_metrics(group.p_forward, group.p_rc, float(config["prediction"]["threshold"]))
        group["prediction_difference"] = each["prediction_difference"]
        group["prediction_flip"] = each["prediction_flip"]
        ensemble_rows.append({"model_type": model, "n": int(len(group)), "auroc": float(roc_auc_score(group.label, group.p_forward)), "auprc": float(average_precision_score(group.label, group.p_forward)), **consistency})
    ensemble_detail.to_csv(output / "test_prediction_ensemble_detail.csv", index=False)
    pd.DataFrame(ensemble_rows).to_csv(output / "test_prediction_ensemble_summary.csv", index=False)

    bootstrap_rows = []
    specs = [
        ("H1a", "CNN-Raw test prediction asymmetry", predictions[predictions.model_type.eq("CNN-Raw")], "prediction_difference", 100),
        ("H1b", "CNN-Raw Exact ISM asymmetry among prediction-consistent samples", attributions[attributions.model_type.eq("CNN-Raw") & attributions.prediction_consistent], "absolute_normalized_l1", 101),
        ("H3a", "CNN-Aug residual Exact ISM asymmetry", attributions[attributions.model_type.eq("CNN-Aug")], "absolute_normalized_l1", 102),
    ]
    for hypothesis, analysis, frame, endpoint, offset in specs:
        result = bootstrap_endpoint(frame, endpoint, config, offset)
        bootstrap_rows.append({"family": "confirmatory", "hypothesis": hypothesis, "analysis": analysis, "endpoint": endpoint, "contrast": "vs_zero_or_frozen_threshold", **result})
    for offset, (later, earlier, name) in enumerate(CONTRASTS):
        pred_pair = paired_difference(predictions, "prediction_difference", later, earlier)
        result = bootstrap_endpoint(pred_pair, "difference", config, 200 + offset)
        bootstrap_rows.append({"family": "confirmatory", "hypothesis": "H2" if later == "CNN-Aug" else "H3b", "analysis": "paired test prediction asymmetry", "endpoint": "prediction_difference", "contrast": name, **result})
        attr_pair = paired_difference(attributions, "absolute_normalized_l1", later, earlier)
        result = bootstrap_endpoint(attr_pair, "difference", config, 210 + offset)
        bootstrap_rows.append({"family": "confirmatory", "hypothesis": "H3a" if later == "CNN-Aug" else "H3b", "analysis": "paired test Exact ISM asymmetry", "endpoint": "absolute_normalized_l1", "contrast": name, **result})

    rcps_attr = attributions[attributions.model_type.eq("CNN-RCPS") & attributions.label.eq(1) & attributions.strong_hit_080][["fold_id", "seed", "sample_id", "top7_motif_recall"]]
    rcps_disruption = disruptions[disruptions.model_type.eq("CNN-RCPS")][["fold_id", "seed", "sample_id", "motif_minus_flank_logit_drop"]]
    h4_data = rcps_attr.merge(rcps_disruption, on=["fold_id", "seed", "sample_id"], validate="one_to_one")
    h4_data["diagnostic_failure"] = ((h4_data.top7_motif_recall < float(config["biological"]["low_recall_threshold"])) | (h4_data.motif_minus_flank_logit_drop <= 0)).astype(float)
    h4_result = bootstrap_endpoint(h4_data, "diagnostic_failure", config, 300)
    bootstrap_rows.append({"family": "confirmatory", "hypothesis": "H4", "analysis": "RCPS biological diagnostic failure rate", "endpoint": "diagnostic_failure", "contrast": "vs_zero", **h4_result})
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap["q_bh"] = benjamini_hochberg(bootstrap.p_two_sided.tolist())
    bootstrap.to_csv(output / "test_confirmatory_bootstrap.csv", index=False)

    decisions = config["hypothesis_decisions"]
    lookup = {(row.hypothesis, row.analysis): row for row in bootstrap.itertuples()}
    h1a = lookup[("H1a", "CNN-Raw test prediction asymmetry")]
    h1b = lookup[("H1b", "CNN-Raw Exact ISM asymmetry among prediction-consistent samples")]
    h3a = lookup[("H3a", "CNN-Aug residual Exact ISM asymmetry")]
    h2 = bootstrap[(bootstrap.hypothesis.eq("H2")) & bootstrap.contrast.eq("CNN-Aug_minus_CNN-Raw")].iloc[0]
    rcps_prediction_max = float(run_summary.loc[run_summary.model_type.eq("CNN-RCPS"), "rcps_prediction_max_abs_error"].max())
    rcps_ism_max = float(invariance.loc[invariance.model_type.eq("CNN-RCPS"), "exact_ism_aligned_matrix_max_abs_error"].max())
    decision_rows = [
        {"hypothesis": "H1a", "status": "reproduced" if h1a.ci95_low > float(decisions["h1a_raw_prediction_ci_lower_gt"]) else "not_reproduced", "evidence": f"CI lower={h1a.ci95_low:.6g}"},
        {"hypothesis": "H1b", "status": "reproduced" if h1b.ci95_low > float(decisions["h1b_raw_exact_ism_ci_lower_gt"]) else "not_reproduced", "evidence": f"CI lower={h1b.ci95_low:.6g}"},
        {"hypothesis": "H2", "status": "reproduced" if h2.ci95_high < float(decisions["h2_aug_minus_raw_prediction_ci_upper_lt"]) else "not_reproduced", "evidence": f"contrast CI upper={h2.ci95_high:.6g}"},
        {"hypothesis": "H3a", "status": "reproduced" if h3a.ci95_low > float(decisions["h3a_aug_exact_ism_ci_lower_gt"]) else "not_reproduced", "evidence": f"CI lower={h3a.ci95_low:.6g}"},
        {"hypothesis": "H3b", "status": "reproduced" if rcps_prediction_max <= float(decisions["h3b_prediction_max_le"]) and rcps_ism_max <= float(decisions["h3b_exact_ism_matrix_max_le"]) else "not_reproduced", "evidence": f"prediction max={rcps_prediction_max:.6g}; ISM max={rcps_ism_max:.6g}"},
        {"hypothesis": "H3c", "status": "exploratory_not_retested", "evidence": "Transformer excluded by frozen protocol"},
        {"hypothesis": "H4", "status": "reproduced_as_distinction" if h4_result["ci95_low"] > float(decisions["h4_diagnostic_ci_lower_gt"]) else "not_reproduced", "evidence": f"diagnostic rate={h4_result['estimate']:.6g}; CI=[{h4_result['ci95_low']:.6g},{h4_result['ci95_high']:.6g}]"},
    ]
    write_json(output / "test_hypothesis_decisions.json", decision_rows)

    motif_summary = attributions[attributions.label.eq(1) & attributions.strong_hit_080].groupby("model_type", observed=True).agg(n=("sample_id", "size"), motif_mass_fraction=("motif_mass_fraction", "mean"), top7_recall=("top7_motif_recall", "mean"), circular_enrichment=("motif_mass_enrichment_vs_circular", "mean")).reset_index()
    motif_summary.to_csv(output / "test_motif_localization_summary.csv", index=False)
    disruption_summary = disruptions.groupby("model_type", observed=True).agg(n=("sample_id", "size"), motif_logit_drop=("motif_logit_drop", "mean"), flank_logit_drop=("flank_logit_drop", "mean"), motif_minus_flank=("motif_minus_flank_logit_drop", "mean")).reset_index()
    disruption_summary.to_csv(output / "test_motif_disruption_summary.csv", index=False)
    create_figures(run_summary, attributions, disruptions, output)

    technical_gates = {
        "phase": "Phase 2F one-time sealed test",
        "protocol_revision": PROTOCOL_REVISION,
        "input_audit_passed": bool(audit["all_passed"]),
        "checkpoint_count": int(len(manifest["checkpoints"])),
        "expected_checkpoint_count": int(config["checkpoints"]["expected_count"]),
        "test_rows": int(config["dataset"]["expected_rows"]),
        "prediction_rows": int(len(predictions)),
        "expected_prediction_rows": expected_prediction_rows,
        "attribution_samples": int(config["attribution"]["expected_samples"]),
        "attribution_rows": int(len(attributions)),
        "expected_attribution_rows": expected_attribution_rows,
        "original_validation_rows_evaluated": 0,
        "test_unseal_authorized": True,
        "test_execution_count": 1,
        "rcps_prediction_gate_passed": bool(rcps_prediction_max <= float(config["invariance_gates"]["prediction_max_abs_tolerance"])),
        "rcps_exact_ism_gate_passed": bool(rcps_ism_max <= float(config["invariance_gates"]["exact_ism_aligned_max_abs_tolerance"])),
        "all_finite_primary_endpoints": bool(np.isfinite(run_summary[["auroc", "auprc", "prediction_mean_absolute_difference", "symmetry_flip_rate"]].to_numpy(float)).all() and np.isfinite(attributions.absolute_normalized_l1.to_numpy(float)).all()),
    }
    technical_gates["all_execution_gates_passed"] = bool(
        technical_gates["input_audit_passed"]
        and technical_gates["checkpoint_count"] == technical_gates["expected_checkpoint_count"]
        and technical_gates["prediction_rows"] == technical_gates["expected_prediction_rows"]
        and technical_gates["attribution_rows"] == technical_gates["expected_attribution_rows"]
        and technical_gates["rcps_prediction_gate_passed"]
        and technical_gates["rcps_exact_ism_gate_passed"]
        and technical_gates["all_finite_primary_endpoints"]
    )
    write_json(output / "phase2f_gate_assessment.json", technical_gates)
    completion = {
        "status": "completed",
        "protocol_revision": PROTOCOL_REVISION,
        "completed_at_utc": utc_now(),
        "test_execution_count": 1,
        "models": MODEL_ORDER,
        "all_execution_gates_passed": technical_gates["all_execution_gates_passed"],
        "post_test_tuning_allowed": False,
    }
    write_json(ROOT / config["execution"]["completion_path"], completion)
    update_ledger(config, status="completed_locked", completed_at_utc=completion["completed_at_utc"], completion_sha256=sha256_file(ROOT / config["execution"]["completion_path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2F one-time sealed-test evaluator")
    parser.add_argument("--config", default="configs/phase2f_one_time_test.yaml")
    parser.add_argument("--freeze-inputs", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    validate_frozen_config(config)
    torch.set_num_threads(int(config["cpu_threads"]))
    if args.freeze_inputs:
        manifest = build_input_manifest(config_path, config)
        print(json.dumps({"status": "frozen", "checkpoints": len(manifest["checkpoints"])}, indent=2))
        return
    completion = ROOT / config["execution"]["completion_path"]
    if completion.exists():
        raise RuntimeError("P2F is already complete; second execution forbidden")
    audit = verify_input_manifest(config_path, config)
    manifest = json.loads((ROOT / config["checkpoints"]["manifest_path"]).read_text(encoding="utf-8"))
    start_unseal_ledger(config_path, config, audit, args.resume)
    output = ROOT / config["execution"]["output_dir"]
    write_json(output / "input_audit.json", audit)
    shutil.copy2(config_path, output / "resolved_config.yaml")
    shutil.copy2(ROOT / config["protocol_path"], output / "frozen_protocol.md")
    full = pd.read_csv(ROOT / config["dataset"]["path"])
    test = validate_test_population(full, config)
    frozen = freeze_attribution_cohort(test, config, output)
    update_ledger(config, test_rows_loaded=True, observed_test_rows=int(len(test)), frozen_attribution_samples=int(len(frozen)))
    prediction_stage(test, config, manifest, output)
    attribution_stage(frozen, config, manifest, output)
    finalize(config, manifest, audit, output)
    print(json.dumps({"status": "completed_locked", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
