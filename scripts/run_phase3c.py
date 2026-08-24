from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))

from scripts.run_phase2e import benjamini_hochberg
from scripts.run_phase2f import (
    aggregate_attributions,
    aggregate_predictions,
    attribution_stage,
    bootstrap_endpoint,
    create_figures,
    freeze_attribution_cohort,
    paired_difference,
    prediction_stage,
    sha256_file,
    write_json,
)


PROTOCOL_REVISION = "phase3c_one_time_test_v1"
CONFIG_PATH = ROOT / "configs" / "phase3c_one_time_test.yaml"
MODEL_ORDER = ["CNN-Raw", "CNN-Aug", "CNN-RCPS"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_config(config: dict) -> None:
    if config.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("unexpected Phase 3C protocol revision")
    if config.get("device") != "cpu":
        raise ValueError("Phase 3C v1 is frozen to CPU")
    training = config["training"]
    if training.get("model_types") != MODEL_ORDER:
        raise ValueError("model family changed")
    if training.get("folds") != [1, 2, 3] or training.get("seeds") != [42, 123, 2026]:
        raise ValueError("fold or seed grid changed")
    if int(training.get("expected_total_count")) != 54 or training.get("retraining_allowed"):
        raise ValueError("checkpoint inventory or retraining rule changed")
    tasks = config["tasks"]
    if set(tasks) != {"p3_gata1_fetal", "p3_ctcf_gm12878"}:
        raise ValueError("task family changed")
    if tasks["p3_gata1_fetal"]["role"] != "primary_confirmatory":
        raise ValueError("fetal GATA1 must remain the primary task")
    if tasks["p3_ctcf_gm12878"]["role"] != "supportive_generalization":
        raise ValueError("CTCF must remain supportive generalization")
    attr = config["attribution"]
    if attr.get("method") != "exact_ism" or attr.get("difference") != "logit":
        raise ValueError("primary attribution method changed")
    if int(attr.get("h1b_target_samples")) != 100:
        raise ValueError("H1b target cohort changed")
    if float(attr.get("prediction_consistent_max_abs_difference")) != 0.01:
        raise ValueError("prediction-consistency threshold changed")
    decision = config["decision"]
    if not decision.get("test_unseal_authorized"):
        raise ValueError("test unsealing is not authorized")
    if any(decision.get(key) for key in (
        "change_protocol_after_unseal", "tune_after_test", "select_models_after_test",
        "post_test_retraining_allowed",
    )):
        raise ValueError("forbidden post-test change is enabled")
    if config["execution"].get("allow_force_rerun"):
        raise ValueError("force rerun must remain disabled")


def checkpoint_records(config: dict) -> list[dict]:
    source = ROOT / config["training"]["source_results"] / "runs"
    records: list[dict] = []
    for summary_path in sorted(source.glob("*/run_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        checkpoint = summary_path.parent / "best_checkpoint.pt"
        history = summary_path.parent / "training_history.csv"
        if not checkpoint.exists() or not history.exists():
            raise FileNotFoundError(f"incomplete run directory: {summary_path.parent}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved.get("test_evaluated") is not False:
            raise ValueError(f"checkpoint records test evaluation: {checkpoint}")
        if summary.get("status") != "completed" or summary.get("test_evaluated") is not False:
            raise ValueError(f"invalid training summary: {summary_path}")
        if float(summary.get("checkpoint_reload_max_abs_error", np.inf)) != 0.0:
            raise ValueError(f"checkpoint reload invariance failed: {summary_path}")
        if sha256_file(checkpoint) != summary.get("checkpoint_sha256"):
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        if sha256_file(history) != summary.get("history_sha256"):
            raise ValueError(f"training-history hash mismatch: {history}")
        records.append({
            "run_name": summary_path.parent.name,
            "task_id": str(summary["task_id"]),
            "fold_id": f"fold_{int(summary['fold'])}",
            "fold": int(summary["fold"]),
            "model_type": str(summary["model_type"]),
            "seed": int(summary["seed"]),
            "checkpoint_path": checkpoint.relative_to(ROOT).as_posix(),
            "checkpoint_size": int(checkpoint.stat().st_size),
            "checkpoint_sha256": sha256_file(checkpoint),
            "history_path": history.relative_to(ROOT).as_posix(),
            "history_sha256": sha256_file(history),
            "run_summary_path": summary_path.relative_to(ROOT).as_posix(),
            "run_summary_sha256": sha256_file(summary_path),
            "validation_auroc": float(summary["validation_metrics"]["auroc"]),
            "test_evaluated": False,
            "checkpoint_reload_max_abs_error": 0.0,
        })
    expected = {
        (task, fold, model, seed)
        for task in config["tasks"]
        for fold in config["training"]["folds"]
        for model in config["training"]["model_types"]
        for seed in config["training"]["seeds"]
    }
    observed = {(r["task_id"], r["fold"], r["model_type"], r["seed"]) for r in records}
    if observed != expected or len(records) != int(config["training"]["expected_total_count"]):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"checkpoint grid incomplete; missing={missing}; extra={extra}")
    return sorted(records, key=lambda r: (r["task_id"], r["fold"], r["model_type"], r["seed"]))


def resource_record(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "size": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def build_input_manifest(config_path: Path, config: dict) -> dict:
    completion = ROOT / config["training"]["completion_path"]
    completion_data = json.loads(completion.read_text(encoding="utf-8"))
    if completion_data.get("status") != "completed" or completion_data.get("test_evaluated") is not False:
        raise ValueError("P3B completion lock is invalid")
    training_manifest = ROOT / config["training"]["training_manifest_path"]
    training_manifest_data = json.loads(training_manifest.read_text(encoding="utf-8"))
    if training_manifest_data.get("test_unsealing_allowed") is not False:
        raise ValueError("P3B training manifest no longer records a sealed test")
    resources = {
        "config": resource_record(config_path),
        "protocol": resource_record(ROOT / config["protocol_path"]),
        "runner": resource_record(Path(__file__).resolve()),
        "phase3b_completion": resource_record(completion),
        "phase3b_training_manifest": resource_record(training_manifest),
        "phase3b_training_summary": resource_record(ROOT / config["training"]["source_results"] / "training_run_summary.json"),
    }
    for task_id, task in config["tasks"].items():
        pwm = ROOT / task["pwm_resource"]
        resources[f"{task_id}_pwm"] = resource_record(pwm)
        test_path = ROOT / task["sealed_test_path"]
        record = resource_record(test_path)
        if record["sha256"].lower() != str(task["sealed_test_sha256"]).lower():
            raise ValueError(f"sealed test byte hash changed: {task_id}")
        resources[f"{task_id}_sealed_test_bytes"] = record
    manifest = {
        "schema_version": 1,
        "protocol_revision": PROTOCOL_REVISION,
        "created_before_semantic_test_rows_loaded": True,
        "test_predictions_observed_before_freeze": False,
        "test_attributions_observed_before_freeze": False,
        "resources": resources,
        "checkpoints": checkpoint_records(config),
    }
    path = ROOT / config["training"]["checkpoint_manifest_path"]
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("existing Phase 3C input manifest differs from frozen inputs")
    else:
        write_json(path, manifest)
    return manifest


def verify_input_manifest(config_path: Path, config: dict) -> dict:
    path = ROOT / config["training"]["checkpoint_manifest_path"]
    if not path.exists():
        raise FileNotFoundError("run --freeze-inputs before test unsealing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("protocol_revision") != PROTOCOL_REVISION:
        raise ValueError("input manifest protocol changed")
    checks = []
    for name, record in manifest["resources"].items():
        observed = sha256_file(ROOT / record["path"])
        checks.append({"resource": name, "expected": record["sha256"], "observed": observed, "passed": observed == record["sha256"]})
    for record in manifest["checkpoints"]:
        for kind in ("checkpoint", "history", "run_summary"):
            observed = sha256_file(ROOT / record[f"{kind}_path"])
            expected = record[f"{kind}_sha256"]
            checks.append({"resource": f"{record['run_name']}:{kind}", "expected": expected, "observed": observed, "passed": observed == expected})
    if not all(row["passed"] for row in checks):
        raise ValueError("Phase 3C frozen-input checksum audit failed")
    return {
        "manifest_path": path.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(path),
        "checks": checks,
        "all_passed": True,
    }


def start_unseal_ledger(config_path: Path, config: dict, audit: dict, resume: bool) -> dict:
    output = ROOT / config["execution"]["output_dir"]
    ledger_path = ROOT / config["execution"]["ledger_path"]
    completion_path = ROOT / config["execution"]["completion_path"]
    if completion_path.exists():
        raise RuntimeError("Phase 3C is already complete; a second test execution is forbidden")
    fingerprints = {
        "config_sha256": sha256_file(config_path),
        "manifest_sha256": audit["manifest_sha256"],
        "protocol_revision": PROTOCOL_REVISION,
    }
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError("test was already unsealed; use --resume for the identical interrupted run")
        if ledger.get("fingerprints") != fingerprints:
            raise ValueError("resume fingerprints changed")
        ledger["resume_count"] = int(ledger.get("resume_count", 0)) + 1
        ledger["last_resumed_at_utc"] = utc_now()
        ledger["status"] = "resuming_same_frozen_run"
    else:
        if resume:
            raise RuntimeError("no interrupted Phase 3C ledger exists")
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("Phase 3C output is non-empty without an unseal ledger")
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


def update_ledger(config: dict, **updates) -> None:
    path = ROOT / config["execution"]["ledger_path"]
    ledger = json.loads(path.read_text(encoding="utf-8"))
    ledger.update(updates)
    write_json(path, ledger)


def task_runtime_config(config: dict, task_id: str, *, h1b: bool = False) -> dict:
    task = config["tasks"][task_id]
    if h1b:
        per = 1
        expected = int(config["attribution"]["h1b_target_samples"])
        salt = str(config["attribution"]["h1b_selection_salt"])
    else:
        per = int(task["base_attribution_per_chromosome_per_label"])
        expected = int(task["base_attribution_expected_samples"])
        salt = str(config["attribution"]["base_selection_salt"])
    return {
        "dataset": {
            "chromosomes": task["chromosomes"],
            "expected_rows": int(task["expected_rows"]),
            "expected_per_label": int(task["expected_per_label"]),
        },
        "prediction": config["prediction"],
        "attribution": {
            "per_chromosome_per_label": per,
            "expected_samples": expected,
            "selection_salt": salt,
            "prediction_consistent_max_abs_difference": float(config["attribution"]["prediction_consistent_max_abs_difference"]),
            "top_k": int(config["attribution"]["top_k"]),
            "ism_batch_size": int(config["attribution"]["ism_batch_size"]),
        },
        "biological": {
            "pwm_resource": task["pwm_resource"],
            **config["biological"],
        },
        "invariance_gates": config["invariance_gates"],
        "statistics": config["statistics"],
        "execution": config["execution"],
    }


def validate_test_population(frame: pd.DataFrame, task_id: str, config: dict) -> pd.DataFrame:
    task = config["tasks"][task_id]
    required = {"task_id", "sample_id", "pair_id", "label", "chromosome", "sequence", "canonical_key", "fixed_partition"}
    if not required.issubset(frame.columns):
        raise ValueError(f"sealed test schema changed: {task_id}")
    if set(frame["task_id"].astype(str)) != {task_id} or set(frame["fixed_partition"].astype(str)) != {"test"}:
        raise ValueError(f"sealed test identity changed: {task_id}")
    if len(frame) != int(task["expected_rows"]):
        raise ValueError(f"sealed test row count changed: {task_id}")
    if set(frame["chromosome"].astype(str)) != set(task["chromosomes"]):
        raise ValueError(f"sealed test chromosomes changed: {task_id}")
    counts = frame.groupby("label").size().to_dict()
    expected = int(task["expected_per_label"])
    if counts != {0: expected, 1: expected}:
        raise ValueError(f"sealed test label balance changed: {task_id}")
    if frame.sample_id.duplicated().any() or frame.canonical_key.duplicated().any():
        raise ValueError(f"sealed test identifiers are not unique: {task_id}")
    return frame.sort_values(["chromosome", "label", "sample_id"], kind="mergesort").reset_index(drop=True)


def selection_digest(row: pd.Series, salt: str) -> str:
    value = f"{salt}|{row.chromosome}|{int(row.label)}|{row.sample_id}|{row.canonical_key}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def freeze_h1b_cohort(test: pd.DataFrame, predictions: pd.DataFrame, config: dict, task_output: Path) -> tuple[pd.DataFrame, int]:
    raw = predictions[predictions.model_type.eq("CNN-Raw")]
    ensemble = raw.groupby("sample_id", observed=True)[["p_forward", "p_rc"]].mean().reset_index()
    ensemble["ensemble_prediction_difference"] = np.abs(ensemble.p_forward - ensemble.p_rc)
    threshold = float(config["attribution"]["prediction_consistent_max_abs_difference"])
    eligible_ids = set(ensemble.loc[ensemble.ensemble_prediction_difference <= threshold, "sample_id"].astype(str))
    eligible = test[test.sample_id.astype(str).isin(eligible_ids)].copy()
    pool_size = int(len(eligible))
    target = int(config["attribution"]["h1b_target_samples"])
    eligible["selection_digest"] = eligible.apply(selection_digest, axis=1, salt=str(config["attribution"]["h1b_selection_salt"]))
    groups = {
        key: group.sort_values("selection_digest", kind="mergesort").to_dict("records")
        for key, group in eligible.groupby(["chromosome", "label"], observed=True)
    }
    selected: list[dict] = []
    keys = sorted(groups, key=lambda key: (str(key[0]), int(key[1])))
    while len(selected) < min(target, pool_size):
        changed = False
        for key in keys:
            if groups[key] and len(selected) < min(target, pool_size):
                selected.append(groups[key].pop(0))
                changed = True
        if not changed:
            break
    cohort = pd.DataFrame(selected)
    columns = ["sample_id", "pair_id", "canonical_key", "chromosome", "label", "sequence", "selection_digest"]
    cohort = cohort[columns] if len(cohort) else pd.DataFrame(columns=columns)
    path = task_output / "h1b" / "frozen_h1b_attribution_samples.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        pd.testing.assert_frame_equal(existing, cohort, check_dtype=False)
    else:
        cohort.to_csv(path, index=False)
    write_json(task_output / "h1b" / "h1b_cohort_gate.json", {
        "prediction_consistency_threshold": threshold,
        "eligible_pool_size": pool_size,
        "target_samples": target,
        "selected_samples": int(len(cohort)),
        "gate_passed": pool_size >= target and len(cohort) == target,
    })
    return cohort, pool_size


def prepare_h1b_predictions(base_output: Path, h1b_output: Path, cohort: pd.DataFrame, records: list[dict]) -> None:
    ids = set(cohort.sample_id.astype(str))
    for record in records:
        if record["model_type"] != "CNN-Raw":
            continue
        source = base_output / "runs" / record["run_name"] / "test_prediction_pairs.csv"
        target_dir = h1b_output / "runs" / record["run_name"]
        target_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(source)
        selected = frame[frame.sample_id.astype(str).isin(ids)].copy()
        if len(selected) != len(cohort):
            raise ValueError(f"H1b prediction cohort incomplete: {record['run_name']}")
        selected.to_csv(target_dir / "test_prediction_pairs.csv", index=False)


def ensemble_summary(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    detail = predictions.groupby(["model_type", "sample_id", "chromosome", "label"], observed=True)[["p_forward", "p_rc"]].mean().reset_index()
    for model in MODEL_ORDER:
        group = detail[detail.model_type.eq(model)]
        difference = np.abs(group.p_forward.to_numpy(float) - group.p_rc.to_numpy(float))
        flip = (group.p_forward.to_numpy(float) >= threshold) != (group.p_rc.to_numpy(float) >= threshold)
        rows.append({
            "model_type": model,
            "n": int(len(group)),
            "auroc": float(roc_auc_score(group.label, group.p_forward)),
            "auprc": float(average_precision_score(group.label, group.p_forward)),
            "prediction_mean_absolute_difference": float(np.mean(difference)),
            "prediction_median_absolute_difference": float(np.median(difference)),
            "prediction_p95_absolute_difference": float(np.quantile(difference, 0.95)),
            "symmetry_flip_rate": float(np.mean(flip)),
        })
    return pd.DataFrame(rows)


def bootstrap_or_none(frame: pd.DataFrame, column: str, cfg: dict, offset: int) -> dict | None:
    if frame.empty or frame[column].isna().all():
        return None
    return bootstrap_endpoint(frame, column, cfg, offset)


def finalize_task(task_id: str, config: dict, records: list[dict], task_output: Path, h1b_pool_size: int) -> dict:
    base_cfg = task_runtime_config(config, task_id)
    predictions, run_summary = aggregate_predictions(task_output)
    base_attr, disruptions, invariance = aggregate_attributions(task_output)
    h1b_paths = sorted((task_output / "h1b" / "runs").glob("*/test_attribution_results.csv"))
    h1b_attr = pd.concat([pd.read_csv(path) for path in h1b_paths], ignore_index=True) if h1b_paths else pd.DataFrame()
    ensemble = ensemble_summary(predictions, float(config["prediction"]["threshold"]))
    ensemble.to_csv(task_output / "test_prediction_ensemble_summary.csv", index=False)

    h1a = bootstrap_endpoint(predictions[predictions.model_type.eq("CNN-Raw")], "prediction_difference", base_cfg, 100)
    h2_pairs = paired_difference(predictions, "prediction_difference", "CNN-Aug", "CNN-Raw")
    h2 = bootstrap_endpoint(h2_pairs, "difference", base_cfg, 200)
    h3a = bootstrap_endpoint(base_attr[base_attr.model_type.eq("CNN-Aug")], "absolute_normalized_l1", base_cfg, 102)
    h1b = bootstrap_or_none(h1b_attr[h1b_attr.model_type.eq("CNN-Raw")] if not h1b_attr.empty else h1b_attr, "absolute_normalized_l1", base_cfg, 101)

    rcps_prediction_max = float(run_summary.loc[run_summary.model_type.eq("CNN-RCPS"), "rcps_prediction_max_abs_error"].max())
    rcps_ism_max = float(invariance.loc[invariance.model_type.eq("CNN-RCPS"), "exact_ism_aligned_matrix_max_abs_error"].max())

    rcps_attr = base_attr[base_attr.model_type.eq("CNN-RCPS") & base_attr.label.eq(1) & base_attr.strong_hit_080][["fold_id", "seed", "sample_id", "top7_motif_recall"]]
    rcps_disruption = disruptions[disruptions.model_type.eq("CNN-RCPS")][["fold_id", "seed", "sample_id", "motif_minus_flank_logit_drop"]]
    h4_data = rcps_attr.merge(rcps_disruption, on=["fold_id", "seed", "sample_id"], validate="one_to_one")
    h4_data["diagnostic_failure"] = ((h4_data.top7_motif_recall < float(config["biological"]["low_recall_threshold"])) | (h4_data.motif_minus_flank_logit_drop <= 0)).astype(float)
    h4 = bootstrap_or_none(h4_data, "diagnostic_failure", base_cfg, 300)

    rows = []
    for hypothesis, analysis, endpoint, result in (
        ("H1a", "CNN-Raw prediction asymmetry", "prediction_difference", h1a),
        ("H1b", "CNN-Raw Exact ISM asymmetry in frozen prediction-consistent cohort", "absolute_normalized_l1", h1b),
        ("H2", "CNN-Aug minus CNN-Raw paired prediction asymmetry", "difference", h2),
        ("H3a", "CNN-Aug residual Exact ISM asymmetry", "absolute_normalized_l1", h3a),
        ("H4", "RCPS biological diagnostic failure rate", "diagnostic_failure", h4),
    ):
        rows.append({"task_id": task_id, "hypothesis": hypothesis, "analysis": analysis, "endpoint": endpoint, **(result or {"n": 0, "estimate": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_two_sided": np.nan})})
    bootstrap = pd.DataFrame(rows)
    finite_p = bootstrap.p_two_sided.fillna(1.0).tolist()
    bootstrap["q_bh"] = benjamini_hochberg(finite_p)
    bootstrap.to_csv(task_output / "test_confirmatory_bootstrap.csv", index=False)

    d = config["hypothesis_decisions"]
    width_ok = bool(((bootstrap.ci95_high - bootstrap.ci95_low).dropna() <= float(config["interpretability_gates"]["maximum_primary_ci_width"])).all())
    h1b_gate = h1b_pool_size >= int(config["interpretability_gates"]["minimum_h1b_prediction_consistent_samples"]) and h1b is not None
    decision_rows = [
        {"hypothesis": "H1a", "status": "reproduced" if h1a["ci95_low"] > float(d["h1a_raw_prediction_ci_lower_gt"]) else "not_reproduced", "evidence": f"CI lower={h1a['ci95_low']:.6g}"},
        {"hypothesis": "H1b", "status": "reproduced" if h1b_gate and h1b["ci95_low"] > float(d["h1b_raw_exact_ism_ci_lower_gt"]) else ("not_reproduced" if h1b_gate else "not_estimable"), "evidence": f"eligible pool={h1b_pool_size}; CI lower={h1b['ci95_low']:.6g}" if h1b else f"eligible pool={h1b_pool_size}"},
        {"hypothesis": "H2", "status": "reproduced" if h2["ci95_high"] < float(d["h2_aug_minus_raw_prediction_ci_upper_lt"]) else "not_reproduced", "evidence": f"contrast CI upper={h2['ci95_high']:.6g}"},
        {"hypothesis": "H3a", "status": "reproduced" if h3a["ci95_low"] > float(d["h3a_aug_exact_ism_ci_lower_gt"]) else "not_reproduced", "evidence": f"CI lower={h3a['ci95_low']:.6g}"},
        {"hypothesis": "H3b", "status": "reproduced" if rcps_prediction_max <= float(d["h3b_prediction_max_le"]) and rcps_ism_max <= float(d["h3b_exact_ism_matrix_max_le"]) else "not_reproduced", "evidence": f"prediction max={rcps_prediction_max:.6g}; ISM max={rcps_ism_max:.6g}"},
        {"hypothesis": "H3c", "status": "exploratory_not_retested", "evidence": "Transformer excluded from confirmatory Phase 3C"},
        {"hypothesis": "H4", "status": "diagnostic_distinction_observed" if h4 and h4["ci95_low"] > float(d["h4_diagnostic_ci_lower_gt"]) else ("not_observed" if h4 else "not_estimable"), "evidence": f"CI=[{h4['ci95_low']:.6g},{h4['ci95_high']:.6g}]" if h4 else "no strong-motif diagnostic cohort"},
    ]
    write_json(task_output / "test_hypothesis_decisions.json", decision_rows)

    validation = pd.DataFrame(records)
    validation_median = float(validation.validation_auroc.median())
    test_min = float(ensemble.auroc.min())
    gates = {
        "task_id": task_id,
        "checkpoint_count": int(len(records)),
        "expected_checkpoint_count": int(config["training"]["expected_count_per_task"]),
        "h1b_eligible_pool_size": int(h1b_pool_size),
        "h1b_sample_gate_passed": bool(h1b_gate),
        "validation_median_auroc": validation_median,
        "validation_performance_gate_passed": validation_median >= float(config["interpretability_gates"]["minimum_validation_median_auroc"]),
        "minimum_test_ensemble_auroc": test_min,
        "test_performance_gate_passed": test_min >= float(config["interpretability_gates"]["minimum_test_ensemble_auroc"]),
        "primary_ci_width_gate_passed": width_ok,
        "rcps_prediction_gate_passed": rcps_prediction_max <= float(config["invariance_gates"]["prediction_max_abs_tolerance"]),
        "rcps_exact_ism_gate_passed": rcps_ism_max <= float(config["invariance_gates"]["exact_ism_aligned_max_abs_tolerance"]),
    }
    gates["all_interpretability_gates_passed"] = bool(all([
        gates["checkpoint_count"] == gates["expected_checkpoint_count"],
        gates["h1b_sample_gate_passed"],
        gates["validation_performance_gate_passed"],
        gates["test_performance_gate_passed"],
        gates["primary_ci_width_gate_passed"],
        gates["rcps_prediction_gate_passed"],
        gates["rcps_exact_ism_gate_passed"],
    ]))
    write_json(task_output / "phase3c_task_gate_assessment.json", gates)
    if not disruptions.empty:
        create_figures(run_summary, base_attr, disruptions, task_output)
    return {"task_id": task_id, "decisions": decision_rows, "gates": gates}


def execute(config_path: Path, config: dict, manifest: dict, audit: dict, resume: bool) -> None:
    start_unseal_ledger(config_path, config, audit, resume)
    output = ROOT / config["execution"]["output_dir"]
    write_json(output / "input_audit.json", audit)
    shutil.copy2(config_path, output / "resolved_config.yaml")
    shutil.copy2(ROOT / config["protocol_path"], output / "frozen_protocol.md")
    task_results = []
    loaded_rows = {}
    for task_id, task in config["tasks"].items():
        task_output = output / task_id
        task_output.mkdir(parents=True, exist_ok=True)
        test = validate_test_population(pd.read_csv(ROOT / task["sealed_test_path"]), task_id, config)
        loaded_rows[task_id] = int(len(test))
        records = [record for record in manifest["checkpoints"] if record["task_id"] == task_id]
        base_cfg = task_runtime_config(config, task_id)
        prediction_stage(test, base_cfg, {"checkpoints": records}, task_output)
        predictions, _ = aggregate_predictions(task_output)
        base = freeze_attribution_cohort(test, base_cfg, task_output)
        h1b, h1b_pool = freeze_h1b_cohort(test, predictions, config, task_output)
        attribution_stage(base, base_cfg, {"checkpoints": records}, task_output)
        if len(h1b) == int(config["attribution"]["h1b_target_samples"]):
            h1b_output = task_output / "h1b"
            prepare_h1b_predictions(task_output, h1b_output, h1b, records)
            raw_records = [record for record in records if record["model_type"] == "CNN-Raw"]
            attribution_stage(h1b, task_runtime_config(config, task_id, h1b=True), {"checkpoints": raw_records}, h1b_output)
        task_results.append(finalize_task(task_id, config, records, task_output, h1b_pool))
    update_ledger(config, test_rows_loaded=True, observed_test_rows=loaded_rows, test_predictions_generated=True, test_attributions_generated=True)

    primary = next(result for result in task_results if result["task_id"] == "p3_gata1_fetal")
    supportive = next(result for result in task_results if result["task_id"] == "p3_ctcf_gm12878")
    primary_lookup = {row["hypothesis"]: row["status"] for row in primary["decisions"]}
    required = ["H1a", "H1b", "H2", "H3a", "H3b"]
    primary_full = all(primary_lookup.get(h) == "reproduced" for h in required)
    overall = {
        "primary_task": "p3_gata1_fetal",
        "primary_full_replication": primary_full,
        "primary_hypothesis_status": {h: primary_lookup.get(h) for h in required},
        "supportive_task": "p3_ctcf_gm12878",
        "supportive_interpretability_gates_passed": supportive["gates"]["all_interpretability_gates_passed"],
        "ctcf_non_support_interpretation": "possible_functional_boundary" if supportive["gates"]["all_interpretability_gates_passed"] else "inconclusive",
        "h4_excluded_from_composite_success": True,
        "transformer_excluded_from_confirmatory_success": True,
    }
    write_json(output / "phase3c_overall_decision.json", overall)
    completion = {
        "status": "completed",
        "protocol_revision": PROTOCOL_REVISION,
        "completed_at_utc": utc_now(),
        "test_execution_count": 1,
        "checkpoint_count": int(len(manifest["checkpoints"])),
        "post_test_tuning_allowed": False,
        "post_test_retraining_allowed": False,
    }
    write_json(ROOT / config["execution"]["completion_path"], completion)
    update_ledger(config, status="completed_locked", completed_at_utc=completion["completed_at_utc"], completion_sha256=sha256_file(ROOT / config["execution"]["completion_path"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3C one-time sealed-test evaluator")
    parser.add_argument("--config", default="configs/phase3c_one_time_test.yaml")
    parser.add_argument("--freeze-inputs", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = load_config(config_path)
    validate_config(config)
    torch.set_num_threads(int(config["cpu_threads"]))
    if args.freeze_inputs:
        manifest = build_input_manifest(config_path, config)
        print(json.dumps({"status": "frozen", "checkpoints": len(manifest["checkpoints"])}, indent=2))
        return
    audit = verify_input_manifest(config_path, config)
    if args.preflight:
        print(json.dumps({"status": "preflight_passed", "checks": len(audit["checks"]), "test_rows_loaded": False}, indent=2))
        return
    manifest = json.loads((ROOT / config["training"]["checkpoint_manifest_path"]).read_text(encoding="utf-8"))
    execute(config_path, config, manifest, audit, args.resume)
    print(json.dumps({"status": "completed_locked", "output": config["execution"]["output_dir"]}, indent=2))


if __name__ == "__main__":
    main()
