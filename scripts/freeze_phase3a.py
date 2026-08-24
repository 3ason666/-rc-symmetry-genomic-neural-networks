from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase3a_external_replication.yaml"
EXPECTED_PROTOCOL_REVISION = "phase3a_external_replication_selection_v2"
FORBIDDEN_PHASE2_EXPERIMENTS = {"ENCSR000EFT", "ENCSR000EWM", "ENCSR868FGK"}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_registry(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def validate_phase3a(config: dict[str, Any], registry: dict[str, Any], root: Path = ROOT) -> None:
    if config.get("protocol_revision") != EXPECTED_PROTOCOL_REVISION:
        raise ValueError("unexpected Phase 3A protocol revision")
    if config.get("decision", {}).get("p3_model_results_observed") is not False:
        raise ValueError("Phase 3 model results must be unobserved at freeze")
    if config.get("decision", {}).get("p3_training_allowed") is not False:
        raise ValueError("Phase 3 training must remain blocked during Phase 3A")
    boundary = config.get("phase2_boundary", {})
    if any(boundary.get(key) is not False for key in (
        "test_data_access_allowed", "test_prediction_access_allowed", "test_attribution_access_allowed", "p2f_rerun_allowed"
    )):
        raise ValueError("Phase 2F locked-test access must remain disabled")

    tasks = config.get("tasks", [])
    if len(tasks) != 2:
        raise ValueError("Phase 3A requires exactly two confirmatory tasks")
    roles = {task.get("role") for task in tasks}
    required_roles = {
        "primary_confirmatory_same_tf_different_cellular_environment",
        "supportive_generalization_different_tf_different_cellular_environment",
    }
    if roles != required_roles:
        raise ValueError("confirmatory task roles do not cover both frozen axes")
    if {task.get("target") for task in tasks} != {"GATA1", "CTCF"}:
        raise ValueError("frozen target family must be GATA1 plus CTCF")
    if config.get("common_dataset", {}).get("assembly") != "GRCh38":
        raise ValueError("Phase 3 confirmatory tasks must use GRCh38")
    if config.get("common_dataset", {}).get("sequence_length_bp") != 256:
        raise ValueError("sequence length must remain 256 bp")

    registry_tasks = registry.get("confirmatory_tasks", [])
    if {task.get("task_id") for task in registry_tasks} != {task.get("task_id") for task in tasks}:
        raise ValueError("config and candidate registry task IDs disagree")
    selected_accessions: set[str] = set()
    for task in registry_tasks:
        experiment = task.get("experiment", {})
        if experiment.get("status") != "released" or experiment.get("assembly") != "GRCh38":
            raise ValueError("selected experiments must be released GRCh38 resources")
        if experiment.get("accession") in FORBIDDEN_PHASE2_EXPERIMENTS:
            raise ValueError("Phase 2 development experiment selected for Phase 3")
        for item in task.get("files", []):
            accession = item.get("accession")
            if not accession or accession in selected_accessions:
                raise ValueError("selected file accessions must be present and unique")
            selected_accessions.add(accession)
            if item.get("status") != "released" or item.get("assembly") != "GRCh38":
                raise ValueError("all selected files must be released GRCh38 files")
            md5sum = item.get("md5sum", "")
            if len(md5sum) != 32 or any(char not in "0123456789abcdef" for char in md5sum):
                raise ValueError("every selected file needs a lowercase 32-character MD5")
            if not str(item.get("url", "")).startswith("https://www.encodeproject.org/files/"):
                raise ValueError("primary resource URLs must point to official ENCODE files")

    if set(registry.get("forbidden_phase2_development_resources", [])) != FORBIDDEN_PHASE2_EXPERIMENTS:
        raise ValueError("forbidden Phase 2 resource registry changed")
    if config.get("models", {}).get("model_types") != ["CNN-Raw", "CNN-Aug", "CNN-RCPS"]:
        raise ValueError("Phase 3 model family must remain frozen")
    if config.get("models", {}).get("seeds") != [42, 123, 2026]:
        raise ValueError("Phase 3 seeds must remain frozen")

    split = config.get("split", {})
    if split.get("shared_fixed_test_across_folds") is not True:
        raise ValueError("all Phase 3 folds must share one permanently sealed test set")
    if split.get("development_folds") != 3 or split.get("fold_scope") != "rotate_train_and_validation_only":
        raise ValueError("Phase 3 must use three development-only train/validation folds")
    if "early_stopping" not in split.get("test_forbidden_uses", []):
        raise ValueError("sealed test set must be forbidden for early stopping")

    gates = config.get("quality_gates", {})
    expected_gates = {
        "minimum_retained_positive_per_task": 1500,
        "minimum_train_per_class": 1000,
        "minimum_validation_per_class": 200,
        "minimum_test_per_class": 200,
        "minimum_h1b_prediction_consistent_test_subgroup": 100,
        "primary_matched_negative_rate_min": 0.90,
        "sensitivity_matched_negative_rate_min": 0.80,
        "fail_if_matched_negative_rate_below": 0.80,
    }
    for key, expected in expected_gates.items():
        if gates.get(key) != expected:
            raise ValueError(f"unexpected Phase 3 quality gate: {key}")
    if gates.get("gc_balance", {}).get("ks_d_max") != 0.10:
        raise ValueError("GC KS balance gate must remain 0.10")
    if gates.get("accessibility_balance", {}).get("ks_d_max") != 0.10:
        raise ValueError("accessibility KS balance gate must remain 0.10")

    leakage = config.get("leakage_controls", {})
    near = leakage.get("near_duplicate_primary", {})
    if near.get("sequence_identity_min") != 0.90 or near.get("bidirectional_coverage_min") != 0.90:
        raise ValueError("primary near-duplicate rule must remain 90/90")
    if near.get("reverse_complement_aware") is not True:
        raise ValueError("near-duplicate audit must remain RC-aware")
    if leakage.get("cross_split_collision_keep_priority") != ["test", "validation", "train"]:
        raise ValueError("cross-split near-duplicate priority must protect test then validation")

    framework = config.get("success_framework", {})
    if framework.get("primary_task") != "p3_gata1_fetal":
        raise ValueError("fetal GATA1 must remain the primary confirmatory task")
    if framework.get("generalization_task") != "p3_ctcf_gm12878":
        raise ValueError("GM12878 CTCF must remain the supportive generalization task")
    if framework.get("folds_are_independent_biological_replicates") is not False:
        raise ValueError("folds cannot be treated as independent biological replicates")
    if framework.get("fold_direction_stability_role") != "robustness_only_not_success_vote":
        raise ValueError("fold direction agreement cannot be a success vote")
    if framework.get("full_replication_requires") != ["H1a", "H1b", "H2", "H3a", "H3b"]:
        raise ValueError("full-replication hypothesis family changed")

    cross_phase = config.get("post_primary_cross_phase_audit", {})
    if cross_phase.get("enabled_before_primary_results_frozen") is not False:
        raise ValueError("P2F/P3 overlap audit cannot run before P3 primary results freeze")
    if cross_phase.get("p2f_predictions_or_attributions_allowed") is not False:
        raise ValueError("post-primary overlap audit cannot access P2F predictions or attributions")
    if cross_phase.get("sensitivity_results_replace_primary") is not False:
        raise ValueError("cross-phase sensitivity results cannot replace the primary analysis")

    completion_path = _resolve(root, boundary["completion_path"])
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("status") != boundary.get("required_status"):
        raise ValueError("Phase 2F completion lock is not in the required state")


def freeze(config_path: Path = DEFAULT_CONFIG, root: Path = ROOT) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    registry_path = _resolve(root, config["candidate_registry_path"])
    protocol_path = _resolve(root, config["protocol_path"])
    registry = load_registry(registry_path)
    validate_phase3a(config, registry, root=root)

    output_dir = _resolve(root, config["execution"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = _resolve(root, config["phase2_boundary"]["completion_path"])
    selected_files = [
        item["accession"]
        for task in registry["confirmatory_tasks"]
        for item in task["files"]
    ]
    manifest = {
        "status": "phase3a_selection_frozen",
        "protocol_revision": config["protocol_revision"],
        "freeze_date": config["freeze_date"],
        "results_observed_before_freeze": False,
        "training_allowed": False,
        "confirmatory_task_ids": [task["task_id"] for task in config["tasks"]],
        "selected_file_accessions": selected_files,
        "sha256": {
            "config": sha256_path(config_path),
            "protocol": sha256_path(protocol_path),
            "candidate_registry": sha256_path(registry_path),
            "phase2f_completion_lock": sha256_path(completion_path),
        },
        "next_gate": "download_md5_qc_dataset_construction_and_split_freeze",
    }
    gate = {
        "status": "pass_selection_freeze_training_still_blocked",
        "checks": {
            "p2f_lock_verified_without_test_data_access": True,
            "same_tf_external_task_selected": True,
            "different_tf_external_task_selected": True,
            "all_selected_files_released_grch38": True,
            "all_selected_files_have_md5": True,
            "no_phase3_model_results_observed": True,
            "primary_task_and_generalization_roles_frozen": True,
            "single_permanent_test_policy_frozen": True,
            "attainable_sample_and_balance_gates_frozen": True,
            "near_duplicate_and_post_primary_overlap_rules_frozen": True,
            "model_training_allowed": False,
        },
        "known_risks": [
            "Fetal erythroblast DNase resources are independent unreplicated profiles and not the ChIP biosample.",
            "Both ChIP source experiments carry historical ENCODE audit warnings; selected processed files are released current-pipeline GRCh38 outputs.",
            "Five of seven selected BED files are still pending download and structural QC.",
            "The frozen near-duplicate implementation and CTCF performance/power-precision manifest must be completed before dataset split and training release.",
        ],
    }
    manifest_path = _resolve(root, config["execution"]["freeze_manifest_path"])
    gate_path = _resolve(root, config["execution"]["gate_assessment_path"])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    gate_path.write_text(json.dumps(gate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and freeze Phase 3A external resources.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    manifest = freeze(args.config)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
