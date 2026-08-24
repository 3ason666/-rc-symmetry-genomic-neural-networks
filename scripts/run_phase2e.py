from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.run_phase2d import MODEL_COLORS, MODEL_ORDER, load_checkpoint, run_dir_name
from src.dna_utils import batch_one_hot_encode, reverse_complement
from src.metrics import prediction_consistency_metrics
from src.training import DEVICE, predict_sequences


DNA = "ACGT"
BASE_INDEX = {base: index for index, base in enumerate(DNA)}
COMPLEMENT = str.maketrans("ACGT", "TGCA")
CONFIRMATORY_CONTRASTS = [("CNN-Aug", "CNN-Raw"), ("CNN-RCPS", "CNN-Aug")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_frozen_config(config: dict) -> None:
    if config.get("protocol_revision") != "phase2e_biological_correctness_v1":
        raise ValueError("P2E protocol revision is not frozen v1")
    inputs = config["inputs"]
    decision = config["decision"]
    if inputs.get("allowed_source_split") != "train":
        raise ValueError("P2E source split must remain train")
    if inputs.get("forbidden_splits") != ["validation", "test"]:
        raise ValueError("P2E forbidden split boundary changed")
    if inputs.get("test_policy") != "sealed_no_model_access":
        raise ValueError("P2E test policy changed")
    if decision.get("access_original_validation") or decision.get("unseal_test_now"):
        raise ValueError("P2E config requests forbidden split access")
    if decision.get("allow_post_result_changes") or decision.get("phase2f_authorized"):
        raise ValueError("P2E config permits a post-result or test action")
    models = config["models"]
    if models.get("model_types") != MODEL_ORDER:
        raise ValueError("P2E model order changed")
    if models.get("seeds") != [42, 123, 2026]:
        raise ValueError("P2E seed set changed")
    if models.get("folds") != ["fold_a", "fold_b", "fold_c"]:
        raise ValueError("P2E fold set changed")
    if models.get("retraining_allowed") or models.get("tuning_allowed"):
        raise ValueError("P2E cannot retrain or tune models")
    motif = config["motif"]
    if motif.get("matrix_id") != "MA0035.5":
        raise ValueError("P2E motif matrix changed")
    if float(motif.get("primary_relative_score_threshold")) != 0.80:
        raise ValueError("P2E primary motif threshold changed")


def load_pwm(resource_path: Path, pseudocount: float, background: dict[str, float]):
    resource = json.loads(resource_path.read_text(encoding="utf-8"))
    counts = np.array([resource["pfm"][base] for base in DNA], dtype=float)
    probabilities = (counts + pseudocount) / (counts.sum(axis=0, keepdims=True) + 4 * pseudocount)
    bg = np.array([background[base] for base in DNA], dtype=float)[:, None]
    pwm = np.log2(probabilities / bg)
    minimum = float(pwm.min(axis=0).sum())
    maximum = float(pwm.max(axis=0).sum())
    return resource, pwm, minimum, maximum


def _score_kmer(kmer: str, pwm: np.ndarray) -> float:
    return float(sum(pwm[BASE_INDEX[base], index] for index, base in enumerate(kmer)))


def scan_pwm(sequence: str, pwm: np.ndarray, minimum: float, maximum: float) -> dict:
    sequence = sequence.upper()
    width = pwm.shape[1]
    best = None
    for start in range(len(sequence) - width + 1):
        kmer = sequence[start : start + width]
        candidates = [(_score_kmer(kmer, pwm), "+", kmer)]
        rc_kmer = reverse_complement(kmer)
        candidates.append((_score_kmer(rc_kmer, pwm), "-", rc_kmer))
        for raw_score, strand, oriented_kmer in candidates:
            relative = (raw_score - minimum) / (maximum - minimum)
            candidate = {
                "motif_start": start,
                "motif_end": start + width,
                "motif_strand": strand,
                "motif_kmer": oriented_kmer,
                "motif_raw_score": raw_score,
                "motif_relative_score": float(np.clip(relative, 0.0, 1.0)),
            }
            rank = (-candidate["motif_raw_score"], start, 0 if strand == "+" else 1)
            if best is None or rank < best[0]:
                best = (rank, candidate)
    if best is None:
        raise ValueError("Sequence is shorter than the motif")
    return best[1]


def motif_oriented_column(strand: str, width: int, original_offset: int) -> int:
    return original_offset if strand == "+" else width - 1 - original_offset


def motif_target_base(
    pwm: np.ndarray, strand: str, original_offset: int, original_base: str
) -> str:
    column = motif_oriented_column(strand, pwm.shape[1], original_offset)
    order = np.argsort(pwm[:, column], kind="stable")
    for index in order:
        oriented = DNA[int(index)]
        target = oriented if strand == "+" else oriented.translate(COMPLEMENT)
        if target != original_base:
            return target
    raise RuntimeError("No alternative motif-disruption base")


def disrupt_best_motif(sequence: str, hit: dict, pwm: np.ndarray) -> tuple[str, list[int]]:
    edited = list(sequence.upper())
    changed = []
    width = pwm.shape[1]
    for offset in range(width):
        position = int(hit["motif_start"]) + offset
        target = motif_target_base(
            pwm, str(hit["motif_strand"]), offset, edited[position]
        )
        if target != edited[position]:
            edited[position] = target
            changed.append(position)
    return "".join(edited), changed


def select_matched_flank_start(sequence_length: int, motif_start: int, width: int) -> int:
    motif_end = motif_start + width
    mirrored = sequence_length - motif_end
    candidates = list(range(sequence_length - width + 1))
    candidates.sort(key=lambda start: (abs(start - mirrored), start))
    for start in candidates:
        end = start + width
        if end <= motif_start or start >= motif_end:
            return start
    raise ValueError("No non-overlapping flank exists")


def edit_matched_flank(
    sequence: str, flank_start: int, motif_changed_positions: list[int], motif_start: int, width: int
) -> tuple[str, list[int]]:
    cycle = {"A": "C", "C": "G", "G": "T", "T": "A"}
    edited = list(sequence.upper())
    offsets = [position - motif_start for position in motif_changed_positions]
    changed = []
    for offset in offsets:
        position = flank_start + min(max(offset, 0), width - 1)
        edited[position] = cycle[edited[position]]
        changed.append(position)
    return "".join(edited), changed


def localization_metrics(values: np.ndarray, hit: dict, top_k: int, shifts: np.ndarray) -> dict:
    attribution = np.asarray(values, dtype=float)
    mass = np.abs(attribution)
    start, end = int(hit["motif_start"]), int(hit["motif_end"])
    total = float(mass.sum())
    observed = float(mass[start:end].sum() / total) if total > 0 else math.nan
    top = np.argsort(-mass, kind="stable")[:top_k]
    recall = float(np.isin(np.arange(start, end), top).mean())
    positions = np.arange(len(mass), dtype=float)
    center = (start + end - 1) / 2.0
    distance = float(np.sum(mass * np.abs(positions - center)) / total) if total > 0 else math.nan
    controls = []
    if total > 0:
        for shift in shifts:
            shifted = np.roll(mass, int(shift))
            controls.append(float(shifted[start:end].sum() / total))
    control_mean = float(np.mean(controls)) if controls else math.nan
    enrichment = observed / control_mean if control_mean > 0 else math.nan
    return {
        "motif_mass_fraction": observed,
        "top7_motif_recall": recall,
        "weighted_distance_to_motif": distance,
        "circular_control_mass_fraction": control_mean,
        "motif_mass_enrichment_vs_circular": enrichment,
    }


@torch.inference_mode()
def predict_logits(model, sequences: list[str], batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(sequences), batch_size):
        inputs = batch_one_hot_encode(sequences[start : start + batch_size]).to(DEVICE)
        outputs.append(model(inputs).detach().cpu().numpy())
    return np.concatenate(outputs) if outputs else np.array([], dtype=float)


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    value_column: str,
    replicates: int,
    seed: int,
) -> dict:
    required = {"fold_id", "seed", "sample_id", value_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing bootstrap columns: {sorted(required - set(frame.columns))}")
    data = frame[list(required)].dropna().copy()
    if data.empty:
        return {"estimate": math.nan, "ci95_low": math.nan, "ci95_high": math.nan, "p_two_sided": math.nan, "n": 0}
    estimate = float(data.groupby(["fold_id", "seed"], observed=True)[value_column].mean().mean())
    rng = np.random.default_rng(seed)
    folds = sorted(data.fold_id.unique())
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        sampled_fold_values = []
        for sampled_fold in rng.choice(folds, size=len(folds), replace=True):
            fold = data[data.fold_id.eq(sampled_fold)]
            seeds = sorted(fold.seed.unique())
            sampled_seed_values = []
            for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
                group = fold[fold.seed.eq(sampled_seed)]
                values = group[value_column].to_numpy(dtype=float)
                sampled_seed_values.append(float(rng.choice(values, size=len(values), replace=True).mean()))
            sampled_fold_values.append(float(np.mean(sampled_seed_values)))
        draws[index] = float(np.mean(sampled_fold_values))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    lower_tail = (float(np.sum(draws <= 0)) + 1.0) / (replicates + 1.0)
    upper_tail = (float(np.sum(draws >= 0)) + 1.0) / (replicates + 1.0)
    p = min(1.0, 2.0 * min(lower_tail, upper_tail))
    return {
        "estimate": estimate,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "p_two_sided": p,
        "n": int(len(data)),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return result.tolist()
    order = finite[np.argsort(values[finite], kind="stable")]
    ranked = values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result.tolist()


def verify_inputs(config: dict, root: Path) -> dict:
    checks = {
        "phase2d_resolved_config": (config["inputs"]["phase2d_dir"] + "/resolved_config.yaml", config["inputs"]["phase2d_resolved_config_sha256"]),
        "phase2d_gate": (config["inputs"]["phase2d_dir"] + "/phase2d_gate_assessment.json", config["inputs"]["phase2d_gate_sha256"]),
        "frozen_samples": (config["inputs"]["frozen_samples_path"], config["inputs"]["frozen_samples_sha256"]),
        "attribution_results": (config["inputs"]["attribution_results_path"], config["inputs"]["attribution_results_sha256"]),
        "conservative_dataset": (config["inputs"]["conservative_dataset_path"], config["inputs"]["conservative_dataset_sha256"]),
        "motif_resource": (config["motif"]["resource_path"], config["motif"]["resource_sha256"]),
    }
    rows = []
    for name, (relative, expected) in checks.items():
        path = root / relative
        observed = sha256_file(path)
        rows.append({"resource": name, "path": relative, "expected_sha256": expected, "observed_sha256": observed, "passed": observed.lower() == expected.lower()})
    phase2d_gate = json.loads((root / config["inputs"]["phase2d_dir"] / "phase2d_gate_assessment.json").read_text(encoding="utf-8"))
    if not phase2d_gate.get("all_scientific_interpretation_gates_passed") or not phase2d_gate.get("test_seal_intact"):
        raise ValueError("Phase 2D gates or test seal are not intact")
    expected_checkpoints = int(config["models"]["expected_checkpoints"])
    phase2d = root / config["inputs"]["phase2d_dir"]
    checkpoints = list((phase2d / "runs").glob("*/best_checkpoint.pt"))
    exact_files = list((phase2d / "runs").glob("*/exact_ism_attributions.npz"))
    audit = {
        "checksums": rows,
        "all_checksums_passed": bool(all(row["passed"] for row in rows)),
        "observed_checkpoints": len(checkpoints),
        "expected_checkpoints": expected_checkpoints,
        "observed_exact_ism_files": len(exact_files),
        "expected_exact_ism_files": int(config["models"]["expected_exact_ism_files"]),
        "phase2d_gates_passed": True,
        "phase2d_test_seal_intact": True,
    }
    if not audit["all_checksums_passed"] or len(checkpoints) != expected_checkpoints or len(exact_files) != audit["expected_exact_ism_files"]:
        raise ValueError("P2E input audit failed")
    return audit


def scan_frozen_samples(config: dict, root: Path, output_dir: Path, pwm, minimum, maximum) -> pd.DataFrame:
    samples = pd.read_csv(root / config["inputs"]["frozen_samples_path"])
    expected = {(fold, label): 40 for fold in config["models"]["folds"] for label in (0, 1)}
    if samples.groupby(["fold_id", "label"]).size().to_dict() != expected:
        raise ValueError("Frozen attribution cohort is incomplete")
    rows = []
    primary = float(config["motif"]["primary_relative_score_threshold"])
    sensitivity = [float(value) for value in config["motif"]["sensitivity_thresholds"]]
    for record in samples.to_dict("records"):
        hit = scan_pwm(record["sequence"], pwm, minimum, maximum)
        rows.append({
            "fold_id": record["fold_id"], "sample_id": record["sample_id"],
            "chromosome": record["chromosome"], "label": int(record["label"]),
            **hit,
            "strong_hit_080": hit["motif_relative_score"] >= primary,
            "strong_hit_085": hit["motif_relative_score"] >= sensitivity[0],
            "strong_hit_090": hit["motif_relative_score"] >= sensitivity[1],
        })
    frame = pd.DataFrame(rows).sort_values(["fold_id", "label", "sample_id"])
    frame.to_csv(output_dir / "motif_scan_samples.csv", index=False)
    return frame


def localization_stage(config: dict, root: Path, output_dir: Path, scans: pd.DataFrame) -> pd.DataFrame:
    phase2d = root / config["inputs"]["phase2d_dir"]
    sample_meta = pd.read_csv(root / config["inputs"]["frozen_samples_path"]).set_index("sample_id")
    scan_meta = scans.set_index("sample_id")
    rng = np.random.default_rng(int(config["localization"]["circular_shift_seed"]))
    shifts = np.sort(rng.choice(np.arange(1, 256), size=int(config["localization"]["circular_shift_count"]), replace=False))
    rows = []
    for fold_id in config["models"]["folds"]:
        for model_type in MODEL_ORDER:
            for seed in config["models"]["seeds"]:
                run_dir = phase2d / "runs" / run_dir_name(fold_id, model_type, int(seed))
                arrays = np.load(run_dir / "exact_ism_attributions.npz", allow_pickle=False)
                sample_ids = arrays["sample_ids"].astype(str)
                forward = arrays["forward_absolute"]
                if forward.shape != (80, 256) or len(sample_ids) != 80:
                    raise ValueError(f"Unexpected Exact ISM shape in {run_dir}")
                for sample_id, values in zip(sample_ids, forward):
                    meta = sample_meta.loc[sample_id]
                    hit = scan_meta.loc[sample_id]
                    if int(meta.label) != 1:
                        continue
                    metrics = localization_metrics(values, hit, int(config["localization"]["top_k"]), shifts)
                    rows.append({
                        "fold_id": fold_id, "model_type": model_type, "seed": int(seed),
                        "sample_id": sample_id, "chromosome": meta.chromosome,
                        "motif_start": int(hit.motif_start), "motif_end": int(hit.motif_end),
                        "motif_strand": hit.motif_strand,
                        "motif_relative_score": float(hit.motif_relative_score),
                        "strong_hit_080": bool(hit.strong_hit_080),
                        **metrics,
                    })
    frame = pd.DataFrame(rows).sort_values(["fold_id", "seed", "model_type", "sample_id"])
    frame.to_csv(output_dir / "motif_localization_results.csv", index=False)
    return frame


def disruption_stage(config: dict, root: Path, output_dir: Path, scans: pd.DataFrame, pwm) -> pd.DataFrame:
    phase2d = root / config["inputs"]["phase2d_dir"]
    samples = pd.read_csv(root / config["inputs"]["frozen_samples_path"])
    candidates = samples[samples.label.eq(1)].merge(scans, on=["fold_id", "sample_id", "chromosome", "label"])
    candidates = candidates[candidates.strong_hit_080].copy()
    prepared = {}
    width = pwm.shape[1]
    for row in candidates.to_dict("records"):
        hit = {key: row[key] for key in ("motif_start", "motif_end", "motif_strand")}
        motif_sequence, motif_changes = disrupt_best_motif(row["sequence"], hit, pwm)
        flank_start = select_matched_flank_start(len(row["sequence"]), int(row["motif_start"]), width)
        flank_sequence, flank_changes = edit_matched_flank(
            row["sequence"], flank_start, motif_changes, int(row["motif_start"]), width
        )
        if len(motif_changes) != len(flank_changes):
            raise ValueError("Motif and flank edit counts differ")
        prepared[row["sample_id"]] = {
            **row, "motif_disrupted_sequence": motif_sequence,
            "flank_disrupted_sequence": flank_sequence,
            "motif_mutation_count": len(motif_changes), "flank_start": flank_start,
        }
    rows = []
    batch_size = int(config["disruption"]["prediction_batch_size"])
    for fold_id in config["models"]["folds"]:
        fold_records = [value for value in prepared.values() if value["fold_id"] == fold_id]
        original = [row["sequence"] for row in fold_records]
        motif_edited = [row["motif_disrupted_sequence"] for row in fold_records]
        flank_edited = [row["flank_disrupted_sequence"] for row in fold_records]
        for model_type in MODEL_ORDER:
            for seed in config["models"]["seeds"]:
                run_dir = phase2d / "runs" / run_dir_name(fold_id, model_type, int(seed))
                model, _ = load_checkpoint(run_dir / "best_checkpoint.pt")
                original_logits = predict_logits(model, original, batch_size)
                motif_logits = predict_logits(model, motif_edited, batch_size)
                flank_logits = predict_logits(model, flank_edited, batch_size)
                original_prob = 1.0 / (1.0 + np.exp(-original_logits))
                motif_prob = 1.0 / (1.0 + np.exp(-motif_logits))
                flank_prob = 1.0 / (1.0 + np.exp(-flank_logits))
                for index, record in enumerate(fold_records):
                    motif_logit_drop = float(original_logits[index] - motif_logits[index])
                    flank_logit_drop = float(original_logits[index] - flank_logits[index])
                    rows.append({
                        "fold_id": fold_id, "model_type": model_type, "seed": int(seed),
                        "sample_id": record["sample_id"], "chromosome": record["chromosome"],
                        "motif_relative_score": float(record["motif_relative_score"]),
                        "motif_start": int(record["motif_start"]), "motif_strand": record["motif_strand"],
                        "flank_start": int(record["flank_start"]),
                        "mutation_count": int(record["motif_mutation_count"]),
                        "original_logit": float(original_logits[index]),
                        "motif_disrupted_logit": float(motif_logits[index]),
                        "flank_disrupted_logit": float(flank_logits[index]),
                        "motif_logit_drop": motif_logit_drop,
                        "flank_logit_drop": flank_logit_drop,
                        "motif_minus_flank_logit_drop": motif_logit_drop - flank_logit_drop,
                        "original_probability": float(original_prob[index]),
                        "motif_disrupted_probability": float(motif_prob[index]),
                        "flank_disrupted_probability": float(flank_prob[index]),
                        "motif_probability_drop": float(original_prob[index] - motif_prob[index]),
                        "flank_probability_drop": float(original_prob[index] - flank_prob[index]),
                    })
    frame = pd.DataFrame(rows).sort_values(["fold_id", "seed", "model_type", "sample_id"])
    frame.to_csv(output_dir / "motif_disruption_results.csv", index=False)
    return frame


def conservative_stage(config: dict, root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase2d = root / config["inputs"]["phase2d_dir"]
    data = pd.read_csv(root / config["inputs"]["conservative_dataset_path"])
    source = data[data.split.eq(config["inputs"]["allowed_source_split"])].copy()
    if len(source) != int(config["conservative_sensitivity"]["expected_source_rows"]):
        raise ValueError("Conservative sensitivity row count changed")
    p2d_config = yaml.safe_load((phase2d / "resolved_config.yaml").read_text(encoding="utf-8"))
    fold_chromosomes = {fold["fold_id"]: fold["holdout_chromosomes"] for fold in p2d_config["cross_validation"]["folds"]}
    rows = []
    summaries = []
    batch_size = int(config["conservative_sensitivity"]["prediction_batch_size"])
    threshold = float(config["conservative_sensitivity"]["threshold"])
    for fold_id in config["models"]["folds"]:
        fold = source[source.chromosome.isin(fold_chromosomes[fold_id])].copy()
        if set(fold.split.unique()) != {"train"}:
            raise ValueError("Forbidden conservative split reached a model")
        sequences = fold.sequence.astype(str).tolist()
        rc_sequences = [reverse_complement(sequence) for sequence in sequences]
        labels = fold.label.to_numpy(dtype=int)
        for model_type in MODEL_ORDER:
            for seed in config["models"]["seeds"]:
                run_dir = phase2d / "runs" / run_dir_name(fold_id, model_type, int(seed))
                model, _ = load_checkpoint(run_dir / "best_checkpoint.pt")
                forward = predict_sequences(model, sequences, batch_size)
                reverse = predict_sequences(model, rc_sequences, batch_size)
                consistency, per_sample = prediction_consistency_metrics(forward, reverse, threshold)
                predictions = (forward >= threshold).astype(int)
                summaries.append({
                    "fold_id": fold_id, "model_type": model_type, "seed": int(seed),
                    "n": int(len(fold)), "auroc": float(roc_auc_score(labels, forward)),
                    "auprc": float(average_precision_score(labels, forward)),
                    "accuracy": float(np.mean(predictions == labels)),
                    **consistency,
                })
                for index, record in enumerate(fold[["sample_id", "pair_id", "chromosome", "label"]].to_dict("records")):
                    rows.append({
                        "fold_id": fold_id, "model_type": model_type, "seed": int(seed), **record,
                        "probability_forward": float(forward[index]), "probability_rc": float(reverse[index]),
                        "prediction_difference": float(per_sample["prediction_difference"][index]),
                        "prediction_flip": int(per_sample["prediction_flip"][index]),
                    })
    detail = pd.DataFrame(rows).sort_values(["fold_id", "seed", "model_type", "sample_id"])
    summary = pd.DataFrame(summaries).sort_values(["fold_id", "seed", "model_type"])
    detail.to_csv(output_dir / "conservative_prediction_results.csv", index=False)
    summary.to_csv(output_dir / "conservative_run_summary.csv", index=False)
    return detail, summary


def paired_model_difference(frame: pd.DataFrame, endpoint: str, later: str, earlier: str) -> pd.DataFrame:
    keys = ["fold_id", "seed", "sample_id"]
    left = frame[frame.model_type.eq(later)][keys + [endpoint]].rename(columns={endpoint: "later"})
    right = frame[frame.model_type.eq(earlier)][keys + [endpoint]].rename(columns={endpoint: "earlier"})
    paired = left.merge(right, on=keys, validate="one_to_one")
    paired["difference"] = paired.later - paired.earlier
    return paired


def statistics_stage(config: dict, output_dir: Path, localization: pd.DataFrame, disruption: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    reps = int(config["statistics"]["hierarchical_bootstrap_replicates"])
    base_seed = int(config["statistics"]["hierarchical_bootstrap_seed"])
    strong = localization[localization.strong_hit_080].copy()
    tests = []
    for offset, (later, earlier) in enumerate(CONFIRMATORY_CONTRASTS):
        paired = paired_model_difference(strong, "motif_mass_fraction", later, earlier)
        result = hierarchical_bootstrap(paired, "difference", reps, base_seed + offset)
        tests.append({"family": "confirmatory", "endpoint": "motif_mass_fraction", "contrast": f"{later} minus {earlier}", **result})
    for offset, (later, earlier) in enumerate(CONFIRMATORY_CONTRASTS, start=10):
        paired = paired_model_difference(disruption, "motif_minus_flank_logit_drop", later, earlier)
        result = hierarchical_bootstrap(paired, "difference", reps, base_seed + offset)
        tests.append({"family": "confirmatory", "endpoint": "motif_minus_flank_logit_drop", "contrast": f"{later} minus {earlier}", **result})
    for offset, model_type in enumerate(MODEL_ORDER, start=20):
        subset = disruption[disruption.model_type.eq(model_type)].copy()
        result = hierarchical_bootstrap(subset, "motif_minus_flank_logit_drop", reps, base_seed + offset)
        tests.append({"family": "confirmatory", "endpoint": "motif_minus_flank_logit_drop_vs_zero", "contrast": model_type, **result})
    table = pd.DataFrame(tests)
    table["q_bh"] = benjamini_hochberg(table.p_two_sided.tolist())
    table.to_csv(output_dir / "confirmatory_bootstrap_results.csv", index=False)

    rcps_local = strong[strong.model_type.eq("CNN-RCPS")][["fold_id", "seed", "sample_id", "top7_motif_recall"]]
    rcps_disruption = disruption[disruption.model_type.eq("CNN-RCPS")][["fold_id", "seed", "sample_id", "motif_minus_flank_logit_drop"]]
    diagnostic = rcps_local.merge(rcps_disruption, on=["fold_id", "seed", "sample_id"], validate="one_to_one")
    diagnostic["biological_diagnostic_failure"] = (
        diagnostic.top7_motif_recall.lt(float(config["localization"]["low_recall_threshold"]))
        | diagnostic.motif_minus_flank_logit_drop.le(0)
    ).astype(float)
    h4_bootstrap = hierarchical_bootstrap(diagnostic, "biological_diagnostic_failure", reps, base_seed + 99)
    h4 = {
        "hypothesis": "H4 attribution symmetry is not equivalent to biological correctness",
        "rcps_phase2d_symmetry_gate_required": True,
        "diagnostic": "top7 motif recall < 0.5 OR motif-minus-flank logit specificity <= 0",
        **h4_bootstrap,
        "verdict": "supported_as_distinction" if h4_bootstrap["ci95_low"] > 0 else "inconclusive",
        "interpretation_limit": "The diagnostic does not label every case biologically wrong and does not prove a causal mechanism.",
    }
    write_json(output_dir / "h4_assessment.json", h4)
    return table, h4


def create_figures(scans, localization, disruption, conservative_summary, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for label, name, color in [(0, "Matched negative", "#6B7280"), (1, "GATA1 positive", "#2563EB")]:
        values = scans.loc[scans.label.eq(label), "motif_relative_score"]
        ax.hist(values, bins=np.linspace(0.45, 1.0, 24), alpha=0.55, density=True, label=name, color=color)
    ax.axvline(0.80, color="#B45309", linestyle="--", label="Frozen threshold 0.80")
    ax.set(xlabel="Best MA0035.5 relative score", ylabel="Density", title="Frozen GATA1 motif scores")
    ax.legend(frameon=True)
    fig.tight_layout(); fig.savefig(figures / "motif_score_distribution.png", dpi=180); fig.savefig(figures / "motif_score_distribution.pdf"); plt.close(fig)

    strong = localization[localization.strong_hit_080]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    data = [strong.loc[strong.model_type.eq(model), "motif_mass_fraction"] for model in MODEL_ORDER]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, model in zip(parts["bodies"], MODEL_ORDER): body.set_facecolor(MODEL_COLORS[model]); body.set_alpha(0.55)
    ax.set_xticks(range(1, 4), MODEL_ORDER); ax.set(ylabel="Exact ISM mass within best motif", title="Attribution localization on strong-hit positives")
    fig.tight_layout(); fig.savefig(figures / "motif_localization_by_model.png", dpi=180); fig.savefig(figures / "motif_localization_by_model.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    data = [disruption.loc[disruption.model_type.eq(model), "motif_minus_flank_logit_drop"] for model in MODEL_ORDER]
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, model in zip(parts["bodies"], MODEL_ORDER): body.set_facecolor(MODEL_COLORS[model]); body.set_alpha(0.55)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(range(1, 4), MODEL_ORDER); ax.set(ylabel="Motif drop minus flank drop (logit)", title="GATA1 motif-disruption specificity")
    fig.tight_layout(); fig.savefig(figures / "motif_disruption_by_model.png", dpi=180); fig.savefig(figures / "motif_disruption_by_model.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for index, model in enumerate(MODEL_ORDER):
        values = conservative_summary.loc[conservative_summary.model_type.eq(model), "auroc"]
        ax.scatter(np.full(len(values), index), values, color=MODEL_COLORS[model], s=34)
        ax.hlines(values.mean(), index - 0.24, index + 0.24, color="black", linewidth=2)
    ax.set_xticks(range(3), MODEL_ORDER); ax.set(ylabel="Chromosome-holdout AUROC", title="Conservative-IDR sensitivity performance")
    fig.tight_layout(); fig.savefig(figures / "conservative_auroc.png", dpi=180); fig.savefig(figures / "conservative_auroc.pdf"); plt.close(fig)


def summarize(config, scans, localization, disruption, conservative_detail, conservative_summary, tests, h4, audit, output_dir):
    strong_counts = scans.groupby("label").strong_hit_080.agg(["sum", "count"]).reset_index().to_dict("records")
    localization_summary = localization[localization.strong_hit_080].groupby("model_type", observed=True).agg(
        n=("sample_id", "size"), motif_mass_fraction_mean=("motif_mass_fraction", "mean"),
        top7_recall_mean=("top7_motif_recall", "mean"), circular_enrichment_mean=("motif_mass_enrichment_vs_circular", "mean")
    ).reset_index()
    disruption_summary = disruption.groupby("model_type", observed=True).agg(
        n=("sample_id", "size"), motif_logit_drop_mean=("motif_logit_drop", "mean"),
        flank_logit_drop_mean=("flank_logit_drop", "mean"), specificity_mean=("motif_minus_flank_logit_drop", "mean")
    ).reset_index()
    conservative = conservative_summary.groupby("model_type", observed=True).agg(
        runs=("seed", "size"), auroc_mean=("auroc", "mean"), auprc_mean=("auprc", "mean"),
        prediction_difference_mean=("prediction_mean_absolute_difference", "mean"), flip_rate_mean=("symmetry_flip_rate", "mean")
    ).reset_index()
    localization_summary.to_csv(output_dir / "motif_localization_summary.csv", index=False)
    disruption_summary.to_csv(output_dir / "motif_disruption_summary.csv", index=False)
    conservative.to_csv(output_dir / "conservative_model_summary.csv", index=False)
    scans.groupby("label", observed=True).agg(
        n=("sample_id", "size"), mean_score=("motif_relative_score", "mean"),
        hit_rate_080=("strong_hit_080", "mean"), hit_rate_085=("strong_hit_085", "mean"),
        hit_rate_090=("strong_hit_090", "mean")
    ).reset_index().to_csv(output_dir / "motif_scan_summary.csv", index=False)
    threshold_rows = []
    for threshold in (0.80, 0.85, 0.90):
        localized = localization[localization.motif_relative_score.ge(threshold)]
        disrupted = disruption[disruption.motif_relative_score.ge(threshold)]
        for model_type in MODEL_ORDER:
            loc = localized[localized.model_type.eq(model_type)]
            dis = disrupted[disrupted.model_type.eq(model_type)]
            threshold_rows.append({
                "threshold": threshold, "model_type": model_type, "n": int(len(loc)),
                "motif_mass_fraction_mean": float(loc.motif_mass_fraction.mean()),
                "top7_motif_recall_mean": float(loc.top7_motif_recall.mean()),
                "circular_enrichment_mean": float(loc.motif_mass_enrichment_vs_circular.mean()),
                "disruption_specificity_mean": float(dis.motif_minus_flank_logit_drop.mean()),
            })
    pd.DataFrame(threshold_rows).to_csv(output_dir / "threshold_sensitivity_summary.csv", index=False)
    gate = {
        "phase": "Phase 2E biological correctness",
        "input_audit_passed": bool(audit["all_checksums_passed"]),
        "expected_checkpoints": int(config["models"]["expected_checkpoints"]),
        "observed_checkpoints": int(audit["observed_checkpoints"]),
        "expected_exact_ism_files": int(config["models"]["expected_exact_ism_files"]),
        "observed_exact_ism_files": int(audit["observed_exact_ism_files"]),
        "frozen_samples": int(len(scans)),
        "positive_samples": int((scans.label == 1).sum()),
        "negative_samples": int((scans.label == 0).sum()),
        "localization_rows": int(len(localization)),
        "disruption_rows": int(len(disruption)),
        "conservative_prediction_rows": int(len(conservative_detail)),
        "expected_conservative_prediction_rows": int(config["conservative_sensitivity"]["expected_source_rows"]) * 9,
        "conservative_run_rows": int(len(conservative_summary)),
        "original_validation_model_rows": 0,
        "test_model_rows": 0,
        "test_seal_intact": True,
        "all_finite_primary_endpoints": bool(
            np.isfinite(localization.loc[localization.strong_hit_080, "motif_mass_fraction"]).all()
            and np.isfinite(disruption["motif_minus_flank_logit_drop"]).all()
        ),
    }
    gate["all_scientific_interpretation_gates_passed"] = bool(
        gate["input_audit_passed"]
        and gate["observed_checkpoints"] == gate["expected_checkpoints"]
        and gate["observed_exact_ism_files"] == gate["expected_exact_ism_files"]
        and gate["frozen_samples"] == 240
        and gate["localization_rows"] == 1080
        and gate["conservative_prediction_rows"] == gate["expected_conservative_prediction_rows"]
        and gate["conservative_run_rows"] == 27
        and gate["all_finite_primary_endpoints"]
        and gate["test_seal_intact"]
    )
    write_json(output_dir / "phase2e_gate_assessment.json", gate)
    completion = {
        "protocol_revision": config["protocol_revision"],
        "strong_hit_counts": strong_counts,
        "h4_verdict": h4["verdict"],
        "all_gates_passed": gate["all_scientific_interpretation_gates_passed"],
        "test_remained_sealed": True,
        "next_phase_authorized": False,
    }
    write_json(output_dir / "phase2e_completion.json", completion)
    return gate


def run(config_path: Path, output_dir: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_frozen_config(config)
    torch.set_num_threads(int(config.get("cpu_threads", 1)))
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "resolved_config.yaml")
    shutil.copy2(root / config["protocol_path"], output_dir / "frozen_protocol.md")
    audit = verify_inputs(config, root)
    write_json(output_dir / "input_audit.json", audit)
    resource_path = root / config["motif"]["resource_path"]
    resource, pwm, minimum, maximum = load_pwm(resource_path, float(config["motif"]["pseudocount"]), config["motif"]["background"])
    write_json(output_dir / "pwm_metadata.json", {
        "matrix_id": resource["matrix_id"], "width": int(pwm.shape[1]),
        "minimum_score": minimum, "maximum_score": maximum,
        "pwm_log2_odds": {base: pwm[index].tolist() for index, base in enumerate(DNA)},
    })
    scans = scan_frozen_samples(config, root, output_dir, pwm, minimum, maximum)
    localization = localization_stage(config, root, output_dir, scans)
    disruption = disruption_stage(config, root, output_dir, scans, pwm)
    conservative_detail, conservative_summary = conservative_stage(config, root, output_dir)
    tests, h4 = statistics_stage(config, output_dir, localization, disruption)
    create_figures(scans, localization, disruption, conservative_summary, output_dir)
    gate = summarize(config, scans, localization, disruption, conservative_detail, conservative_summary, tests, h4, audit, output_dir)
    if not gate["all_scientific_interpretation_gates_passed"]:
        raise RuntimeError("P2E completed but scientific interpretation gates failed")
    print(json.dumps({"output_dir": str(output_dir), "gates": gate, "h4": h4}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen Phase 2E biological-correctness analysis")
    parser.add_argument("--config", default="configs/phase2e_biological_correctness.yaml")
    parser.add_argument("--output-dir", default="results/phase2e_biological_correctness")
    args = parser.parse_args()
    run(Path(args.config), Path(args.output_dir))


if __name__ == "__main__":
    main()
