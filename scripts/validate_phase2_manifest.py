from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


MD5_RE = re.compile(r"^[0-9a-f]{32}$")


def validate_manifest(path: Path) -> list[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if data.get("task", {}).get("assembly") != "GRCh38":
        errors.append("task assembly must be GRCh38")
    if not isinstance(data.get("formal_training_allowed"), bool):
        errors.append("formal_training_allowed must be boolean")

    files = data.get("files", {})
    required = {
        "positive_peaks",
        "accessible_negative_pool",
        "reference_fasta",
        "blacklist",
        "ctcf_positive_control",
    }
    missing = sorted(required.difference(files))
    if missing:
        errors.append(f"missing file records: {', '.join(missing)}")

    for role, record in files.items():
        if record.get("status") != "released":
            errors.append(f"{role}: status is not released")
        if record.get("assembly") != "GRCh38":
            errors.append(f"{role}: assembly is not GRCh38")
        md5 = str(record.get("md5sum", ""))
        if not MD5_RE.fullmatch(md5):
            errors.append(f"{role}: invalid md5sum")
        url = str(record.get("url", ""))
        if not url.startswith("https://www.encodeproject.org/"):
            errors.append(f"{role}: URL must be an official HTTPS ENCODE URL")

    if data.get("formal_training_allowed") is True:
        if data.get("freeze_status") != "frozen_for_phase2_pilot":
            errors.append("training may be allowed only for a frozen_for_phase2_pilot manifest")
        project_root = path.resolve().parents[1]
        final_audit_path = project_root / "metadata" / "phase2" / "final_dataset_audit.json"
        reference_audit_path = project_root / "metadata" / "phase2" / "reference_equivalence_audit.json"
        dataset_path = project_root / "data" / "phase2" / "processed" / "phase2_matched_dataset.csv"
        try:
            final_audit = json.loads(final_audit_path.read_text(encoding="utf-8"))
            reference_audit = json.loads(reference_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot read final training-gate audits: {error}")
        else:
            if final_audit.get("formal_training_allowed") is not True:
                errors.append("final dataset audit does not allow formal training")
            if not final_audit.get("matching_balance", {}).get("passed"):
                errors.append("matching balance gate failed")
            if not final_audit.get("leakage_audit", {}).get("passed"):
                errors.append("primary leakage gate failed")
            if not final_audit.get("conservative_peak_sensitivity", {}).get("leakage_audit", {}).get("passed"):
                errors.append("conservative sensitivity leakage gate failed")
            if reference_audit.get("passed") is not True:
                errors.append("reference equivalence gate failed")
            if dataset_path.exists():
                observed_dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
                if observed_dataset_sha256 != final_audit.get("dataset_sha256"):
                    errors.append("final dataset SHA-256 does not match the audit")
                if observed_dataset_sha256 != data.get("current_qc_state", {}).get("final_dataset_sha256"):
                    errors.append("final dataset SHA-256 does not match the manifest")
            else:
                errors.append("final matched dataset is missing")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Phase 2 ENCODE manifest")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "phase2_dataset_manifest.yaml",
    )
    args = parser.parse_args()

    errors = validate_manifest(args.manifest)
    digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    print(f"manifest={args.manifest}")
    print(f"sha256={digest}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    freeze_status = yaml.safe_load(args.manifest.read_text(encoding="utf-8")).get("freeze_status")
    if freeze_status == "frozen_for_phase2_pilot":
        print("status=valid_frozen_manifest")
    else:
        print("status=valid_provisional_manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
