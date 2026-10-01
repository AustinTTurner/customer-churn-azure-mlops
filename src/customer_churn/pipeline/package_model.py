"""Package and verify a trained churn model artifact"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def calculate_sha256 (
    path: Path,
) -> str:
    """Calculate a SHA-256 hash for a file"""

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter (
            lambda: file_handle.read(8192),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    """Validate and package the trained model"""

    parser = argparse.ArgumentParser()

    parser.add_argument (
        "--model-input",
        required = True,
        help = "Directory containing training artifacts.",
    )

    parser.add_argument (
        "--package-output",
        required = True,
        help = "Directory for packaged model artifacts.",
    )

    args = parser.parse_args()

    model_input = Path (
        args.model_input
    )

    package_output = Path (
        args.package_output
    )

    source_model = (
        model_input / "model.joblib"
    )

    source_manifest = (
        model_input / "manifest.json"
    )

    if not source_model.is_file():
        raise FileNotFoundError (
            f"Model artifact not found: "
            f"{source_model}"
        )

    if not source_manifest.is_file():
        raise FileNotFoundError (
            f"Model manifest not found: "
            f"{source_manifest}"
        )

    package_output.mkdir (
        parents = True,
        exist_ok = True,
    )

    packaged_model = (
        package_output / "model.joblib"
    )

    packaged_manifest = (
        package_output / "manifest.json"
    )

    shutil.copy2 (
        source_model,
        packaged_model,
    )

    shutil.copy2 (
        source_manifest,
        packaged_manifest,
    )

    model_hash = calculate_sha256 (
        packaged_model
    )

    training_manifest = json.loads (
        packaged_manifest.read_text (
            encoding = "utf-8"
        )
    )

    package_manifest = {
        "model_name": training_manifest [
            "model_name"
        ],
        "model_version": training_manifest [
            "model_version"
        ],
        "decision_threshold": (
            training_manifest [
                "decision_threshold"
            ]
        ),
        "sha256": model_hash,
        "artifact": "model.joblib",
        "manifest": "manifest.json",
        "package_status": "verified",
    }

    package_manifest_path = (
        package_output
        / "package_manifest.json"
    )

    package_manifest_path.write_text (
        json.dumps (
            package_manifest,
            indent = 4,
        ),
        encoding = "utf-8",
    )

    print()
    print("Model packaging completed.")
    print(f"Model SHA-256: {model_hash}")
    print(
        f"Package output: "
        f"{package_output}"
    )


if __name__ == "__main__":
    main()