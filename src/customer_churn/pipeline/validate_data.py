"""Validate the cleaned churn dataset for pipeline training"""

import argparse
import json
from pathlib import Path

import pandas as pd

from customer_churn.features.preprocessing import MODEL_FEATURES

TARGET_COLUMN = "Churn"


def main() -> None:
    """Validate pipeline input data and write validated output"""

    parser = argparse.ArgumentParser()

    parser.add_argument (
        "--input-data",
        required = True,
        help = "Path to the cleaned churn CSV.",
    )

    parser.add_argument (
        "--output-dir",
        required = True,
        help = "Directory for validated pipeline data.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_data)
    output_dir = Path(args.output_dir)

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError (
            "Input dataset contains no records."
        )

    required_columns = set (
        MODEL_FEATURES + [TARGET_COLUMN]
    )

    missing_columns = sorted (
        required_columns - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError (
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError (
            "Target column contains missing values."
        )

    valid_targets = {"Yes", "No"}

    observed_targets = set (
        dataframe[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    unexpected_targets = (
        observed_targets - valid_targets
    )

    if unexpected_targets:
        raise ValueError (
            "Unexpected target values found: "
            f"{sorted(unexpected_targets)}"
        )

    output_dir.mkdir (
        parents = True,
        exist_ok = True,
    )

    validated_path = (
        output_dir / "validated_data.csv"
    )

    dataframe.to_csv (
        validated_path,
        index = False,
    )

    report = {
        "status": "passed",
        "rows": int(len(dataframe)),
        "columns": int(len(dataframe.columns)),
        "target_column": TARGET_COLUMN,
        "target_values": sorted (
            observed_targets
        ),
        "required_feature_count": len (
            MODEL_FEATURES
        ),
    }

    report_path = (
        output_dir / "validation_report.json"
    )

    report_path.write_text (
        json.dumps (
            report,
            indent = 4,
        ),
        encoding = "utf-8",
    )

    print()
    print("Data validation completed successfully.")
    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print(
        f"Validated data: {validated_path}"
    )
    print(
        f"Validation report: {report_path}"
    )


if __name__ == "__main__":
    main()