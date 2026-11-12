
from pathlib import Path

import numpy as np
import pandas as pd


class DataQualityAnalyzer:
    """
    Performs data quality assessment on petrophysical well-log data.

    Checks:
    - Missing values
    - Duplicate depths
    - Duplicate records
    - Invalid numerical values
    - Potential outliers using IQR
    - Constant and near-constant curves
    - Depth continuity
    - Statistical summaries
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        depth_column: str = "DEPTH",
        null_values: list | None = None,
    ):
        self.df = dataframe.copy()
        self.depth_column = depth_column
        self.null_values = null_values or [-999.25, -999.0, -9999.0]

        if self.df.empty:
            raise ValueError("Input DataFrame is empty.")

        if depth_column not in self.df.columns:
            raise ValueError(
                f"Depth column '{depth_column}' not found."
            )

    def _numeric_data(self):
        """Return numeric columns excluding depth."""

        return self.df.select_dtypes(
            include=np.number
        ).drop(
            columns=[self.depth_column],
            errors="ignore",
        )

    def missing_values(self):
        """Calculate missing-value counts and percentages."""

        missing = self.df.isna().sum()

        report = pd.DataFrame({
            "missing_count": missing,
            "missing_percentage": (
                missing / len(self.df)
            ) * 100,
        })

        return report.sort_values(
            "missing_percentage",
            ascending=False,
        )

    def duplicate_records(self):
        """Count duplicate rows."""

        return int(self.df.duplicated().sum())

    def duplicate_depths(self):
        """Identify repeated depth measurements."""

        return self.df[
            self.df.duplicated(
                subset=[self.depth_column],
                keep=False,
            )
        ].sort_values(self.depth_column)

    def invalid_values(self):
        """
        Identify non-finite values and declared LAS null sentinels.

        Sentinel values are reported, not automatically replaced.
        """

        numeric = self._numeric_data()

        report = {}

        for column in numeric.columns:
            values = numeric[column]

            report[column] = {
                "non_finite": int(
                    (~np.isfinite(values)).sum()
                ),
                "null_sentinels": int(
                    values.isin(self.null_values).sum()
                ),
            }

        return pd.DataFrame(report).T

    def outlier_report(self):
        """
        Identify potential outliers using the IQR rule.

        Outliers are flagged for investigation, not removed.
        """

        numeric = self._numeric_data()
        report = []

        for column in numeric.columns:
            values = numeric[column].dropna()

            if values.empty:
                continue

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = (
                (values < lower) |
                (values > upper)
            )

            report.append({
                "curve": column,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(outliers.sum()),
                "outlier_percentage": (
                    outliers.mean() * 100
                ),
            })

        return pd.DataFrame(report)

    def constant_curves(self, tolerance: float = 1e-8):
        """Identify constant or near-constant numerical curves."""

        numeric = self._numeric_data()
        report = []

        for column in numeric.columns:
            values = numeric[column].dropna()

            if values.empty:
                continue

            unique_count = values.nunique()
            std = values.std()

            report.append({
                "curve": column,
                "unique_values": unique_count,
                "standard_deviation": std,
                "is_constant": unique_count <= 1,
                "is_near_constant": (
                    unique_count > 1 and std <= tolerance
                ),
            })

        return pd.DataFrame(report)

    def depth_continuity(self):
        """Assess depth sampling intervals."""

        depth = (
            pd.to_numeric(
                self.df[self.depth_column],
                errors="coerce",
            )
            .dropna()
            .sort_values()
        )

        differences = depth.diff().dropna()

        if differences.empty:
            return {
                "minimum_depth": depth.min(),
                "maximum_depth": depth.max(),
                "median_step": np.nan,
                "mean_step": np.nan,
                "minimum_step": np.nan,
                "maximum_step": np.nan,
                "non_increasing_steps": 0,
            }

        return {
            "minimum_depth": depth.min(),
            "maximum_depth": depth.max(),
            "median_step": differences.median(),
            "mean_step": differences.mean(),
            "minimum_step": differences.min(),
            "maximum_step": differences.max(),
            "non_increasing_steps": int(
                (differences <= 0).sum()
            ),
        }

    def statistical_summary(self):
        """Generate descriptive statistics for numerical curves."""

        return self.df.describe().T

    def generate_report(self):
        """Generate a complete data quality report."""

        return {
            "dataset_shape": self.df.shape,
            "missing_values": self.missing_values(),
            "duplicate_records": self.duplicate_records(),
            "duplicate_depths": self.duplicate_depths(),
            "invalid_values": self.invalid_values(),
            "outliers": self.outlier_report(),
            "constant_curves": self.constant_curves(),
            "depth_continuity": self.depth_continuity(),
            "statistics": self.statistical_summary(),
        }

    def export_report(self, output_directory: str | Path):
        """Export individual quality reports as CSV files."""

        output_directory = Path(output_directory)
        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        report = self.generate_report()

        for name, result in report.items():
            if isinstance(result, pd.DataFrame):
                result.to_csv(
                    output_directory / f"{name}.csv"
                )

        summary = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "duplicate_records": report["duplicate_records"],
            **report["depth_continuity"],
        }

        pd.Series(summary).to_csv(
            output_directory / "dataset_summary.csv"
        )

        return output_directory
