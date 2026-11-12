import numpy as np
import pandas as pd


class DataCleaner:
    """Clean and prepare petrophysical well-log data."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        depth_column: str = "DEPTH",
        null_values: list | None = None,
    ):
        self.df = dataframe.copy()
        self.depth_column = depth_column
        self.null_values = null_values or [-999.25, -999.0, -9999.0]

        if self.depth_column not in self.df.columns:
            raise ValueError(f"{self.depth_column} column not found.")

    def convert_numeric(self):
        """Convert measurement columns to numeric."""

        for column in self.df.columns:
            if column != self.depth_column:
                self.df[column] = pd.to_numeric(
                    self.df[column], errors="coerce"
                )

        self.df[self.depth_column] = pd.to_numeric(
            self.df[self.depth_column], errors="coerce"
        )

        return self

    def replace_invalid_values(self):
        """Replace null sentinels and infinite values with NaN."""

        self.df.replace(self.null_values, np.nan, inplace=True)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)

        return self

    def remove_invalid_depths(self):
        """Remove rows without valid depth measurements."""

        self.df.dropna(subset=[self.depth_column], inplace=True)
        self.df.sort_values(self.depth_column, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self

    def handle_duplicate_depths(self, strategy="mean"):
        """Resolve repeated depth measurements."""

        if strategy not in {"mean", "first"}:
            raise ValueError("Strategy must be 'mean' or 'first'.")

        if strategy == "mean":
            numeric_columns = self.df.select_dtypes(
                include=np.number
            ).columns.tolist()

            self.df = (
                self.df.groupby(self.depth_column, as_index=False)[
                    numeric_columns
                ].mean()
            )
        else:
            self.df.drop_duplicates(
                subset=[self.depth_column],
                keep="first",
                inplace=True,
            )

        return self

    def interpolate_missing(self, columns=None):
        """
        Interpolate missing log measurements along depth.

        Only fills gaps between existing measurements.
        Does not extrapolate at the beginning or end of a log.
        """

        if columns is None:
            columns = [
                column for column in self.df.select_dtypes(
                    include=np.number
                ).columns
                if column != self.depth_column
            ]

        self.df[columns] = self.df[columns].interpolate(
            method="linear",
            limit_area="inside",
        )

        return self

    def get_cleaned_data(self):
        """Return cleaned DataFrame."""

        return self.df.copy()

    def missing_value_report(self):
        """Report remaining missing values."""

        return pd.DataFrame({
            "missing_count": self.df.isna().sum(),
            "missing_percentage": (
                self.df.isna().mean() * 100
            ),
        }).sort_values("missing_percentage", ascending=False)

    def run(self, duplicate_strategy="mean"):
        """Execute the preprocessing workflow."""

        return (
            self.convert_numeric()
            .replace_invalid_values()
            .remove_invalid_depths()
            .handle_duplicate_depths(duplicate_strategy)
            .interpolate_missing()
            .get_cleaned_data()
        )