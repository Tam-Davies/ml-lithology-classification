
from pathlib import Path

import lasio
import pandas as pd


class LASLoader:
    """
    Load and inspect LAS well-log files.

    Responsibilities:
    - Validate the LAS file path.
    - Read LAS data.
    - Retrieve well metadata.
    - Identify available log curves.
    - Convert log measurements into a DataFrame.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.las = None

    def load(self):
        """Read the LAS file and return the LAS object."""

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"LAS file not found: {self.file_path}"
            )

        if self.file_path.suffix.lower() != ".las":
            raise ValueError("Expected a .las file.")

        self.las = lasio.read(self.file_path)

        return self.las

    def _check_loaded(self):
        """Ensure the LAS file has been loaded."""

        if self.las is None:
            raise RuntimeError(
                "LAS file has not been loaded. Call load() first."
            )

    def get_curves(self) -> list[dict]:
        """Return available curve mnemonics and descriptions."""

        self._check_loaded()

        return [
            {
                "mnemonic": curve.mnemonic,
                "unit": curve.unit,
                "description": curve.descr,
            }
            for curve in self.las.curves
        ]

    def get_metadata(self) -> dict:
        """Return basic well-header information."""

        self._check_loaded()

        well_info = {}

        for item in self.las.well:
            well_info[item.mnemonic] = item.value

        return well_info

    def to_dataframe(self) -> pd.DataFrame:
        """Convert LAS measurements into a DataFrame."""

        self._check_loaded()

        df = self.las.df().reset_index()

        # Standardize the depth column name.
        df.rename(
            columns={df.columns[0]: "DEPTH"},
            inplace=True,
        )

        return df
