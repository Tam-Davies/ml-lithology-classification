import numpy as np
import pandas as pd


class PetrophysicalFeatureEngineer:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def calculate_igr(self, gr_column="GR"):
        """Calculate Index Gamma Ray."""

        gr = self.df[gr_column]
        gr_min = gr.quantile(0.05)
        gr_max = gr.quantile(0.95)

        self.df["IGR"] = (
            (gr - gr_min) / (gr_max - gr_min)
        ).clip(0, 1)

        return self

    def calculate_vshale(self, method="tertiary"):
        """Calculate Larionov shale volume."""

        igr = self.df["IGR"].clip(0, 1)

        if method == "tertiary":
            self.df["VSH"] = (
                0.083 * (2 ** (3.7 * igr) - 1)
            )

        elif method == "pre_tertiary":
            self.df["VSH"] = (
                0.33 * (2 ** (2 * igr) - 1)
            )

        else:
            raise ValueError("Unknown Larionov method.")

        self.df["VSH"] = self.df["VSH"].clip(0, 1)

        return self

    def calculate_porosity_difference(self):
        """Calculate density-neutron porosity separation."""

        if {"DPOR", "CNPOR"}.issubset(self.df.columns):
            self.df["POROSITY_SEPARATION"] = (
                self.df["CNPOR"] - self.df["DPOR"]
            )

        return self

    def calculate_resistivity_ratio(self):
        """Calculate deep-to-medium resistivity ratio."""

        if {"RILD", "RILM"}.issubset(self.df.columns):
            denominator = self.df["RILM"].replace(0, np.nan)

            self.df["RESISTIVITY_RATIO"] = (
                self.df["RILD"] / denominator
            )

        return self

    def transform(self, vshale_method="tertiary"):
        return (
            self.calculate_igr()
            .calculate_vshale(vshale_method)
            .calculate_porosity_difference()
            .calculate_resistivity_ratio()
            .get_features()
        )

    def get_features(self):
        return self.df.copy()