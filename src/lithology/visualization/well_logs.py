import matplotlib.pyplot as plt
import numpy as np


class WellLogVisualizer:

    def __init__(self, dataframe):
        self.df = dataframe.copy()

    def plot_log_tracks(self, depth_column="DEPTH", figsize=(16, 12)):
        tracks = [
            {"curve": "GR", "title": "Gamma Ray", "unit": "API", "color": "green"},
            {"curve": "SP", "title": "Spontaneous Potential", "unit": "mV", "color": "black"},
            {"curve": "RILD", "title": "Deep Resistivity", "unit": "ohm.m", "color": "red"},
            {"curve": "RILM", "title": "Medium Resistivity", "unit": "ohm.m", "color": "blue"},
            {"curve": "RHOB", "title": "Bulk Density", "unit": "g/cc", "color": "brown"},
            {"curve": "DPOR", "title": "Density Porosity", "unit": "PU", "color": "red"},
            {"curve": "CNPOR", "title": "Neutron Porosity", "unit": "PU", "color": "blue"},
            {"curve": "PE", "title": "Photoelectric Factor", "unit": "barns/electron", "color": "purple"},
        ]

        fig, axes = plt.subplots(
            ncols=len(tracks),
            nrows=1,
            figsize=figsize,
            sharey=True,
        )

        depth = self.df[depth_column]

        for ax, track in zip(axes, tracks):
            curve = track["curve"]

            ax.plot(
                self.df[curve],
                depth,
                color=track["color"],
                linewidth=0.7,
            )

            ax.set_title(track["title"], fontsize=9, fontweight="bold")
            ax.set_xlabel(track["unit"], fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.tick_params(axis="both", labelsize=8)
            ax.set_ylim(depth.max(), depth.min())

        axes[0].set_ylabel("Measured Depth (ft)")
        fig.suptitle(
            "SUPPES #1-14 — Petrophysical Well Log Panel",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout()

        return fig, axes