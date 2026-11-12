
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class WellLogVisualizer:
    """Visualize petrophysical logs and Vshale-derived lithology."""

    LITHOLOGY_COLORS = {
        "Clean Sand": "#FFD700",
        "Silty Sand": "#FFA500",
        "Sandy Shale": "#A0522D",
        "Shale": "#4B2F1A",
    }

    def __init__(self, dataframe):
        self.df = dataframe.copy()

        required = [
            "DEPTH", "GR", "IGR", "Vsh_Tertiary",
            "Vsh_PreTertiary", "RHOB", "DPOR",
            "CNPOR", "Lithology"
        ]

        missing = [col for col in required if col not in self.df.columns]

        if missing:
            raise ValueError(
                f"Missing columns required for visualization: {missing}"
            )

        self.df = self.df.sort_values("DEPTH").reset_index(drop=True)

    def plot_lithology_tracks(self, figsize=(18, 13)):
        fig, axes = plt.subplots(
            nrows=1,
            ncols=5,
            figsize=figsize,
            sharey=True,
            gridspec_kw={"width_ratios": [1, 1.2, 1, 1.2, 0.55]}
        )

        depth = self.df["DEPTH"]

        fig.suptitle(
            "Complete Petrophysical Log — WEEKS #1-10, Kansas\n"
            "With Vshale and Lithology Classification",
            fontsize=14,
            fontweight="bold",
            y=1.01
        )

        # Track 1: Gamma Ray
        ax = axes[0]

        ax.plot(
            self.df["GR"], depth,
            color="green", linewidth=0.8
        )

        ax.fill_betweenx(
            depth,
            self.df["GR"],
            75,
            where=self.df["GR"] < 75,
            color="yellow",
            alpha=0.5
        )

        ax.axvline(
            75, color="red",
            linestyle="--", linewidth=1
        )

        ax.set_xlabel("GR (GAPI)")
        ax.set_ylabel("Depth (ft)")
        ax.set_title("Gamma Ray")

        # Track 2: Vshale methods
        ax = axes[1]

        ax.plot(
            self.df["IGR"], depth,
            color="steelblue", linewidth=0.8,
            alpha=0.7, label="Linear IGR"
        )

        ax.plot(
            self.df["Vsh_Tertiary"], depth,
            color="darkorange", linewidth=0.8,
            alpha=0.7, label="Larionov Tertiary"
        )

        ax.plot(
            self.df["Vsh_PreTertiary"], depth,
            color="darkgreen", linewidth=1.2,
            label="Larionov Pre-Tertiary"
        )

        ax.axvline(
            0.5, color="red",
            linestyle="--", linewidth=1
        )

        ax.set_xlim(0, 1)
        ax.set_xlabel("Vshale")
        ax.set_title("Vshale\nThree Methods")
        ax.legend(fontsize=7, loc="lower right")

        # Track 3: Bulk density
        ax = axes[2]

        ax.plot(
            self.df["RHOB"], depth,
            color="red", linewidth=0.8
        )

        ax.set_xlabel("RHOB (g/cc)")
        ax.set_title("Bulk Density")

        # Track 4: Density and neutron porosity
        ax = axes[3]

        ax.plot(
            self.df["DPOR"], depth,
            color="blue", linewidth=0.8,
            label="DPOR"
        )

        ax.plot(
            self.df["CNPOR"], depth,
            color="purple", linewidth=0.8,
            linestyle="--", label="CNPOR"
        )

        ax.set_xlabel("Porosity (PU)")
        ax.set_title("Porosity\nDPOR vs CNPOR")
        ax.legend(fontsize=7)

        # Track 5: Lithology classification
        ax = axes[4]

        for i in range(len(self.df) - 1):
            lithology = self.df["Lithology"].iloc[i]

            if lithology not in self.LITHOLOGY_COLORS:
                continue

            ax.fill_betweenx(
                [depth.iloc[i], depth.iloc[i + 1]],
                0, 1,
                color=self.LITHOLOGY_COLORS[lithology],
                alpha=0.9
            )

        patches = [
            mpatches.Patch(color=color, label=name)
            for name, color in self.LITHOLOGY_COLORS.items()
        ]

        ax.legend(
            handles=patches,
            fontsize=7,
            loc="lower right"
        )

        ax.set_xlabel("Lithology")
        ax.set_title("Lithology\nClassification")
        ax.set_xticks([])

        # Shared depth formatting
        for ax in axes:
            ax.grid(True, alpha=0.2)
            ax.invert_yaxis()

        plt.tight_layout()

        return fig, axes
