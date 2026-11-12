
from pathlib import Path
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


FEATURES = ["GR", "RHOB", "DPOR", "RILD", "SP", "PE", "Vsh_PreTertiary"]
CLASSES = ["Clean Sand", "Silty Sand", "Sandy Shale", "Shale"]
COLORS = ["#FFD700", "#FFA500", "#A0522D", "#4B2F1A"]


class LithologyTrainer:
    def __init__(self, df, output_dir):
        self.df = df.sort_values("DEPTH").reset_index(drop=True).copy()
        self.output = Path(output_dir)
        self.figures = self.output / "figures"
        self.models_dir = self.output / "models"
        self.reports = self.output / "reports"

        for folder in [self.figures, self.models_dir, self.reports]:
            folder.mkdir(parents=True, exist_ok=True)

        required = FEATURES + ["Lith_Label", "Lithology", "DEPTH"]
        missing = set(required) - set(self.df.columns)

        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        if self.df[required].isna().any().any():
            raise ValueError("Model dataset contains missing values.")

    def explore(self):
        counts = self.df["Lithology"].value_counts().reindex(CLASSES, fill_value=0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(CLASSES, counts, color=COLORS)
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].set_title("Lithology Distribution")

        axes[1].pie(counts, labels=CLASSES, colors=COLORS, autopct="%1.1f%%")
        axes[1].set_title("Lithology Proportions")

        plt.tight_layout()
        fig.savefig(self.figures / "class_distribution.png", dpi=300)
        plt.show()

        fig, axes = plt.subplots(2, 4, figsize=(15, 9))
        for ax, feature in zip(axes.flat, FEATURES):
            values = [
                self.df.loc[self.df["Lithology"] == name, feature]
                for name in CLASSES
            ]
            box = ax.boxplot(values, patch_artist=True)
            for patch, color in zip(box["boxes"], COLORS):
                patch.set_facecolor(color)
            ax.set_title(feature)
            ax.set_xticks(range(1, 5), ["CS", "SS", "SSh", "Sh"])
        axes.flat[-1].set_visible(False)

        plt.tight_layout()
        fig.savefig(self.figures / "feature_distributions.png", dpi=300)
        plt.show()

    def prepare_data(self):
        X = self.df[FEATURES]
        y = self.df["Lith_Label"]

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        )

        self.X_bal, self.y_bal = SMOTE(
            random_state=42, k_neighbors=5
        ).fit_resample(self.X_train, self.y_train)

        self.scaler = StandardScaler()
        self.X_bal_scaled = self.scaler.fit_transform(self.X_bal)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train(self):
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, solver="lbfgs"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=10,
                min_samples_split=5, min_samples_leaf=2,
                random_state=42, n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200, max_depth=6,
                learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, random_state=42,
                eval_metric="mlogloss", verbosity=0
            )
        }

        self.results = {}

        for name, model in models.items():
            X_train = (
                self.X_bal_scaled
                if name == "Logistic Regression"
                else self.X_bal
            )
            X_test = (
                self.X_test_scaled
                if name == "Logistic Regression"
                else self.X_test
            )

            model.fit(X_train, self.y_bal)
            predictions = model.predict(X_test)

            self.results[name] = {
                "model": model,
                "predictions": predictions,
                "accuracy": accuracy_score(self.y_test, predictions)
            }

        self.best_name = max(
            self.results,
            key=lambda name: self.results[name]["accuracy"]
        )
        self.best_model = self.results[self.best_name]["model"]

    def evaluate(self):
        reports = {}

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, (name, result) in zip(axes, self.results.items()):
            predictions = result["predictions"]

            reports[name] = classification_report(
                self.y_test, predictions,
                labels=range(4), target_names=CLASSES,
                output_dict=True, zero_division=0
            )

            cm = confusion_matrix(
                self.y_test, predictions, labels=range(4)
            )
            ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{name}\n{result['accuracy']:.2%}")
            ax.set_xticks(range(4), ["CS", "SS", "SSh", "Sh"])
            ax.set_yticks(range(4), ["CS", "SS", "SSh", "Sh"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            for i in range(4):
                for j in range(4):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

        plt.tight_layout()
        fig.savefig(self.figures / "confusion_matrices.png", dpi=300)
        plt.show()

        pd.DataFrame(reports).to_json(
            self.reports / "classification_reports.json", indent=4
        )

        for name, report in reports.items():
            print(f"\n{name}\n")
            print(pd.DataFrame(report).round(3).to_string())

    def cross_validate(self):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.cv_results = {}

        for name, result in self.results.items():
            X = (
                self.X_bal_scaled
                if name == "Logistic Regression"
                else self.X_bal
            )
            scores = cross_val_score(
                result["model"], X, self.y_bal,
                cv=cv, scoring="accuracy"
            )
            self.cv_results[name] = scores

        names = list(self.results)
        test_scores = [self.results[n]["accuracy"] * 100 for n in names]
        cv_means = [self.cv_results[n].mean() * 100 for n in names]
        cv_stds = [self.cv_results[n].std() * 100 for n in names]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(names, test_scores)
        axes[0].set_title("Test Accuracy")
        axes[1].bar(names, cv_means, yerr=cv_stds, capsize=5)
        axes[1].set_title("5-Fold Cross-Validation")
        for ax in axes:
            ax.tick_params(axis="x", rotation=15)
            ax.set_ylabel("Accuracy (%)")

        plt.tight_layout()
        fig.savefig(self.figures / "model_comparison.png", dpi=300)
        plt.show()

        print({
            name: {
                "test_accuracy": round(self.results[name]["accuracy"], 4),
                "cv_mean": round(scores.mean(), 4),
                "cv_std": round(scores.std(), 4)
            }
            for name, scores in self.cv_results.items()
        })

    def feature_importance(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
            importance = pd.Series(
                self.results[name]["model"].feature_importances_,
                index=FEATURES
            ).sort_values()
            importance.plot.barh(ax=ax)
            ax.set_title(name)

        plt.tight_layout()
        fig.savefig(self.figures / "feature_importance.png", dpi=300)
        plt.show()

    def depth_tracks(self):
        X = self.df[FEATURES]
        if self.best_name == "Logistic Regression":
            X = self.scaler.transform(X)

        predictions = self.best_model.predict(X)
        mapping = dict(enumerate(CLASSES))
        self.df["Lith_Predicted"] = [mapping[int(x)] for x in predictions]

        fig, axes = plt.subplots(1, 4, figsize=(12, 13), sharey=True)
        depth = self.df["DEPTH"].to_numpy()

        axes[0].plot(self.df["GR"], depth, color="green")
        axes[0].axvline(75, color="red", linestyle="--")
        axes[0].set_title("Gamma Ray")
        axes[0].set_xlabel("GR (API)")

        axes[1].plot(self.df["RHOB"], depth, color="red")
        axes[1].set_title("Bulk Density")
        axes[1].set_xlabel("RHOB (g/cc)")

        for ax, column, title in [
            (axes[2], "Lithology", "Actual Lithology"),
            (axes[3], "Lith_Predicted", f"Predicted: {self.best_name}")
        ]:
            for i in range(len(self.df) - 1):
                color = COLORS[CLASSES.index(self.df[column].iloc[i])]
                ax.fill_betweenx(
                    depth[i:i+2], 0, 1, color=color
                )
            ax.set_title(title)
            ax.set_xlim(0, 1)
            ax.set_xticks([])

        for ax in axes:
            ax.invert_yaxis()
            ax.grid(alpha=0.2)

        plt.tight_layout()
        fig.savefig(self.figures / "lithology_depth_tracks.png", dpi=300)
        plt.show()

    def save(self):
        filename = f"classifier_{self.best_name.replace(' ', '_')}_week3.pkl"

        joblib.dump(self.best_model, self.models_dir / filename)
        joblib.dump(self.scaler, self.models_dir / "scaler_week3.pkl")

        (self.models_dir / "classifier_features_week3.json").write_text(
            json.dumps(FEATURES, indent=2)
        )
        (self.models_dir / "classifier_classes_week3.json").write_text(
            json.dumps(dict(enumerate(CLASSES)), indent=2)
        )

        self.df.to_csv(
            self.output / "weeks1_10_with_predictions.csv", index=False
        )
