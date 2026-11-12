
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier


FEATURES = [
    "GR", "RHOB", "DPOR", "RILD", "SP", "PE",
    "Vsh_PreTertiary"
]

CLASS_NAMES = [
    "Clean Sand", "Silty Sand", "Sandy Shale", "Shale"
]


def train_models(X_train, X_test, y_train, y_test, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        joblib.dump(
            model,
            output / f"{name.lower().replace(' ', '_')}.pkl"
        )

        results[name] = {
            "model": model,
            "predictions": predictions,
            "accuracy": accuracy_score(y_test, predictions)
        }

    scores = pd.DataFrame({
        name: {"Accuracy": result["accuracy"]}
        for name, result in results.items()
    }).T.sort_values("Accuracy", ascending=False)

    best_name = scores.index[0]
    best_model = results[best_name]["model"]

    joblib.dump(best_model, output / "best_model.pkl")
    scores.to_csv(output / "model_comparison.csv")

    return results, scores, best_name


def evaluate_model(y_test, predictions, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2, 3]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output / "confusion_matrix.png", dpi=300)
    plt.show()
    plt.close()
