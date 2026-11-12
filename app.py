
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "weeks1_10_processed.csv"
MODEL_DIR = ROOT / "artifacts" / "models"

FEATURES = ["GR", "RHOB", "DPOR", "RILD", "SP", "PE", "Vsh_PreTertiary"]
VSH_FEATURES = ["GR", "RHOB", "DPOR", "RILD", "SP", "PE"]

CLASS_NAMES = {
    0: "Clean Sand",
    1: "Silty Sand",
    2: "Sandy Shale",
    3: "Shale"
}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

st.set_page_config(
    page_title="Lithology Intelligence Dashboard",
    page_icon="🪨",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


df = load_data()

st.title("Lithology Intelligence Dashboard")
st.caption("Petrophysical well-log analysis | Lithology classification | Vsh regression")

with st.sidebar:
    st.header("Model Configuration")
    model_name = st.selectbox("Classification Model", list(MODEL_FILES))
    st.divider()
    st.caption("Vsh prediction uses a separate Random Forest regression model.")

model_path = MODEL_DIR / MODEL_FILES[model_name]
vsh_path = MODEL_DIR / "vsh_regressor.pkl"

if not model_path.exists() or not vsh_path.exists():
    st.error("Required model files are missing. Rerun model training in the notebook.")
    st.stop()

if not all(col in df.columns for col in FEATURES + ["DEPTH", "Lith_Label"]):
    st.error("The processed dataset is missing required columns.")
    st.stop()

classifier = load_model(str(model_path))
vsh_regressor = load_model(str(vsh_path))

df = df.sort_values("DEPTH").copy()

df["Predicted_Label"] = classifier.predict(df[FEATURES]).astype(int)
df["Predicted_Vsh"] = vsh_regressor.predict(df[VSH_FEATURES])
df["Original_Lithology"] = df["Lith_Label"].map(CLASS_NAMES)
df["Predicted_Lithology"] = df["Predicted_Label"].map(CLASS_NAMES)

agreement = (df["Lith_Label"] == df["Predicted_Label"]).mean()
vsh_mae = (df["Vsh_PreTertiary"] - df["Predicted_Vsh"]).abs().mean()

st.subheader("Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Selected Classifier", model_name)
c2.metric("Depth Samples", f"{len(df):,}")
c3.metric("Classification Agreement", f"{agreement:.2%}")
c4.metric("Vsh MAE", f"{vsh_mae:.4f}")

st.divider()

tab1, tab2, tab3 = st.tabs([
    "Well Log Analysis",
    "Model Predictions",
    "Prediction Data"
])

with tab1:
    st.subheader("Petrophysical Well Log Tracks")

    tracks = [
        ("GR", "Gamma Ray"),
        ("Vsh_PreTertiary", "Actual Vsh"),
        ("RHOB", "Bulk Density"),
        ("DPOR", "Density Porosity"),
        ("CNPOR", "Neutron Porosity")
    ]

    tracks = [(col, label) for col, label in tracks if col in df.columns]

    fig, axes = plt.subplots(
        1, len(tracks),
        figsize=(3 * len(tracks), 10),
        sharey=True
    )

    if len(tracks) == 1:
        axes = [axes]

    for ax, (column, label) in zip(axes, tracks):
        ax.plot(df[column], df["DEPTH"], linewidth=0.7)
        ax.set_title(label)
        ax.set_xlabel(column)
        ax.grid(alpha=0.3)
        ax.invert_yaxis()

    axes[0].set_ylabel("Depth")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    st.subheader("Actual vs Predicted Vsh")

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.plot(
        df["Vsh_PreTertiary"],
        df["DEPTH"],
        label="Actual Vsh",
        linewidth=1
    )

    ax.plot(
        df["Predicted_Vsh"],
        df["DEPTH"],
        label="Predicted Vsh",
        linewidth=1
    )

    ax.set_xlabel("Vsh")
    ax.set_ylabel("Depth")
    ax.set_title("Vsh Regression Profile")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Original vs Predicted Lithology")

    fig, ax = plt.subplots(figsize=(8, 10))

    ax.step(
        df["Lith_Label"],
        df["DEPTH"],
        where="mid",
        label="Original Lithology",
        linewidth=1
    )

    ax.step(
        df["Predicted_Label"],
        df["DEPTH"],
        where="mid",
        label="Predicted Lithology",
        linewidth=1
    )

    ax.set_xticks(list(CLASS_NAMES))
    ax.set_xticklabels(list(CLASS_NAMES.values()), rotation=25, ha="right")
    ax.set_xlabel("Lithology")
    ax.set_ylabel("Depth")
    ax.set_title(f"Lithology Classification — {model_name}")
    ax.invert_yaxis()
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Lithology and Vsh Prediction Results")

    comparison = df[
        [
            "DEPTH",
            "Vsh_PreTertiary",
            "Predicted_Vsh",
            "Lith_Label",
            "Predicted_Label",
            "Original_Lithology",
            "Predicted_Lithology"
        ]
    ].rename(columns={"Vsh_PreTertiary": "Actual_Vsh"})

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Prediction Results",
        comparison.to_csv(index=False),
        file_name="lithology_prediction_results.csv",
        mime="text/csv"
    )
