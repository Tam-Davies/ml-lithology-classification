
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "weeks1_10_processed.csv"
MODEL_DIR = ROOT / "artifacts" / "models"

FEATURES = [
    "GR", "RHOB", "DPOR", "RILD", "SP", "PE",
    "Vsh_PreTertiary"
]

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
    page_title="Lithology Classification",
    page_icon="🪨",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


df = load_data()

st.title("ML-Based Lithology Classification")
st.caption("Petrophysical well-log visualization and machine-learning predictions")

missing = [col for col in FEATURES + ["DEPTH", "Lith_Label"] if col not in df.columns]

if missing:
    st.error(f"Missing required columns: {missing}")
    st.stop()

model_name = st.selectbox(
    "Select classification model",
    list(MODEL_FILES.keys())
)

model_path = MODEL_DIR / MODEL_FILES[model_name]

if not model_path.exists():
    st.error(f"{model_name} model not found. Rerun model training first.")
    st.stop()

model = load_model(str(model_path))

X = df[FEATURES]
df["Predicted_Label"] = model.predict(X).astype(int)

df["Original_Lithology"] = df["Lith_Label"].map(CLASS_NAMES)
df["Predicted_Lithology"] = df["Predicted_Label"].map(CLASS_NAMES)

accuracy = (df["Lith_Label"] == df["Predicted_Label"]).mean()

col1, col2, col3 = st.columns(3)
col1.metric("Selected Model", model_name)
col2.metric("Rows Analysed", f"{len(df):,}")
col3.metric("Agreement with Original Labels", f"{accuracy:.2%}")

st.divider()

st.subheader("Well Log Display")

log_tracks = [
    ("GR", "Gamma Ray"),
    ("Vsh_PreTertiary", "Vshale"),
    ("RHOB", "Bulk Density"),
    ("DPOR", "Density Porosity"),
    ("CNPOR", "Neutron Porosity")
]

available_tracks = [
    (column, label)
    for column, label in log_tracks
    if column in df.columns
]

fig, axes = plt.subplots(
    1, len(available_tracks),
    figsize=(3 * len(available_tracks), 9),
    sharey=True
)

if len(available_tracks) == 1:
    axes = [axes]

for ax, (column, label) in zip(axes, available_tracks):
    ax.plot(df[column], df["DEPTH"], linewidth=0.7)
    ax.set_title(label)
    ax.set_xlabel(column)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

axes[0].set_ylabel("Depth")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

st.subheader("Original vs Predicted Lithology")

fig, ax = plt.subplots(figsize=(7, 10))

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

ax.set_xticks(list(CLASS_NAMES.keys()))
ax.set_xticklabels(list(CLASS_NAMES.values()), rotation=25, ha="right")
ax.set_xlabel("Lithology")
ax.set_ylabel("Depth")
ax.set_title(f"Original and Predicted Lithology — {model_name}")
ax.invert_yaxis()
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.divider()

st.subheader("Lithology Comparison Table")

comparison = df[
    [
        "DEPTH",
        "Original_Lithology",
        "Predicted_Lithology",
        *FEATURES
    ]
].sort_values("DEPTH")

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download Prediction Results",
    comparison.to_csv(index=False),
    file_name="lithology_predictions.csv",
    mime="text/csv"
)
