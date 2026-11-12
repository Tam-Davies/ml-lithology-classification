
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "weeks1_10_processed.csv"
MODEL_DIR = ROOT / "artifacts" / "models"

FEATURES = [
    "GR", "RHOB", "DPOR", "RILD", "SP", "PE",
    "Vsh_PreTertiary"
]

VSH_FEATURES = [
    "GR", "RHOB", "DPOR", "RILD", "SP", "PE"
]

CLASS_NAMES = {
    0: "Clean Sand",
    1: "Silty Sand",
    2: "Sandy Shale",
    3: "Shale"
}

LITHOLOGY_COLORS = {
    0: "#F2D34F",
    1: "#E9953B",
    2: "#A66A3F",
    3: "#555555"
}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

st.set_page_config(
    page_title="Lithology Classification Dashboard",
    page_icon="🪨",
    layout="wide"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model(path):
    return joblib.load(path)


df = load_data().sort_values("DEPTH").copy()

st.title("Lithology Classification Dashboard")
st.caption(
    "Machine Learning-Based Lithology Classification and Vshale Prediction"
)

with st.sidebar:
    st.subheader("Model Selection")

    model_name = st.selectbox(
        "Classification Model",
        list(MODEL_FILES.keys())
    )

    st.divider()

    st.caption(
        "Vshale regression uses the trained Random Forest regressor."
    )


model_path = MODEL_DIR / MODEL_FILES[model_name]
vsh_path = MODEL_DIR / "vsh_regressor.pkl"

if not model_path.exists() or not vsh_path.exists():
    st.error("Required model files are missing. Run model training first.")
    st.stop()


required_columns = FEATURES + VSH_FEATURES + [
    "DEPTH",
    "Lith_Label"
]

missing = [
    col for col in set(required_columns)
    if col not in df.columns
]

if missing:
    st.error(f"Missing dataset columns: {missing}")
    st.stop()


classifier = load_model(str(model_path))
vsh_model = load_model(str(vsh_path))

df["Predicted_Label"] = classifier.predict(
    df[FEATURES]
).astype(int)

df["Predicted_Vsh"] = vsh_model.predict(
    df[VSH_FEATURES]
)


# MODEL PERFORMANCE

accuracy = accuracy_score(
    df["Lith_Label"],
    df["Predicted_Label"]
)

precision = precision_score(
    df["Lith_Label"],
    df["Predicted_Label"],
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    df["Lith_Label"],
    df["Predicted_Label"],
    average="weighted",
    zero_division=0
)

st.subheader("Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("Accuracy", f"{accuracy:.3f}")
c2.metric("Precision", f"{precision:.3f}")
c3.metric("F1 Score", f"{f1:.3f}")

st.divider()


# DEPTH SELECTION

st.subheader("Actual vs Predicted Well Profiles")

depth_min = float(df["DEPTH"].min())
depth_max = float(df["DEPTH"].max())

depth_range = st.slider(
    "Select Depth Interval",
    min_value=depth_min,
    max_value=depth_max,
    value=(depth_min, depth_max),
    step=0.5
)

view = df[
    df["DEPTH"].between(
        depth_range[0],
        depth_range[1]
    )
].copy()


# FOUR ALIGNED TRACKS

fig = make_subplots(
    rows=1,
    cols=4,
    shared_yaxes=True,
    horizontal_spacing=0.025,
    column_widths=[0.39, 0.11, 0.39, 0.11],
    subplot_titles=[
        "Actual Vshale",
        "Actual Lithology",
        "Predicted Vshale",
        "Predicted Lithology"
    ]
)


# ACTUAL VSHale CURVE

fig.add_trace(
    go.Scatter(
        x=view["Vsh_PreTertiary"],
        y=view["DEPTH"],
        mode="lines",
        name="Actual Vshale",
        line=dict(
            color="#1769AA",
            width=1.4
        ),
        hovertemplate=(
            "Depth: %{y:.2f} ft<br>"
            "Actual Vshale: %{x:.4f}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=1
)


# ACTUAL LITHOLOGY TRACK

actual_customdata = [
    [CLASS_NAMES[int(v)]]
    for v in view["Lith_Label"]
]

fig.add_trace(
    go.Heatmap(
        x=[0],
        y=view["DEPTH"],
        z=[[int(v)] for v in view["Lith_Label"]],
        customdata=actual_customdata,
        zmin=-0.5,
        zmax=3.5,
        colorscale=[
            [0.00, LITHOLOGY_COLORS[0]],
            [0.25, LITHOLOGY_COLORS[0]],
            [0.25, LITHOLOGY_COLORS[1]],
            [0.50, LITHOLOGY_COLORS[1]],
            [0.50, LITHOLOGY_COLORS[2]],
            [0.75, LITHOLOGY_COLORS[2]],
            [0.75, LITHOLOGY_COLORS[3]],
            [1.00, LITHOLOGY_COLORS[3]]
        ],
        showscale=False,
        xgap=0,
        ygap=0,
        hovertemplate=(
            "Depth: %{y:.2f} ft<br>"
            "Actual Lithology: %{customdata[0]}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=2
)


# PREDICTED VSHale CURVE

fig.add_trace(
    go.Scatter(
        x=view["Predicted_Vsh"],
        y=view["DEPTH"],
        mode="lines",
        name="Predicted Vshale",
        line=dict(
            color="#16856B",
            width=1.4
        ),
        hovertemplate=(
            "Depth: %{y:.2f} ft<br>"
            "Predicted Vshale: %{x:.4f}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=3
)


# PREDICTED LITHOLOGY TRACK

predicted_customdata = [
    [CLASS_NAMES[int(v)]]
    for v in view["Predicted_Label"]
]

fig.add_trace(
    go.Heatmap(
        x=[0],
        y=view["DEPTH"],
        z=[[int(v)] for v in view["Predicted_Label"]],
        customdata=predicted_customdata,
        zmin=-0.5,
        zmax=3.5,
        colorscale=[
            [0.00, LITHOLOGY_COLORS[0]],
            [0.25, LITHOLOGY_COLORS[0]],
            [0.25, LITHOLOGY_COLORS[1]],
            [0.50, LITHOLOGY_COLORS[1]],
            [0.50, LITHOLOGY_COLORS[2]],
            [0.75, LITHOLOGY_COLORS[2]],
            [0.75, LITHOLOGY_COLORS[3]],
            [1.00, LITHOLOGY_COLORS[3]]
        ],
        showscale=False,
        xgap=0,
        ygap=0,
        hovertemplate=(
            "Depth: %{y:.2f} ft<br>"
            "Predicted Lithology: %{customdata[0]}"
            "<extra></extra>"
        )
    ),
    row=1,
    col=4
)


# AXIS FORMATTING

fig.update_xaxes(
    title_text="Vshale",
    range=[0, 1],
    fixedrange=True,
    showgrid=True,
    gridcolor="#E5E5E5",
    row=1,
    col=1
)

fig.update_xaxes(
    range=[-0.5, 0.5],
    showticklabels=False,
    showgrid=False,
    zeroline=False,
    fixedrange=True,
    row=1,
    col=2
)

fig.update_xaxes(
    title_text="Vshale",
    range=[0, 1],
    fixedrange=True,
    showgrid=True,
    gridcolor="#E5E5E5",
    row=1,
    col=3
)

fig.update_xaxes(
    range=[-0.5, 0.5],
    showticklabels=False,
    showgrid=False,
    zeroline=False,
    fixedrange=True,
    row=1,
    col=4
)

fig.update_yaxes(
    title_text="Depth (ft)",
    autorange="reversed",
    fixedrange=True,
    showgrid=True,
    gridcolor="#E5E5E5",
    row=1,
    col=1
)

for col in [2, 3, 4]:
    fig.update_yaxes(
        matches="y",
        showticklabels=False,
        fixedrange=True,
        row=1,
        col=col
    )


# CHART LAYOUT

fig.update_layout(
    height=950,
    hovermode="closest",
    dragmode=False,
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(
        family="Arial",
        size=11,
        color="#222222"
    ),
    margin=dict(
        l=55,
        r=25,
        t=90,
        b=45
    ),
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={
        "displayModeBar": False,
        "scrollZoom": False,
        "staticPlot": False,
        "responsive": True
    }
)


# LITHOLOGY LEGEND

st.markdown("**Lithology Legend**")

legend_columns = st.columns(4)

for column, (class_id, class_name) in zip(
    legend_columns,
    CLASS_NAMES.items()
):
    column.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            gap:8px;
            padding:8px;
            border:1px solid #dddddd;
            border-radius:5px;
            background:white;
            color:#222222;
        ">
            <div style="
                width:18px;
                height:18px;
                background:{LITHOLOGY_COLORS[class_id]};
                border:1px solid #777777;
            "></div>
            <span>{class_name}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


# PREDICTION RESULTS

st.divider()

st.subheader("Lithology Prediction Results")

comparison = df[
    [
        "DEPTH",
        "Vsh_PreTertiary",
        "Predicted_Vsh",
        "Lith_Label",
        "Predicted_Label"
    ]
].copy()

comparison["Actual Lithology"] = comparison[
    "Lith_Label"
].map(CLASS_NAMES)

comparison["Predicted Lithology"] = comparison[
    "Predicted_Label"
].map(CLASS_NAMES)

comparison = comparison.rename(
    columns={
        "Vsh_PreTertiary": "Actual Vshale"
    }
)

st.dataframe(
    comparison[
        [
            "DEPTH",
            "Actual Vshale",
            "Predicted_Vsh",
            "Actual Lithology",
            "Predicted Lithology"
        ]
    ],
    use_container_width=True,
    hide_index=True
)

st.download_button(
    "Download Prediction Results",
    comparison.to_csv(index=False),
    file_name="lithology_predictions.csv",
    mime="text/csv"
)
