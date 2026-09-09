import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

DATASET = "rockfall_synthetic_dataset.csv"
RANDOM_STATE = 42

# -----------------------------
# 1. Load and prepare dataset
# -----------------------------
df = pd.read_csv(DATASET)
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# The supplied synthetic dataset has no real Rockfall target.
# Therefore, this demo creates a transparent risk label for demonstration.
def make_risk_score(data):
    d = data.copy()
    eps = 1e-9
    score = (
        (d["Rainfall"] / (d["Rainfall"].max() + eps)) * 20
        + (d["Slope_Angle"] / 90.0) * 25
        + (d["Soil_Moisture"] / (d["Soil_Moisture"].max() + eps)) * 15
        + (d["Change_in_NDVI"].abs() / (d["Change_in_NDVI"].abs().max() + eps)) * 10
        + (d["Blast_Vibration"] / (d["Blast_Vibration"].max() + eps)) * 15
        + (d["Seismic_Vibration"] / (d["Seismic_Vibration"].max() + eps)) * 15
    )
    return score

df["Risk_Score"] = make_risk_score(df)
df["Risk_Level"] = pd.cut(
    df["Risk_Score"],
    bins=[-np.inf, 35, 65, np.inf],
    labels=["Low", "Medium", "High"]
)

FEATURES = [
    "Rock_Type", "Rainfall", "Slope_Angle", "NDVI",
    "Change_in_NDVI", "Soil_Moisture",
    "Blast_Vibration", "Seismic_Vibration",
    "Year", "Month", "Day"
]
TARGET = "Risk_Level"

X = df[FEATURES]
y = df[TARGET].astype(str)

cat_features = ["Rock_Type"]
num_features = [c for c in FEATURES if c not in cat_features]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", StandardScaler(), num_features)
])

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

report = classification_report(y_test, y_pred, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=["Low", "Medium", "High"])

# -----------------------------
# 2. Plot helpers
# -----------------------------
def save_plot(fig, name):
    path = os.path.join("rockfall_dashboard", name)
    os.makedirs("rockfall_dashboard", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def risk_distribution():
    counts = df["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Synthetic Rockfall Risk Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Areas/Records")
    ax.tick_params(axis="x", rotation=0)
    return fig

def performance_plot():
    metrics = [accuracy, precision, recall, f1]
    names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, metrics)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Random Forest Model Performance")
    for b, v in zip(bars, metrics):
        ax.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.2%}",
                ha="center", fontweight="bold")
    return fig

def confusion_plot():
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(3), ["Low", "Medium", "High"])
    ax.set_yticks(range(3), ["Low", "Medium", "High"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    return fig

def feature_importance_plot():
    # Recover transformed feature names from the fitted preprocessor.
    names = list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    importances = pipeline.named_steps["model"].feature_importances_
    imp = pd.Series(importances, index=names).sort_values(ascending=True).tail(12)

    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind="barh", ax=ax)
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    return fig

def rainfall_slope_plot():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(
        df["Rainfall"], df["Slope_Angle"],
        c=df["Risk_Score"], cmap="viridis",
        s=45, alpha=0.8
    )
    ax.set_title("Rainfall vs Slope Angle — Risk Score")
    ax.set_xlabel("Rainfall")
    ax.set_ylabel("Slope Angle (degrees)")
    fig.colorbar(sc, ax=ax, label="Risk Score")
    return fig

def soil_slope_plot():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(
        df["Soil_Moisture"], df["Slope_Angle"],
        c=df["Risk_Score"], cmap="plasma",
        s=45, alpha=0.8
    )
    ax.set_title("Soil Moisture vs Slope Angle — Risk Score")
    ax.set_xlabel("Soil Moisture")
    ax.set_ylabel("Slope Angle (degrees)")
    fig.colorbar(sc, ax=ax, label="Risk Score")
    return fig

# -----------------------------
# 3. Prediction function
# -----------------------------
def predict_area(
    rock_type, rainfall, slope_angle, ndvi,
    change_ndvi, soil_moisture, blast_vibration, seismic_vibration
):
    row = pd.DataFrame([{
        "Rock_Type": rock_type,
        "Rainfall": rainfall,
        "Slope_Angle": slope_angle,
        "NDVI": ndvi,
        "Change_in_NDVI": change_ndvi,
        "Soil_Moisture": soil_moisture,
        "Blast_Vibration": blast_vibration,
        "Seismic_Vibration": seismic_vibration,
        "Year": int(df["Year"].median()),
        "Month": int(df["Month"].median()),
        "Day": int(df["Day"].median())
    }])

    # Same transparent score formula for the dashboard explanation.
    eps = 1e-9
    score = (
        (rainfall / (df["Rainfall"].max() + eps)) * 20
        + (slope_angle / 90.0) * 25
        + (soil_moisture / (df["Soil_Moisture"].max() + eps)) * 15
        + (abs(change_ndvi) / (df["Change_in_NDVI"].abs().max() + eps)) * 10
        + (blast_vibration / (df["Blast_Vibration"].max() + eps)) * 15
        + (seismic_vibration / (df["Seismic_Vibration"].max() + eps)) * 15
    )

    predicted = pipeline.predict(row)[0]
    probs = pipeline.predict_proba(row)[0]
    classes = pipeline.named_steps["model"].classes_
    prob_df = pd.DataFrame({
        "Risk Level": classes,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    return (
        f"### 🚨 Predicted Risk: **{predicted.upper()}**\n\n"
        f"**Risk score:** {score:.1f} / 100\n\n"
        f"Model confidence for predicted class: **{probs[list(classes).index(predicted)]:.1%}**",
        prob_df,
        risk_gauge(score)
    )

def risk_gauge(score):
    fig, ax = plt.subplots(figsize=(8, 1.7))
    ax.barh(["Risk"], [100], alpha=0.15)
    ax.barh(["Risk"], [min(max(score, 0), 100)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Risk Score (0–100)")
    ax.set_title("Rockfall Risk Gauge")
    ax.axvline(35, linestyle="--", linewidth=1.5)
    ax.axvline(65, linestyle="--", linewidth=1.5)
    ax.text(17.5, 0, "LOW", ha="center", va="center", fontweight="bold")
    ax.text(50, 0, "MEDIUM", ha="center", va="center", fontweight="bold")
    ax.text(82.5, 0, "HIGH", ha="center", va="center", fontweight="bold")
    return fig

# -----------------------------
# 4. Gradio dashboard
# -----------------------------
custom_css = """
.gradio-container {
    max-width: 1250px !important;
}
.hero {
    padding: 22px;
    border-radius: 18px;
    margin-bottom: 16px;
}
.kpi {
    text-align: center;
    padding: 15px;
    border-radius: 14px;
    border: 1px solid rgba(127,127,127,.25);
}
.note {
    padding: 12px;
    border-radius: 12px;
    border-left: 4px solid #f59e0b;
}
"""

with gr.Blocks(css=custom_css, title="AI Rockfall Prediction Dashboard") as demo:
    gr.HTML("""
    <div class="hero">
        <h1>🪨 AI-Based Rockfall Prediction Dashboard</h1>
        <p>
        Predict rockfall risk from rainfall, slope angle, vegetation change,
        soil moisture, blast vibration and seismic vibration.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(elem_classes="kpi"):
            gr.Markdown("### 🤖 Model")
            gr.Markdown("**Random Forest**")
        with gr.Column(elem_classes="kpi"):
            gr.Markdown("### 🎯 Test Accuracy")
            gr.Markdown(f"**{accuracy:.2%}**")
        with gr.Column(elem_classes="kpi"):
            gr.Markdown("### 📊 Records")
            gr.Markdown(f"**{len(df)}**")
        with gr.Column(elem_classes="kpi"):
            gr.Markdown("### ⚠️ High Risk")
            gr.Markdown(f"**{int((df['Risk_Level']=='High').sum())}**")

    gr.Markdown("""
    <div class="note">
    <b>Important:</b> The uploaded synthetic dataset does not contain a real
    rockfall-event target column. The dashboard therefore uses a transparent
    risk-score rule to create demonstration labels. The reported accuracy is
    accuracy against those generated labels, not real-world rockfall accuracy.
    </div>
    """)

    with gr.Tab("🔮 Predict New Area"):
        gr.Markdown("## Enter area conditions")
        with gr.Row():
            with gr.Column():
                rock = gr.Dropdown(
                    ["Igneous", "Metamorphic", "Sedimentary"],
                    value="Metamorphic",
                    label="Rock Type"
                )
                rain = gr.Slider(0, 50, value=30, step=0.1, label="Rainfall")
                slope = gr.Slider(5, 70, value=40, step=0.1, label="Slope Angle (°)")
                ndvi = gr.Slider(0.10, 0.70, value=0.40, step=0.01, label="NDVI")
            with gr.Column():
                change = gr.Slider(-0.05, 0.05, value=-0.02, step=0.001, label="Change in NDVI")
                moisture = gr.Slider(10, 40, value=25, step=0.1, label="Soil Moisture")
                blast = gr.Slider(0, 0.30, value=0.15, step=0.001, label="Blast Vibration")
                seismic = gr.Slider(0, 0.05, value=0.025, step=0.001, label="Seismic Vibration")

        predict_btn = gr.Button("🚨 Predict Rockfall Risk", variant="primary")
        result = gr.Markdown()
        probabilities = gr.Dataframe(
            headers=["Risk Level", "Probability"],
            datatype=["str", "number"],
            label="Class Probabilities",
            interactive=False
        )
        gauge = gr.Plot(label="Risk Gauge")

        predict_btn.click(
            predict_area,
            inputs=[rock, rain, slope, ndvi, change, moisture, blast, seismic],
            outputs=[result, probabilities, gauge]
        )

    with gr.Tab("📈 Model Performance"):
        gr.Markdown("## Model accuracy and evaluation")
        with gr.Row():
            gr.Plot(performance_plot(), label="Performance")
            gr.Plot(confusion_plot(), label="Confusion Matrix")
        gr.Markdown("### Classification Report")
        gr.Code(report, language="text")

    with gr.Tab("📊 Data Insights"):
        gr.Markdown("## Dataset visualizations")
        with gr.Row():
            gr.Plot(risk_distribution(), label="Risk Distribution")
            gr.Plot(feature_importance_plot(), label="Feature Importance")
        with gr.Row():
            gr.Plot(rainfall_slope_plot(), label="Rainfall vs Slope")
            gr.Plot(soil_slope_plot(), label="Soil Moisture vs Slope")

    with gr.Tab("📋 Dataset"):
        gr.Markdown("## Processed dataset")
        display_cols = [
            "Rock_Type", "Date", "Rainfall", "Slope_Angle", "NDVI",
            "Change_in_NDVI", "Soil_Moisture", "Blast_Vibration",
            "Seismic_Vibration", "Risk_Score", "Risk_Level"
        ]
        gr.Dataframe(
            df[display_cols].round(4),
            interactive=False,
            wrap=True,
            label="Rockfall Dataset"
        )

    gr.Markdown("""
    ### 🧠 Project idea
    This dashboard is suitable as a college AI/ML project prototype.
    For a real deployment, replace the generated Risk_Level labels with
    historical rockfall occurrence labels and add geospatial/topographic data.
    """)

if __name__ == "__main__":
    demo.launch()
