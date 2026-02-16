
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

st.set_page_config(page_title="GOOG Stock Prediction - Group 6", layout="wide")
st.title("GOOG Stock Price Prediction")
st.markdown("### Comparing Deep Learning Models for Next-Day Price Forecasting")
st.markdown("**Group 6** | Deep LSTM, Wide LSTM, GRU, Stacked LSTM, Transformer, 1D CNN")

# Load saved results
with open("model_results.json", "r") as f:
    results = json.load(f)

model_names = list(results.keys())

# --- Sidebar ---
st.sidebar.header("Settings")
selected_models = st.sidebar.multiselect(
    "Select models to compare",
    model_names,
    default=model_names
)

# --- Metrics Table ---
st.header("Model Performance Comparison")
metrics_data = {
    name: {"MAE": results[name]["MAE"], "RMSE": results[name]["RMSE"], "R²": results[name]["R2"]}
    for name in selected_models
}
metrics_df = pd.DataFrame(metrics_data).T
metrics_df = metrics_df.sort_values("R²", ascending=False)

# Highlight best
st.dataframe(
    metrics_df.style.highlight_min(subset=["MAE", "RMSE"], color="lightgreen")
                     .highlight_max(subset=["R²"], color="lightgreen")
                     .format("{:.4f}"),
    use_container_width=True
)

# --- Bar Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("MAE & RMSE Comparison")
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(selected_models))
    width = 0.35
    mae_vals = [results[m]["MAE"] for m in selected_models]
    rmse_vals = [results[m]["RMSE"] for m in selected_models]
    ax1.bar(x - width/2, mae_vals, width, label="MAE", color="#4C72B0")
    ax1.bar(x + width/2, rmse_vals, width, label="RMSE", color="#DD8452")
    ax1.set_xticks(x)
    ax1.set_xticklabels(selected_models, rotation=45, ha="right")
    ax1.set_ylabel("Error")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    st.subheader("R² Score Comparison")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    r2_vals = [results[m]["R2"] for m in selected_models]
    colors = ["#55A868" if v == max(r2_vals) else "#4C72B0" for v in r2_vals]
    ax2.bar(selected_models, r2_vals, color=colors)
    ax2.set_ylim(min(r2_vals) - 0.02, 1.0)
    ax2.set_ylabel("R² Score")
    ax2.set_xticklabels(selected_models, rotation=45, ha="right")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig2)

# --- Predictions Plot ---
st.header("Predictions vs Actual Prices")
selected_model = st.selectbox("Select model to view predictions", selected_models)

fig3, ax3 = plt.subplots(figsize=(14, 5))
y_true = results[selected_model]["y_true"]
y_pred = results[selected_model]["y_pred"]
ax3.plot(y_true, label="Actual Price", linewidth=2, color="blue", alpha=0.8)
ax3.plot(y_pred, label="Predicted Price", linewidth=2, color="red", alpha=0.7, linestyle="--")
ax3.set_xlabel("Time Steps (Days)")
ax3.set_ylabel("Stock Price ($)")
ax3.set_title(f"{selected_model}: Actual vs Predicted")
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig3)

# Error distribution
st.subheader(f"{selected_model} - Prediction Error Distribution")
errors = np.array(y_true) - np.array(y_pred)
fig4, ax4 = plt.subplots(figsize=(8, 4))
ax4.hist(errors, bins=40, color="#4C72B0", edgecolor="black", alpha=0.7)
ax4.axvline(x=0, color="red", linestyle="--")
ax4.set_xlabel("Error ($)")
ax4.set_ylabel("Frequency")
ax4.set_title(f"Error Distribution (Mean: ${np.mean(errors):.2f}, Std: ${np.std(errors):.2f})")
plt.tight_layout()
st.pyplot(fig4)

# --- Training Curves ---
st.header("Training Curves")
selected_train = st.selectbox("Select model for training curve", selected_models, key="train")

fig5, ax5 = plt.subplots(figsize=(10, 4))
ax5.plot(results[selected_train]["train_loss"], label="Train Loss")
ax5.plot(results[selected_train]["val_loss"], label="Validation Loss")
ax5.set_xlabel("Epochs")
ax5.set_ylabel("Loss (MSE)")
ax5.set_title(f"{selected_train} Training Curve")
ax5.legend()
ax5.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig5)

# --- All Models Overlay ---
st.header("All Models Overlay")
fig6, ax6 = plt.subplots(figsize=(14, 6))
ax6.plot(results[selected_models[0]]["y_true"], label="Actual", linewidth=2, color="black")
colors_list = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
for idx, name in enumerate(selected_models):
    ax6.plot(results[name]["y_pred"], label=name, alpha=0.7, linewidth=1.5,
             color=colors_list[idx % len(colors_list)], linestyle="--")
ax6.set_xlabel("Time Steps (Days)")
ax6.set_ylabel("Stock Price ($)")
ax6.set_title("All Models: Predictions vs Actual")
ax6.legend()
ax6.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig6)

st.markdown("---")

