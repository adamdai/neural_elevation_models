import streamlit as st
import json
import time
import pandas as pd
from pathlib import Path

STATUS_FILE = "/home/addai/Projects/neural_elevation_models/outputs/sweep_status.json"

st.set_page_config(page_title="NEMo Experiment Dashboard", layout="wide")

st.title("🚀 NEMo Experiment Sweep Dashboard")

def load_status():
    if not Path(STATUS_FILE).exists():
        return None
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except:
        return None

# Sidebar for automatic refresh
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 2)

status = load_status()

if status:
    st.header(status.get("sweep_name", "NEMo Sweep"))
    
    # 1. Overall Progress
    st.subheader("Total Progress")
    idx = status["current_index"]
    total = len(status["experiments"])
    progress_val = (idx + (status["current_progress"] / 100.0)) / total if not status["is_complete"] else 1.0
    st.progress(progress_val)
    st.write(f"Completed {idx if not status['is_complete'] else total} / {total} experiments")

    # 2. Current Experiment
    if not status["is_complete"]:
        st.divider()
        st.subheader(f"Current: `{status['current_name']}`")
        st.progress(status["current_progress"] / 100.0)
        st.write(f"Training Progress: {status['current_progress']:.1f}%")

    # 3. Metrics Table
    st.divider()
    st.subheader("📊 Results Summary")
    
    rows = []
    for exp in status["experiments"]:
        row = {
            "Experiment": exp["name"],
            "Status": exp["status"]
        }
        if exp["metrics"]:
            m = exp["metrics"]
            row.update({
                "RMSE_u": m.get("rmse_u"),
                "MAE_u": m.get("mae_u"),
                "Slope_Err": f"{m.get('slope_err')}°",
                "Norm_Err": f"{m.get('norm_err')}°",
                "Bias": m.get("bias")
            })
        rows.append(row)
    
    df = pd.DataFrame(rows)
    st.table(df)

    if status["is_complete"]:
        st.success("✅ Sweep Complete!")
else:
    st.info("Waiting for status file... Ensure `run_experiments.py` is running.")

# Simple JavaScript to trigger refresh
time.sleep(refresh_rate)
st.rerun()
