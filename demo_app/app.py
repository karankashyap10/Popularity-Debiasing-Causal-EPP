"""Streamlit interface for comparing Ciao recommendation models."""
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from demo_app.backend import OUTPUTS, load_model, recommend, user_choices


st.set_page_config(page_title="Ciao recommender comparison", page_icon="🎬", layout="wide")
st.title("Ciao recommender comparison")
st.caption("Compare personalized recommendations and saved test results across model variants.")

REQUESTED_RUNS = [
    ("LightGCN", "lightgcn", 6, "lightgcn.pt"),
    ("CausalEPP", "causalepp", 4, "causalepp_ciao.pt"),
    ("LLM baseline", "causalepp_llm", 3, "causalepp_ciao.pt"),
    ("CausalEPP · Holt", "causalepp", 5, "causalepp_ciao.pt"),
    ("LLM baseline · Holt", "causalepp_llm", 4, "causalepp_ciao.pt"),
    ("CausalEPP · calibrated", "causalepp", 7, "causalepp_ciao.pt"),
    ("LLM baseline · calibrated", "causalepp_llm", 5, "causalepp_ciao.pt"),
]

def resolve_run(prefix, number, checkpoint):
    expected = f"{prefix}_run-{number:03d}"
    matches = [p for p in OUTPUTS.iterdir() if p.is_dir() and p.name == expected]
    return matches[0] if matches and (matches[0] / checkpoint).exists() else None


resolved_runs = [(label, prefix, number, checkpoint,
                  resolve_run(prefix, number, checkpoint))
                 for label, prefix, number, checkpoint in REQUESTED_RUNS]

@st.cache_resource(show_spinner="Loading model checkpoint…")
def cached_model(folder):
    return load_model(Path(folder), device="cpu")


def model_specs():
    return [(label, folder, None) for label, _, _, _, folder in resolved_runs if folder]


specs = model_specs()
tab_recs, tab_compare = st.tabs(["Recommendations", "Model and calibration comparison"])

with tab_recs:
    st.caption("Using only the seven requested numbered runs. Their saved Holt settings are used directly.")
    missing = [label for label, _, _, _, folder in resolved_runs if folder is None]
    if missing:
        st.warning("Missing requested checkpoint(s): " + ", ".join(missing))
    choices = user_choices()
    selected = st.selectbox(
        "User", choices,
        format_func=lambda pair: f"Mapped ID {pair[0]} · Raw ID {pair[1]}",
    )
    k = st.number_input("Number of recommendations", min_value=5, max_value=100, value=20, step=5)
    if not specs:
        st.info("No compatible model checkpoints were found.")
    elif st.button("Generate recommendations", type="primary"):
        for start in range(0, len(specs), 2):
            cols = st.columns(2)
            for col, (label, folder, method) in zip(cols, specs[start:start + 2]):
                with col:
                    st.subheader(label)
                    try:
                        state = cached_model(str(folder))
                        frame, source = recommend(state, selected[0], int(k), method)
                        st.caption(f"{source} · {folder.name}")
                        st.dataframe(
                            frame, width="stretch", hide_index=True,
                            column_config={"score": st.column_config.NumberColumn(format="%.5g")},
                        )
                    except Exception as exc:
                        st.warning(f"Could not create this list: {exc}")

with tab_compare:
    st.subheader("Saved evaluation metrics")
    st.caption("Base and calibration results are read directly from the JSON files in the selected run folders.")
    metric_rows = []
    missing_metrics = []
    for label, _, _, _, folder in resolved_runs:
        if folder is None:
            missing_metrics.append(f"{label}: requested run folder not found")
            continue
        metrics_path = folder / "test_metrics.json"
        if not metrics_path.exists():
            missing_metrics.append(f"{label}: test_metrics.json not found in {folder.name}")
            continue
        saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metric_rows.append({"Model variant": label, "Output folder": folder.name,
                            "Result": "Base", "Method": "Original", "User group": "All",
                            **saved_metrics})

        calibration_path = folder / "popularity_calibrated_test_metrics.json"
        if calibration_path.exists():
            saved_calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            for method_name, method_results in saved_calibration.items():
                if not isinstance(method_results, dict):
                    continue
                # Calibration JSONs store metrics by user group. Keep every saved group.
                for group_name, group_metrics in method_results.items():
                    if isinstance(group_metrics, dict) and "NDCG@K" in group_metrics:
                        metric_rows.append({
                            "Model variant": label,
                            "Output folder": folder.name,
                            "Result": "Calibrated",
                            "Method": method_name,
                            "User group": group_name,
                            **group_metrics,
                        })
    if metric_rows:
        st.dataframe(pd.DataFrame(metric_rows), width="stretch", hide_index=True)
    if missing_metrics:
        for message in missing_metrics:
            st.warning(message)
