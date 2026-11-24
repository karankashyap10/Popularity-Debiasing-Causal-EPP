# CausalEPP Demo App

The Streamlit demo reads existing model checkpoints and data. It does not train models or modify project data.

From the repository root, install the demo dependency into the project environment and start the app:

```powershell
.\.venv\Scripts\python.exe -m pip install -r demo_app\requirements.txt
.\.venv\Scripts\python.exe -m streamlit run demo_app\app.py
```

The app selects the highest numbered `*_run-NNN` folder for CausalEPP, LLM CausalEPP, and LightGCN. The Recommendations tab accepts a mapped or raw user ID and returns top items with their model scores. The Test metrics tab displays standard and calibrated JSON metrics when those files exist. Run details shows the selected checkpoint and saved configuration.
