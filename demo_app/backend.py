"""Read-only inference helpers for the CausalEPP Streamlit demo."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from forecasting.moving_average import forecast_all
from models import CausalEPP, LightGCN


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs" / "Ciao"
DATA = ROOT / "data" / "Ciao"
RUN_TYPES = {"CausalEPP": "causalepp", "LLM CausalEPP": "causalepp_llm", "LightGCN": "lightgcn"}


def latest_runs():
    result = {}
    for label, prefix in RUN_TYPES.items():
        matches = []
        for path in OUTPUTS.glob(f"{prefix}_run-*"):
            found = re.fullmatch(re.escape(prefix) + r"_run-(\d+)", path.name)
            if found:
                matches.append((int(found.group(1)), path))
        if matches:
            result[label] = max(matches, key=lambda pair: pair[0])[1]
    return result


def _read_data():
    train = pd.read_csv(DATA / "train_data.csv", sep="\t")
    val = pd.read_csv(DATA / "val_data.csv", sep="\t")
    test = pd.read_csv(DATA / "test_data.csv", sep="\t")
    users = pd.read_csv(DATA / "user_mapping.csv")
    items = pd.read_csv(DATA / "item_mapping.csv")
    return train, val, test, users, items


def user_choices():
    _, _, _, mappings, _ = _read_data()
    return [(int(row.user), str(row.raw_user)) for row in mappings.itertuples()]


def load_model(run_dir: Path, device="cpu"):
    train, val, test, users, items = _read_data()
    checkpoint_path = run_dir / ("lightgcn.pt" if run_dir.name.startswith("lightgcn") else "causalepp_ciao.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = checkpoint.get("config", {})
    num_users, num_items = int(checkpoint["num_users"]), int(checkpoint["num_items"])
    is_lightgcn = run_dir.name.startswith("lightgcn")
    dim, layers = int(cfg.get("embedding_dim", 64)), int(cfg.get("lightgcn_layers", 3))
    if is_lightgcn:
        model = LightGCN(num_users, num_items, dim, layers).to(device)
    else:
        model = CausalEPP(num_users, num_items, dim, layers,
                          int(cfg.get("mlp_hidden_dim", 32)),
                          float(cfg.get("alpha", 0.5))).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    graph = LightGCN.normalized_adjacency(train, num_users, num_items, device)
    names = dict(zip(items.item.astype(int), items.raw_item.astype(str)))
    counts = np.load(run_dir / "global_item_counts.npy") if (run_dir / "global_item_counts.npy").exists() else np.bincount(train.item.astype(int), minlength=num_items)
    return {"model": model, "graph": graph, "train": train, "val": val, "test": test,
            "num_users": num_users, "num_items": num_items, "names": names,
            "counts": counts, "config": cfg, "is_lightgcn": is_lightgcn,
            "run_dir": run_dir, "user_mappings": users}


def recommend(state, user_id, top_k=20, forecast_method=None):
    model, graph = state["model"], state["graph"]
    user_id = int(user_id)
    train = state["train"]
    seen = set(train.loc[train.user == user_id, "item"].astype(int))
    counts = state["counts"]
    with torch.no_grad():
        lightgcn = model if state["is_lightgcn"] else model.lightgcn
        user_emb, item_emb = lightgcn.propagate(graph)
        available_matching = None
        if state["is_lightgcn"]:
            scores = user_emb[user_id] @ item_emb.T
            score_source = "LightGCN dot product"
        else:
            edges = np.load(DATA / "time_bin_edges.npy")
            pop_hist = np.load(DATA / "item_local_popularity.npy")
            epp_hist = np.load(DATA / "user_epp.npy")
            p_star, s_star = forecast_all(
                pop_hist, epp_hist,
                window=int(state["config"].get("moving_average_window", 10)),
                item_horizon=float(state["config"].get("item_forecast_horizon", 5.0)),
                user_horizon=float(state["config"].get("user_forecast_horizon", 10.0)),
                method=forecast_method or state["config"].get("forecast_method", "moving_average"),
                level_alpha=float(state["config"].get("forecast_level_alpha", 0.3)),
                trend_beta=float(state["config"].get("forecast_trend_beta", 0.1)))
            matching = (user_emb[user_id].unsqueeze(0) * item_emb).sum(-1)
            if state["config"].get("use_llm_matching"):
                cache_path = Path(state["config"].get("llm_combined_path", ""))
                if not cache_path.is_absolute(): cache_path = ROOT / cache_path
                if not cache_path.exists(): cache_path = OUTPUTS / ".llm_cache" / "combined.json"
                cached = json.loads(cache_path.read_text())
                row = cached.get(str(user_id), {})
                matching = torch.full_like(matching, float("-inf"))
                for item, score in row.items():
                    if int(item) < len(matching): matching[int(item)] = float(score)
                available_matching = torch.isfinite(matching)
                score_source = "Cached LLM matching score + CausalEPP terms"
            else:
                score_source = "CausalEPP final score"
            p = torch.as_tensor(p_star, device=matching.device).clamp(0, 1)
            s = torch.full_like(p, float(np.clip(s_star[user_id], 0, 1)))
            conform = torch.exp(-model.alpha * torch.abs(s - p)) * p * model.item_mlp(item_emb).squeeze(-1)
            scores = torch.tanh(model.q + conform) * F.softplus(matching)
            if available_matching is not None:
                scores[~available_matching] = -torch.inf
        scores = scores.clone()
        if seen: scores[list(seen)] = -torch.inf
        candidates = torch.isfinite(scores).nonzero().flatten()
        k = min(int(top_k), candidates.numel())
        top = candidates[torch.topk(scores[candidates], k).indices]
        vals = scores[top].cpu().numpy()
        ids = top.cpu().numpy().astype(int)
    frame = pd.DataFrame({"item_id": ids, "item": [state["names"].get(i, str(i)) for i in ids],
                          "score": vals})
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame, score_source


def metrics_for(run_dir: Path):
    base_path = run_dir / "test_metrics.json"
    calibrated_path = run_dir / "popularity_calibrated_test_metrics.json"
    base = json.loads(base_path.read_text()) if base_path.exists() else None
    calibrated = json.loads(calibrated_path.read_text()) if calibrated_path.exists() else None
    return base, calibrated
