
"""Full-sort ranking evaluation for CausalEPP."""

import numpy as np
import torch
import torch.nn.functional as F


def ranking_metrics(
    model,
    norm_adj,
    train_df,
    eval_df,
    num_users,
    num_items,
    item_pop,
    user_epp,
    k=20,
    device="cpu",
):
    """
    Evaluate recommendations using full-sort ranking.

    Uses forecasted item popularity p_i* and user EPP s_u*.
    Excludes items already interacted with during training.
    """

    if k < 1:
        raise ValueError("k must be positive.")

    model.eval()

    seen = (
        train_df.groupby("user")["item"]
        .apply(set)
        .to_dict()
    )

    truth = (
        eval_df.groupby("user")["item"]
        .apply(set)
        .to_dict()
    )

    item_pop = torch.as_tensor(
        item_pop, dtype=torch.float32, device=device
    )

    user_epp = torch.as_tensor(
        user_epp, dtype=torch.float32, device=device
    )

    if item_pop.numel() != num_items:
        raise ValueError("item_pop must contain one value per item.")

    if user_epp.numel() < num_users:
        raise ValueError("user_epp must contain one value per user.")

    recalls = []
    precisions = []
    ndcgs = []
    recommended_items = set()

    items = torch.arange(
        num_items, dtype=torch.long, device=device
    )

    with torch.no_grad():
        user_embeddings, item_embeddings = model.lightgcn.propagate(
            norm_adj
        )

        for user, actual_items in truth.items():

            if not actual_items or user >= num_users:
                continue

            users = torch.full(
                (num_items,),
                int(user),
                dtype=torch.long,
                device=device,
            )

            # Equation (6): matching score m_ui.
            matching = (
                user_embeddings[users] * item_embeddings[items]
            ).sum(dim=-1)

            # Forecasted popularity and EPP: p_i* and s_u*.
            p = item_pop[items].clamp(0.0, 1.0)
            s = user_epp[users].clamp(0.0, 1.0)

            # Equation (17): forecasted conformity c_ui*.
            item_mapping = model.item_mlp(
                item_embeddings[items]
            ).squeeze(-1)

            conformity = (
                torch.exp(-model.alpha * torch.abs(s - p))
                * p
                * item_mapping
            )

            # Equation (16): final inference recommendation score.
            scores = (
                torch.tanh(model.q[items] + conformity)
                * F.softplus(matching)
            )

            # Exclude previously interacted items.
            previous_items = seen.get(user, set())

            if previous_items:
                previous_indices = torch.tensor(
                    list(previous_items),
                    dtype=torch.long,
                    device=device,
                )
                scores[previous_indices] = -torch.inf

            valid_count = num_items - len(previous_items)
            top_k = min(k, valid_count)

            if top_k <= 0:
                continue

            top_items = torch.topk(
                scores, top_k
            ).indices.cpu().numpy().tolist()

            recommended_items.update(top_items)

            hits = [
                1 if item in actual_items else 0
                for item in top_items
            ]

            hit_count = sum(hits)

            recalls.append(
                hit_count / len(actual_items)
            )

            precisions.append(
                hit_count / k
            )

            dcg = sum(
                hit / np.log2(rank + 2)
                for rank, hit in enumerate(hits)
            )

            ideal_hits = min(len(actual_items), k)

            idcg = sum(
                1.0 / np.log2(rank + 2)
                for rank in range(ideal_hits)
            )

            ndcgs.append(
                dcg / idcg if idcg > 0 else 0.0
            )

    return {
        "Recall@K": float(np.mean(recalls)) if recalls else 0.0,
        "Precision@K": float(np.mean(precisions)) if precisions else 0.0,
        "NDCG@K": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "CatalogCoverage@K": (
            len(recommended_items) / num_items
            if num_items else 0.0
        ),
    }
