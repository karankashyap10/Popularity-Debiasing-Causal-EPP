"""Filter Ciao source data and remap Colab LLM scores for preprocessing."""

import argparse
import json
from pathlib import Path

import pandas as pd


def filter_data(data_dir, combined_path, ratings_path=None, all_data_path=None):
    data_dir = Path(data_dir)
    combined_path = Path(combined_path)
    ratings_path = Path(ratings_path or data_dir / "ratings.csv")
    all_data_path = Path(all_data_path or data_dir / "all_data.csv")
    user_map_path = data_dir / "user_mapping.csv"
    item_map_path = data_dir / "item_mapping.csv"
    for path in (combined_path, ratings_path, user_map_path, item_map_path):
        if not path.exists():
            raise FileNotFoundError(f"Required input is missing: {path}")

    scores = json.loads(combined_path.read_text(encoding="utf-8"))
    score_users = {int(user) for user in scores}
    score_items = {int(item) for values in scores.values() for item in values}
    users = pd.read_csv(user_map_path)
    items = pd.read_csv(item_map_path)
    if not {"raw_user", "user"}.issubset(users.columns):
        raise ValueError("user_mapping.csv must contain raw_user and user columns")
    if not {"raw_item", "item"}.issubset(items.columns):
        raise ValueError("item_mapping.csv must contain raw_item and item columns")

    kept_users = users[users.user.astype(int).isin(score_users)].copy()
    kept_items = items[items.item.astype(int).isin(score_items)].copy()
    # Match prepare_ciao's stable sorted-ID assignment after filtering.
    kept_users = kept_users.sort_values("raw_user", key=lambda s: s.astype(str)).reset_index(drop=True)
    kept_items = kept_items.sort_values("raw_item", key=lambda s: s.astype(str)).reset_index(drop=True)
    old_to_new_user = {int(old): new for new, old in enumerate(kept_users.user.astype(int))}
    old_to_new_item = {int(old): new for new, old in enumerate(kept_items.item.astype(int))}
    raw_users = set(kept_users.raw_user.astype(str))
    raw_items = set(kept_items.raw_item.astype(str))

    raw = pd.read_csv(ratings_path, header=None, names=["user", "item", "rating", "timestamp"], dtype=str)
    raw = raw[raw.user.astype(str).isin(raw_users) & raw.item.astype(str).isin(raw_items)]
    while not raw.empty:
        before = len(raw)
        uc = raw.user.value_counts()
        ic = raw.item.value_counts()
        raw = raw[raw.user.isin(uc[uc >= 5].index) & raw.item.isin(ic[ic >= 5].index)]
        if len(raw) == before:
            break
    if raw.empty:
        raise ValueError("No interactions remain after cache filtering and 5-core filtering")
    kept_users = kept_users[kept_users.raw_user.astype(str).isin(set(raw.user))].copy()
    kept_items = kept_items[kept_items.raw_item.astype(str).isin(set(raw.item))].copy()
    kept_users = kept_users.sort_values("raw_user", key=lambda s: s.astype(str)).reset_index(drop=True)
    kept_items = kept_items.sort_values("raw_item", key=lambda s: s.astype(str)).reset_index(drop=True)
    old_to_new_user = {int(old): new for new, old in enumerate(kept_users.user.astype(int))}
    old_to_new_item = {int(old): new for new, old in enumerate(kept_items.item.astype(int))}
    filtered_ratings = data_dir / "ratings_llm_filtered.csv"
    raw.to_csv(filtered_ratings, header=False, index=False)

    filtered_all_path = data_dir / "all_data_llm_filtered.csv"
    if all_data_path.exists():
        all_data = pd.read_csv(all_data_path, sep="\t")
        all_data = all_data[
            all_data.user.astype(int).isin(old_to_new_user)
            & all_data.item.astype(int).isin(old_to_new_item)
        ].copy()
        all_data["user"] = all_data.user.astype(int).map(old_to_new_user)
        all_data["item"] = all_data.item.astype(int).map(old_to_new_item)
        all_data.to_csv(filtered_all_path, sep="\t", index=False)

    remapped = {}
    for old_user, item_scores in scores.items():
        old_user_id = int(old_user)
        if old_user_id not in old_to_new_user:
            continue
        new_user = old_to_new_user[old_user_id]
        remapped[str(new_user)] = {
            str(old_to_new_item[int(item)]): float(score)
            for item, score in item_scores.items()
            if int(item) in old_to_new_item
        }
    cache_path = combined_path.with_name("combined.filtered.json")
    cache_path.write_text(json.dumps(remapped), encoding="utf-8")
    print(f"Filtered ratings: {len(raw)} rows -> {filtered_ratings}")
    if filtered_all_path.exists():
        print(f"Filtered all_data: {len(all_data)} rows -> {filtered_all_path}")
    print(f"Retained {len(remapped)} users and {len(old_to_new_item)} cached items -> {cache_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/Ciao")
    parser.add_argument("--combined", default="outputs/Ciao/.llm_cache/combined.json")
    parser.add_argument("--ratings", default=None, help="Raw four-column ratings file")
    parser.add_argument("--all-data", default=None, help="Optional processed all_data.csv")
    args = parser.parse_args()
    filter_data(args.data_dir, args.combined, args.ratings, args.all_data)


if __name__ == "__main__":
    main()
