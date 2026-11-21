# CausalEPP experiments (Ciao)

This is a clean implementation organized similarly to the supplied CausalEPP repository, with the model equation intended to follow the paper rather than the released implementation.

## Dataset
Place the original Ciao ratings file at `data/Ciao/ratings.csv`, with **no header** and columns:

`user_id,item_id,rating,timestamp`

The preprocessing script applies iterative 5-core filtering, remaps IDs, creates chronological 80/10/10 train/validation/test periods, and computes temporal local popularity and EPP histories. Ratings are retained but are **not** used to initialize q or as the quality target.

## Run

```bash
pip install -r requirements.txt
python main.py --preprocess
python main.py --epochs 100
```

If preprocessing was already done, omit `--preprocess`.

## Colab LLM cache filtering and training

`outputs/Ciao/.llm_cache/combined.json` is treated as a `user_id -> {item_id: score}` map. Filter source data and create cache IDs that match the next preprocessing run:

```bash
python -m data.filter_by_combined
python main.py --preprocess --raw-file data/Ciao/ratings_llm_filtered.csv --llm-baseline
```

The filter command writes `ratings_llm_filtered.csv`, `all_data_llm_filtered.csv` when the existing file is available, and `combined.filtered.json` beside the source combined cache. LLM training reads the saved Colab scores and does not run local model scoring when that filtered cache exists. Filtering does not overwrite the original source files.

## Run commands

Each training invocation creates the next folder named `<method>_run-###` under `outputs/Ciao`. Available combinations are:

| Process | Command |
| --- | --- |
| Preprocess only (then stop) | `python -m data.preprocess` |
| CausalEPP using existing processed data | `python main.py` |
| Preprocess and train CausalEPP | `python main.py --preprocess` |
| CausalEPP with Holt forecasting | `python main.py --forecast-method holt` |
| CausalEPP plus popularity calibration | `python main.py --calibrate` |
| CausalEPP with local LLM score | `python main.py --llm-baseline` |
| Standalone LightGCN recommender | `python main.py --lightgcn-only` |
| Preprocess and train LLM variant from filtered ratings | `python main.py --preprocess --raw-file data/Ciao/ratings_llm_filtered.csv --llm-baseline` |
| Preprocess and train calibrated Holt CausalEPP | `python main.py --preprocess --forecast-method holt --calibrate` |
| Train with a chosen epoch count | `python main.py --epochs 100` |

`--preprocess`, `--forecast-method`, `--calibrate`, and `--epochs` can be combined with either CausalEPP variant. `--lightgcn-only` runs the standalone recommender and cannot be combined with `--llm-baseline`. Use `--raw-file PATH` with `--preprocess` to select a different raw ratings file.

## Paper equation mapping

- By default, LightGCN graph propagation supplies matching score `m_ui = e_u^T e_i`; `--llm-baseline` substitutes a local Hugging Face LLM score for `m_ui`.
- Eq. (6): `tanh(q_i + c_ui) * softplus(m_ui)`.
- Eq. (7): `c_ui = exp(-alpha * abs(s_u^t - p_i^t)) * p_i^t * MLP(i)`; the MLP consumes the propagated item embedding.
- Eqs. (8)-(9): training substitutes mean training local popularity for the expectation over popularity, and uses user EPP at the interaction time.
- Eq. (10): quality pairwise loss based on the sign of global interaction-count differences.
- Eq. (11)-(12): BPR plus weighted quality loss.
- Eqs. (14)-(17): moving-average forecast function is provided in `forecasting/moving_average.py` and intended for inference-time intervention.

## Added comparison options

- Select the original trailing-moving-average forecaster (default) or exponential-trend smoothing: `python main.py --forecast-method holt --epochs 100`. Holt's level and trend smoothing values are controlled by `forecast_level_alpha` and `forecast_trend_beta` in `config.py`.
- Add post-training calibrated re-ranking: `python main.py --calibrate --epochs 100`. This keeps the trained CausalEPP model and reports the original ranking, CP_H, and CP_JS (lambda=1) for LowPop, MedPop, and HighPop groups. Lambda selection uses validation only; test labels are used only for final evaluation. JSD uses natural logarithms (maximum ln 2). Results are saved in `calibration_lambdas.json`, `calibration_lambda_diagnostics.json`, `calibration_recommendation_comparison.json`, and `popularity_calibrated_test_metrics.json` under the run folder.
- Every CausalEPP run also saves `recommendation_examples.json` with the top-20 ranked items, final CausalEPP scores, `m_ui` values, held-out test items, and hit flags for the first five test users.
- Replace the LightGCN matching score `m_ui` with saved Colab user-item ratings while retaining the CausalEPP formula and its other terms: run `python main.py --llm-baseline` after filtering and preprocessing. If no combined cache is available, the scorer falls back to local Hugging Face scoring. Set `llm_model_id`, `llm_device`, or `llm_candidate_pool` in `config.py` to configure that fallback.
- Train a standalone LightGCN recommender with `python main.py --lightgcn-only`. It uses BPR training, selects the best checkpoint using validation NDCG, and saves Recall@K, Precision@K, NDCG@K, and CatalogCoverage@K with its checkpoint in its own run folder.
- Add CP_H and CP_JS calibration to standalone LightGCN with `python main.py --lightgcn-only --calibrate`. CP_H is selected on validation; CP_JS fixes lambda at 1.0. The post-processing uses LightGCN dot-product relevance scores, training-only popularity profiles, and writes `popularity_calibrated_test_metrics.json` plus lambda diagnostics.

The supplied popularity-calibration study is about POI recommendation and evaluates LORE/USG context-aware systems plus calibrated re-ranking. This Ciao project does not contain the geographic or social context needed to reproduce LORE/USG, so the added comparison adapts the paper's transferable calibrated-popularity (CP) re-ranking to CausalEPP output lists. Results should be labeled as an adaptation, not a reproduction of that POI experiment.

The local LLM supplies the scalar matching score `m_ui`; the CausalEPP quality and conformity terms remain. LightGCN still supplies the propagated item vector used by `MLP(i)` in the conformity term. LLM-mode validation and test metrics are calculated over each user's scored candidate pool, not the full item catalog, so compare them separately from full-catalog LightGCN metrics. This is a hybrid CausalEPP variant, not a complete removal of LightGCN representations.

## Important implementation notes / limitations

1. The default validation/test path forecasts popularity and EPP before scoring. `moving_average` preserves the original option; `holt` uses exponentially smoothed level and trend.
2. Temporal local popularity is computed as item interaction share within each time bin. The CausalEPP paper defines a short sliding window `[t-w1,t]`; this remains a difference from that experiment.
3. The popularity-calibration study uses POI datasets and final top-10 lists. Here CP is adapted to Ciao's H/M/T item popularity groups and the configured top-K, so its metrics are not directly comparable to the POI paper's reported values.
4. The LLM variant runs locally after downloading the model weights (about 4.1 GB for the default checkpoint). Runtime and memory needs depend on the selected device; each user requires a local scoring generation over the candidate pool.
5. For reproducibility, record dataset source/version and preprocessing settings. Evaluation results depend on the split protocol, negative sampling, forecasting option, calibration settings, and LLM model/version.
