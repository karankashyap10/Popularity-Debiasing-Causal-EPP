# Paper-faithful CausalEPP (Ciao + LightGCN)

This is a clean implementation organized similarly to the supplied CausalEPP repository, with the model equation intended to follow the paper rather than the released implementation.

## Dataset
Place the original Ciao ratings file at `data/Ciao/ratings.csv`, with **no header** and columns:

`user_id,item_id,rating,timestamp`

The preprocessing script applies iterative 5-core filtering, remaps IDs, creates a chronological 90/10 time split, splits the final period by user into validation/test, and computes temporal local popularity and EPP histories. Ratings are retained but are **not** used to initialize q or as the quality target.

## Run

```bash
pip install -r requirements.txt
python main.py --preprocess
python main.py --epochs 100
```

If preprocessing was already done, omit `--preprocess`.

## Paper equation mapping

- LightGCN graph propagation supplies matching score `m_ui = e_u^T e_i`.
- Eq. (6): `tanh(q_i + c_ui) * softplus(m_ui)`.
- Eq. (7): `c_ui = exp(-alpha * abs(s_u^t - p_i^t)) * p_i^t * MLP(i)`; the MLP consumes the propagated item embedding.
- Eqs. (8)-(9): training substitutes mean training local popularity for the expectation over popularity, and uses user EPP at the interaction time.
- Eq. (10): quality pairwise loss based on the sign of global interaction-count differences.
- Eq. (11)-(12): BPR plus weighted quality loss.
- Eqs. (14)-(17): moving-average forecast function is provided in `forecasting/moving_average.py` and intended for inference-time intervention.

## Important implementation notes / limitations

1. The current training/evaluation pipeline uses last observed popularity/EPP for evaluation. The forecast helper exists but is not yet wired into final inference; this is the next implementation step before calling the temporal-intervention stage complete.
2. Temporal local popularity is computed as item interaction share within each time bin. The paper defines a short sliding window `[t-w1,t]`; exact sliding-window construction and its original hyperparameter should be checked against the experiment settings before claiming exact replication.
3. The paper's `q_i` is initialized to zero here, then learned through BPR and quality loss. Its quality loss supervises relative ordering using global interaction counts, not explicit ratings.
4. For reproducibility, record dataset source/version and preprocessing settings. Evaluation results depend on the split protocol and negative sampling.
5. This code is a first implementation pass and has not been validated against the paper's reported numbers. Do not describe it as an exact numerical reproduction until the temporal-window protocol, intervention inference, and benchmark results have been verified.
