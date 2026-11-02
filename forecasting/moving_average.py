
"""Moving-average forecasting for CausalEPP (Equations 2, 14 and 15)."""

import numpy as np


def moving_average(series, window=10):
    """
    Calculate the trailing moving average.

    Equation (2): Moving average over the most recent window.
    """
    x = np.asarray(series, dtype=np.float64).reshape(-1)

    if x.size == 0:
        return np.array([], dtype=np.float64)

    window = max(1, min(int(window), len(x)))

    kernel = np.ones(window, dtype=np.float64) / window

    return np.convolve(x, kernel, mode="valid")


def estimate_trend(series, window=10):
    """
    Estimate the latest trend using the final two moving-average values.

    Returns the change in moving average per time step.
    """
    ma = moving_average(series, window)

    if ma.size < 2:
        return 0.0

    return float(ma[-1] - ma[-2])


def moving_average_forecast(series, window=10, horizon=1.0):
    """
    Forecast the next value using the latest moving-average trend.

    p_i* = p_i^T + (d MA_T(p_i) / dt) * Delta_T^i

    s_u* = s_u^T + (d MA_T(s_u) / dt) * Delta_T^u
    """
    x = np.asarray(series, dtype=np.float64).reshape(-1)

    if x.size == 0:
        return 0.0

    if horizon < 0:
        raise ValueError("Forecast horizon cannot be negative.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Series contains NaN or infinite values.")

    trend = estimate_trend(x, window)

    forecast = float(x[-1]) + trend * float(horizon)

    # Popularity and EPP are proportions.
    return float(np.clip(forecast, 0.0, 1.0))


def forecast_all(
    popularity_history,
    epp_history,
    window=10,
    item_horizon=1.0,
    user_horizon=1.0,
):
    """
    Forecast item popularity and user EPP independently.

    Args:
        popularity_history: (time_steps, num_items)
        epp_history: (time_steps, num_users)

    Returns:
        item_popularity_star: (num_items,)
        user_epp_star: (num_users,)
    """
    popularity = np.asarray(popularity_history, dtype=np.float64)
    epp = np.asarray(epp_history, dtype=np.float64)

    if popularity.ndim != 2 or epp.ndim != 2:
        raise ValueError(
            "Both histories must be 2D arrays: (time_steps, entities)."
        )

    if popularity.shape[0] != epp.shape[0]:
        raise ValueError(
            "Popularity and EPP histories must have the same number of time steps."
        )

    if window < 1:
        raise ValueError("Moving-average window must be positive.")

    item_popularity_star = np.array(
        [
            moving_average_forecast(
                popularity[:, item], window, item_horizon
            )
            for item in range(popularity.shape[1])
        ],
        dtype=np.float32,
    )

    user_epp_star = np.array(
        [
            moving_average_forecast(
                epp[:, user], window, user_horizon
            )
            for user in range(epp.shape[1])
        ],
        dtype=np.float32,
    )

    return item_popularity_star, user_epp_star
