
"""Base interface for temporal forecasting methods in CausalEPP."""

from abc import ABC, abstractmethod

import numpy as np


class BaseForecaster(ABC):
    def __init__(self, window_size=10):
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.window_size = int(window_size)

    @abstractmethod
    def moving_average(self, values):
        """Calculate the moving average over the specified window."""
        pass

    @abstractmethod
    def estimate_trend(self, values):
        """Estimate the temporal trend of a sequence."""
        pass

    @abstractmethod
    def forecast(self, values, horizon=1.0):
        """Forecast the next value using the selected forecasting method."""
        pass

    def forecast_items(self, item_popularity, horizon=1.0):
        """
        Forecast local popularity for all items.

        Args:
            item_popularity: Array of shape (time_steps, num_items).

        Returns:
            Array of shape (num_items,).
        """
        item_popularity = np.asarray(item_popularity, dtype=np.float32)

        if item_popularity.ndim != 2:
            raise ValueError(
                "item_popularity must have shape (time_steps, num_items)."
            )

        return np.array(
            [
                self.forecast(item_popularity[:, item], horizon)
                for item in range(item_popularity.shape[1])
            ],
            dtype=np.float32,
        )

    def forecast_users(self, user_epp, horizon=1.0):
        """
        Forecast evolving personal popularity for all users.

        Args:
            user_epp: Array of shape (time_steps, num_users).

        Returns:
            Array of shape (num_users,).
        """
        user_epp = np.asarray(user_epp, dtype=np.float32)

        if user_epp.ndim != 2:
            raise ValueError(
                "user_epp must have shape (time_steps, num_users)."
            )

        return np.array(
            [
                self.forecast(user_epp[:, user], horizon)
                for user in range(user_epp.shape[1])
            ],
            dtype=np.float32,
        )
