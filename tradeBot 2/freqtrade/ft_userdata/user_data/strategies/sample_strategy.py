# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these imports ---
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,  # @informative decorator
    # Hyperopt Parameters
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
    # timeframe helpers
    timeframe_to_minutes,
    timeframe_to_next_date,
    timeframe_to_prev_date,
    # Strategy helper functions
    merge_informative_pair,
    stoploss_from_absolute,
    stoploss_from_open,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
from technical import qtpylib


class GaussianChannelStochRSIStrategy(IStrategy):
    """
    Gaussian Channel + Stochastic RSI Strategy
    Based on Pine Script strategy by salmanshahidzia
    
    Strategy Logic:
    - Long entry: Price closes below lower Gaussian line AND Stoch RSI K <= D
    - Exit: Price closes above upper Gaussian line
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # # Minimal ROI designed for the strategy
    # minimal_roi = {
    #     "0": 0.05,
    # }

    # Optimal stoploss designed for the strategy
    # stoploss = -1  # -1 basically means no stoploss cuz you've just lost your whole position at that point anyways
    stoploss = -1

    # Trailing stoploss
    trailing_stop = False

    # Optimal timeframe for the strategy
    timeframe = "1m"

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    gaussian_length = IntParameter(low=5, high=50, default=20, space="buy", optimize=False, load=True)
    channel_multiplier = RealParameter(low=1.0, high=5.0, default=2.0, space="buy", optimize=False, load=True)
    rsi_length = IntParameter(low=10, high=30, default=20, space="buy", optimize=False, load=True)
    stoch_length = IntParameter(low=15, high=35, default=25, space="buy", optimize=False, load=True)
    k_length = IntParameter(low=5, high=15, default=8, space="buy", optimize=False, load=True)
    d_length = IntParameter(low=10, high=20, default=13, space="buy", optimize=False, load=True)
    entry_threshold = RealParameter(low=0.0, high=1.0, default=0.03, space="buy", optimize=True, load=True)
    exit_threshold = RealParameter(low=0.0, high=1.0, default=0.0, space="sell", optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

    # Optional order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "gauss_mid": {"color": "yellow"},
            "gauss_upper": {"color": "green"},
            "gauss_lower": {"color": "red"},
        },
        "subplots": {
            "Stoch_RSI": {
                "stoch_k": {"color": "blue"},
                "stoch_d": {"color": "orange"},
            },
        },
    }

    def gaussian_weighted_ma(self, source: pd.Series, length: int) -> pd.Series:
        """
        Calculate Gaussian Weighted Moving Average
        """
        half = (length - 1) / 2.0
        denom = (length / 6.0) ** 2
        
        def gaussian_ma(x):
            weights = np.exp(-((np.arange(length) - half) ** 2) / (2 * denom))
            weights = weights / weights.sum()
            return np.sum(x * weights)
        
        return source.rolling(window=length).apply(gaussian_ma, raw=True)

    def gaussian_weighted_std(self, source: pd.Series, length: int) -> pd.Series:
        """
        Calculate Gaussian Weighted Standard Deviation
        """
        half = (length - 1) / 2.0
        denom = (length / 6.0) ** 2
        
        def gaussian_std(x):
            weights = np.exp(-((np.arange(length) - half) ** 2) / (2 * denom))
            weights = weights / weights.sum()
            gavg = np.sum(x * weights)
            diff = x - gavg
            return np.sqrt(np.sum(diff ** 2 * weights))
        
        return source.rolling(window=length).apply(gaussian_std, raw=True)

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds Gaussian Channel and Stochastic RSI indicators to the dataframe
        """
        # Gaussian Channel calculation
        gauss_mid = self.gaussian_weighted_ma(dataframe['close'], self.gaussian_length.value)
        gauss_std = self.gaussian_weighted_std(dataframe['close'], self.gaussian_length.value)
        
        dataframe['gauss_mid'] = gauss_mid
        dataframe['gauss_upper'] = gauss_mid + gauss_std * self.channel_multiplier.value
        dataframe['gauss_lower'] = gauss_mid - gauss_std * self.channel_multiplier.value

        # Stochastic RSI calculation
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_length.value)
        
        # Calculate RSI lowest and highest over stoch_length period
        dataframe['rsi_lowest'] = dataframe['rsi'].rolling(window=self.stoch_length.value).min()
        dataframe['rsi_highest'] = dataframe['rsi'].rolling(window=self.stoch_length.value).max()
        
        # Calculate Stochastic RSI
        dataframe['stoch'] = 100 * (dataframe['rsi'] - dataframe['rsi_lowest']) / (
            dataframe['rsi_highest'] - dataframe['rsi_lowest'] + 1e-10
        )
        
        # Calculate K and D lines
        dataframe['stoch_k'] = dataframe['stoch'].rolling(window=self.k_length.value).mean()
        dataframe['stoch_d'] = dataframe['stoch_k'].rolling(window=self.d_length.value).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on Gaussian Channel and Stochastic RSI, populates the entry signal
        """
        dataframe.loc[
            (
                # Long entry: Price closes below lower Gaussian line AND Stoch RSI K <= D
                # (dataframe['close'] <= dataframe['gauss_lower'])
                (dataframe['close'] * (1 + self.entry_threshold.value) <= dataframe['gauss_lower'])
                & (dataframe['stoch_k'] <= dataframe['stoch_d'])
                & (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            "enter_long",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on Gaussian Channel, populates the exit signal
        """
        dataframe.loc[
            (
                # Exit condition: Price closes above upper Gaussian line
                (dataframe['close'] >= dataframe['gauss_upper'] * (1 + self.exit_threshold.value))
                & (dataframe['volume'] > 0)  # Make sure Volume is not 0
            ),
            "exit_long",
        ] = 1

        return dataframe
