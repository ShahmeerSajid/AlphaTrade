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

# Seed the RNG for reproducibility
np.random.seed(42)

# EVOLVE-BLOCK-START : Start of file

class RandomStrategy(IStrategy):
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

    # Minimal ROI designed for the strategy
    minimal_roi = {
        "0": 0.05,    # 5% immediate profit
        "30": 0.025,  # 2.5% after 30 minutes
        "60": 0.015,  # 1.5% after 60 minutes
        "120": 0.01   # 1% after 120 minutes
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.03  # Tighter 3% stoploss to minimize losses

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
    gaussian_length = IntParameter(low=5, high=50, default=20, space="buy", optimize=True, load=True)
    channel_multiplier = RealParameter(low=1.0, high=5.0, default=2.0, space="buy", optimize=True, load=True)
    rsi_length = IntParameter(low=10, high=30, default=20, space="buy", optimize=True, load=True)
    stoch_length = IntParameter(low=15, high=35, default=25, space="buy", optimize=True, load=True)
    k_length = IntParameter(low=5, high=15, default=8, space="buy", optimize=True, load=True)
    d_length = IntParameter(low=10, high=20, default=13, space="buy", optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    # startup_candle_count: int = 200

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

    def simple_moving_average(self, source: pd.Series, length: int) -> pd.Series:
        """
        Calculate Simple Moving Average
        """
        return source.rolling(window=length).mean()

    def simple_moving_std(self, source: pd.Series, length: int) -> pd.Series:
        """
        Calculate Simple Moving Standard Deviation
        """
        return source.rolling(window=length).std()

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
        gauss_mid = self.simple_moving_average(dataframe['close'], self.gaussian_length.value)
        gauss_std = self.simple_moving_std(dataframe['close'], self.gaussian_length.value)
        
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
        # Price is below lower Gaussian band (oversold condition)
        gaussian_condition = (dataframe['close'] < dataframe['gauss_lower'])
        
        # Stochastic RSI shows oversold and potential reversal
        stoch_condition = (
            (dataframe['stoch_k'] < 20) &  # Oversold
            (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1)) &  # K line turning up
            (dataframe['stoch_k'] <= dataframe['stoch_d'])  # K crossing up through D
        )
        
        # Volume filter
        volume_condition = (
            (dataframe['volume'] > 0) &  # Basic volume check
            (dataframe['volume'] > dataframe['volume'].rolling(window=24).mean())  # Above average volume
        )

        dataframe["enter_long"] = (
            gaussian_condition &
            stoch_condition &
            volume_condition
        ).astype(int)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on Gaussian Channel and Stochastic RSI, populates the exit signal
        """
        # Exit when price moves above upper Gaussian band
        gaussian_exit = (dataframe['close'] > dataframe['gauss_upper'])
        
        # Exit on bearish Stochastic RSI divergence
        stoch_exit = (
            (dataframe['stoch_k'] > 80) &  # Overbought
            (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1)) &  # K line turning down
            (dataframe['stoch_k'] >= dataframe['stoch_d'])  # K crossing down through D
        )
        
        # Exit on trend weakness
        trend_exit = (
            (dataframe['close'] < dataframe['gauss_mid']) &  # Price below middle band
            (dataframe['stoch_k'] < dataframe['stoch_k'].shift(3))  # Declining momentum
        )

        dataframe["exit_long"] = (
            gaussian_exit |
            stoch_exit |
            trend_exit
        ).astype(int)
        
        return dataframe

# EVOLVE-BLOCK-END: end of file 



