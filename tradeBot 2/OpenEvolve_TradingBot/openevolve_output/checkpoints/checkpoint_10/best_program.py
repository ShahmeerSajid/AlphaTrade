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

    # More aggressive ROI structure
    minimal_roi = {
        "0": 0.03,    # Take 3% profit immediately
        "10": 0.02,   # 2% after 10 minutes
        "20": 0.01,   # 1% after 20 minutes
        "30": 0.005   # 0.5% after 30 minutes
    }

    # More conservative stoploss
    stoploss = -0.05

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
        # # Assign random 0 or 1 to "enter_long" for each row, with seeded RNG
        
        # Entry conditions with trend confirmation
        price_below_lower = (dataframe['close'] < dataframe['gauss_lower']).astype(int)
        trend_confirmation = (dataframe['gauss_mid'].shift(1) > dataframe['gauss_mid'].shift(2)).astype(int)
        stoch_crossover = (
            (dataframe['stoch_k'] <= dataframe['stoch_d']) & 
            (dataframe['stoch_k'].shift(1) > dataframe['stoch_d'].shift(1)) &
            (dataframe['stoch_k'] < 30)  # More conservative oversold condition
        ).astype(int)
        momentum_filter = (dataframe['rsi'] < 40).astype(int)  # Add RSI filter
        
        # Volume filter
        volume_filter = (dataframe['volume'] > 0) & (
            dataframe['volume'] > dataframe['volume'].rolling(window=30).mean()
        ).astype(int)

        # Combine conditions
        dataframe["enter_long"] = (
            price_below_lower & 
            stoch_crossover & 
            oversold & 
            volume_filter
        ).astype(int)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on Gaussian Channel, populates the exit signal
        """
        # Assign random 0 or 1 to "exit_long" for each row, with seeded RNG
        # dataframe["exit_long"] = np.random.randint(0, 2, size=len(dataframe))
        # Exit conditions
        price_above_upper = (dataframe['close'] > dataframe['gauss_upper']).astype(int)
        stoch_overbought = (dataframe['stoch_k'] > 80).astype(int)
        stoch_cross_down = (
            (dataframe['stoch_k'] >= dataframe['stoch_d']) & 
            (dataframe['stoch_k'].shift(1) < dataframe['stoch_d'].shift(1))
        ).astype(int)

        # More sophisticated exit conditions
        dataframe["exit_long"] = (
            (price_above_upper) |  # Original condition
            (stoch_overbought & stoch_cross_down) |  # Original condition
            (dataframe['close'] < dataframe['gauss_mid'] & 
             dataframe['stoch_k'] < dataframe['stoch_k'].shift(1)) |  # Add trailing exit
            (dataframe['rsi'] > 75)  # Add RSI overbought exit
        ).astype(int)
        return dataframe

# EVOLVE-BLOCK-END: end of file 



