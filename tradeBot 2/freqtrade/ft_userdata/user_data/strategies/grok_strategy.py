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

class GrokStrategy(IStrategy):
    """
    Scalping Strategy for 1% Gains
    Based on RSI Oversold + MACD Bullish Crossover + Volume Surge

    Strategy Logic:
    - Long entry: RSI < oversold_threshold (e.g., 30) AND MACD line crosses above signal line AND volume > volume_ma * volume_multiplier
    - Exit: Primarily via ROI at 1%, with optional indicator-based exit if desired
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy - aim for 1% per trade
    minimal_roi = {
        "0": 0.01,
    }

    # Optimal stoploss designed for the strategy - tight for scalping to limit losses
    # stoploss = -0.005  # -0.5% stoploss to protect against quick reversals
    stoploss = -1

    # # Trailing stoploss - enable for potential higher gains, but start after some profit
    # trailing_stop = True
    # trailing_stop_positive = 0.002  # Trail once 0.2% in profit
    # trailing_stop_positive_offset = 0.003  # Start trailing only after 0.3% profit
    # trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy
    timeframe = "1m"

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Hyperoptable parameters
    rsi_length = IntParameter(low=10, high=30, default=14, space="buy", optimize=True, load=True)
    rsi_oversold = IntParameter(low=20, high=40, default=30, space="buy", optimize=True, load=True)
    macd_fast = IntParameter(low=8, high=20, default=12, space="buy", optimize=True, load=True)
    macd_slow = IntParameter(low=20, high=40, default=26, space="buy", optimize=True, load=True)
    macd_signal = IntParameter(low=5, high=15, default=9, space="buy", optimize=True, load=True)
    volume_ma_length = IntParameter(low=10, high=50, default=20, space="buy", optimize=True, load=True)
    volume_multiplier = RealParameter(low=1.2, high=3.0, default=1.5, space="buy", optimize=True, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50  # Based on max indicator periods

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
            "macd": {"color": "blue"},
            "macdsignal": {"color": "orange"},
        },
        "subplots": {
            "RSI": {
                "rsi": {"color": "purple"},
            },
            "Volume": {
                "volume": {"type": "bar", "color": "green"},
                "volume_ma": {"color": "red"},
            },
        },
    }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds RSI, MACD, and Volume MA indicators to the dataframe
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_length.value)

        # MACD
        macd = ta.MACD(
            dataframe,
            fastperiod=self.macd_fast.value,
            slowperiod=self.macd_slow.value,
            signalperiod=self.macd_signal.value
        )
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # Volume MA for surge detection
        dataframe['volume_ma'] = dataframe['volume'].rolling(window=self.volume_ma_length.value).mean()

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the entry signal based on RSI oversold, MACD crossover, and volume surge
        """
        # Conditions
        rsi_oversold_cond = (dataframe['rsi'] < self.rsi_oversold.value)
        
        # MACD bullish crossover: macd crosses above macdsignal
        macd_cross_cond = qtpylib.crossed_above(dataframe['macd'], dataframe['macdsignal'])
        
        # Volume surge: current volume > volume_ma * multiplier
        volume_surge_cond = (dataframe['volume'] > dataframe['volume_ma'] * self.volume_multiplier.value)
        
        has_volume = (dataframe['volume'] > 0)

        # Entry signal when all conditions are met
        dataframe['enter_long'] = (
            rsi_oversold_cond & macd_cross_cond & volume_surge_cond & has_volume
        ).astype(int)

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Populates the exit signal - primarily rely on ROI, but add optional indicator exit
        """
        # Optional: Exit on MACD bearish crossover for protection
        macd_cross_down = qtpylib.crossed_below(dataframe['macd'], dataframe['macdsignal'])
        
        dataframe['exit_long'] = macd_cross_down.astype(int)
        
        # If you want to rely only on ROI/stoploss, set to 0
        # dataframe["exit_long"] = 0

        return dataframe
