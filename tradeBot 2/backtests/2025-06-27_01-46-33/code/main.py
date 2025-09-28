# region imports
from AlgorithmImports import *
import math
from datetime import datetime
# endregion


class GaussianStochRsiBot(QCAlgorithm):

    # ──────────────────────────
    #  PARAMETERS (mirror Pine)
    # ──────────────────────────
    LENGTH        = 20      # Gaussian window
    MULTIPLIER    = 2.0     # Channel σ multiplier
    RSI_LEN       = 20
    STOCH_LEN     = 25
    K_LEN         = 8
    D_LEN         = 13
    START         = datetime(2024, 6, 20)
    END           = datetime(2025, 6, 20)
    SYMBOL        = "BTCUSDT"   # change to BTCUSD, ETHUSD, etc. if you like
    RES           = Resolution.DAILY  # or MINUTE for intraday

    # ──────────────────────────
    #  INITIALISE
    # ──────────────────────────
    def initialize(self):
        self.set_start_date(self.START)
        self.set_end_date(self.END)
        self.set_cash(100_000)

        self.set_brokerage_model(BrokerageName.BINANCE, AccountType.CASH)
        self.symbol = self.add_crypto(self.SYMBOL, self.RES).Symbol

        # --- Indicators & rolling buffers ---
        self.rsi  = self.RSI(self.symbol, self.RSI_LEN, MovingAverageType.Wilders, self.RES)
        self.prices = RollingWindow[float](self.LENGTH)
        self.rsi_win = RollingWindow[float](self.STOCH_LEN)
        self.k_sma = SimpleMovingAverage(self.K_LEN)
        self.d_sma = SimpleMovingAverage(self.D_LEN)

        # Pre-compute normalised Gaussian weights
        half   = (self.LENGTH - 1) / 2.0
        denom  = (self.LENGTH / 6.0) ** 2        # σ = len/6 ⇒ smooth curve
        w_raw  = [math.exp(-((i - half) ** 2) / (2 * denom)) for i in range(self.LENGTH)]
        w_sum  = sum(w_raw)
        self.weights = [w / w_sum for w in w_raw]

    # ──────────────────────────
    #  ON EACH BAR
    # ──────────────────────────
    def on_data(self, slice: Slice):
        bar = slice.Bars.get(self.symbol)
        if bar is None:
            return

        close = bar.Close

        # Update buffers
        self.prices.Add(close)
        if self.rsi.IsReady:
            self.rsi_win.Add(self.rsi.Current.Value)

        # Wait until every buffer/indicator is ready
        if not (self.prices.IsReady and self.rsi_win.IsReady):
            return

        # ── Gaussian channel ────────────────────────────
        closes = list(self.prices)
        g_mid  = sum(c * w for c, w in zip(closes, self.weights))
        g_std  = math.sqrt(sum(((c - g_mid) ** 2) * w for c, w in zip(closes, self.weights)))
        g_up   = g_mid + g_std * self.MULTIPLIER
        g_low  = g_mid - g_std * self.MULTIPLIER

        # ── Stoch RSI (K & D) ───────────────────────────
        rsi_vals = list(self.rsi_win)
        rsi_low  = min(rsi_vals)
        rsi_high = max(rsi_vals)
        stoch    = 100 * (self.rsi.Current.Value - rsi_low) / max(rsi_high - rsi_low, 1e-10)

        self.k_sma.Update(self.Time, stoch)
        if not self.k_sma.IsReady:
            return
        self.d_sma.Update(self.Time, self.k_sma.Current.Value)
        if not self.d_sma.IsReady:
            return

        k = self.k_sma.Current.Value
        d = self.d_sma.Current.Value

        # ── Entry / exit rules (faithful to your Pine) ──
        long_cond  = close * 1.05 <= g_low and k <= d
        exit_cond  = close >= g_up

        if long_cond and not self.Portfolio.Invested:
            self.set_holdings(self.symbol, 1)
            self.debug(f"[{self.Time}]  LONG @ {close:0.2f}")
        elif exit_cond and self.Portfolio.Invested:
            self.liquidate(self.symbol)
            self.debug(f"[{self.Time}]  EXIT @ {close:0.2f}")
