import pandas as pd
import random

random.seed(42)
period_minutes = 15
token = "BTC"
path = f"user_data/data/kraken/{token}_USD-{period_minutes}m.feather"
df = pd.read_feather(path)

target_profit = 1.01
bought_price = 0.0
starting_balance = 1000
cash_balance = starting_balance
coin_balance = 0
trades = []

for idx, candle in enumerate(df.itertuples()):
    if coin_balance == 0:
        two_hours_minutes = 120
        two_hours_periods = two_hours_minutes // period_minutes
        max_price_in_next_2_hours = df.iloc[idx:idx+two_hours_periods].close.max()
        should_buy = candle.close * target_profit < max_price_in_next_2_hours
        if should_buy:
            bought_price = candle.close
            coin_balance = cash_balance / bought_price
            cash_balance = 0
            trades.append({
                "open_price": bought_price,
                "open_date": candle.date,
            })
    else:
        closing_price = candle.close
        duration_minutes = (candle.date - trades[-1]["open_date"]).total_seconds() / 60
        should_sell = closing_price > bought_price * target_profit
        if should_sell:
            cash_balance += coin_balance * closing_price
            coin_balance = 0
            trades[-1]["close_price"] = closing_price
            trades[-1]["close_date"] = candle.date
            trades[-1]["profit"] = (closing_price - trades[-1]["open_price"]) / trades[-1]["open_price"]
            trades[-1]["duration_minutes"] = duration_minutes
        

if coin_balance > 0:
    cash_balance += coin_balance * df.iloc[-1].close
    coin_balance = 0
    trades[-1]["close_price"] = df.iloc[-1].close
    trades[-1]["close_date"] = df.iloc[-1].date
    trades[-1]["profit"] = (df.iloc[-1].close - trades[-1]["open_price"]) / trades[-1]["open_price"]
    trades[-1]["duration_minutes"] = (df.iloc[-1].date - trades[-1]["open_date"]).total_seconds() / 60

print(f"Token: {token}")
print(f"Period: {period_minutes}m")
print(f"Total profit: {cash_balance - starting_balance}")
print(f"Total trades: {len(trades)}")
print(f"Win rate: {sum(trade['profit'] > 0 for trade in trades) / len(trades)}")
print(f"Average profit: {sum(trade['profit'] for trade in trades) / len(trades)}")
print(f"Average duration (minutes): {sum(trade['duration_minutes'] for trade in trades) / len(trades)}")
print(f"Num trades per day: {len(trades) / (df.date.max() - df.date.min()).days}")
# breakpoint()
