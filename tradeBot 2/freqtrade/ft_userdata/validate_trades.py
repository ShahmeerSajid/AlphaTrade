import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

# timestamp = "2025-07-03_19-59-37"
# timestamp = "2025-07-04_12-26-05"
timestamp = "2025-07-04_13-06-17"
path = f"/Users/salmanshahid/Code/Projects/trading-bot/freqtrade/ft_userdata/user_data/backtest_results/backtest-result-{timestamp}/backtest-result-{timestamp}.json"

with open(path, "r") as f:
    data = json.load(f)

profit_percentages = []
sell_dates = []
trades = data["strategy"]["GaussianChannelStochRSIStrategy"]["trades"]
for trade in trades:
    pair = trade["pair"]
    token = pair.split("/")[0]
    quote = pair.split("/")[1]
    buy = trade["orders"][0]
    sell = trade["orders"][1]
    buy_date = datetime.fromtimestamp(buy["order_filled_timestamp"] / 1000)
    sell_date = datetime.fromtimestamp(sell["order_filled_timestamp"] / 1000)
    profit_pct = sell["safe_price"] / buy["safe_price"] - 1
    profit_percentages.append(profit_pct)
    sell_dates.append(sell_date)
    print(f"Bought {buy['amount']} {token} at {buy['safe_price']} on {buy_date}")
    print(f"Sold {sell['amount']} {token} at {sell['safe_price']} on {sell_date}")
    print(f"Now, have {sell['cost']} {quote} total")
    print(f"Profit percentage: {profit_pct * 100}%")
    print("")
    # print(f"Profit: {sell['profit_abs']}")
    # print(f"Profit percentage: {sell['profit_ratio']}")
    # print(f"Trade duration: {trade['trade_duration']}")
    # print(f"Exit reason: {trade['exit_reason']}")
    # print(f"Initial stop loss: {trade['initial_stop_loss_abs']}")
    # print(f"Stop loss: {trade['stop_loss_abs']}")
    # print(f"Min rate: {trade['min_rate']}")
    # print(f"Max rate: {trade['max_rate']}")

print(f"Max profit percentage: {max(profit_percentages) * 100}%")
print(f"Min profit percentage: {min(profit_percentages) * 100}%")
print(f"Average profit percentage: {sum(profit_percentages) / len(profit_percentages) * 100}%")

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(sell_dates, [pct * 100 for pct in profit_percentages], 'bo-', alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.xlabel('Trade Close Date')
plt.ylabel('Profit Percentage (%)')
plt.title('Profit Percentages by Trade Close Date')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot in the same directory as the JSON file
output_dir = os.path.dirname(path)
output_path = os.path.join(output_dir, "profit_percentages.jpg")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(output_path)
