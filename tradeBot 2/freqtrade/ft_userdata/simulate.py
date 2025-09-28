import random

def get_beta_params(skew):
    """
    Given skew in [0.0, 1.0], return (alpha, beta) for beta distribution.
    - skew=0.0: completely towards range_min (alpha=1, beta=10)
    - skew=1.0: completely towards range_max (alpha=10, beta=1)
    - skew=0.5: uniform (alpha=1, beta=1)
    """
    # Clamp skew to [0,1]
    skew = max(0.0, min(1.0, skew))
    # Choose a base value for "extreme" skew
    extreme = 10.0
    if skew == 0.5:
        alpha = beta = 1.0
    else:
        # Linear interpolation between (1,extreme) and (extreme,1)
        alpha = 1.0 + (extreme - 1.0) * skew
        beta = 1.0 + (extreme - 1.0) * (1.0 - skew)
    return alpha, beta

starting_balance = 5000
final_balance = starting_balance
range_min = 0.9
range_max = 1.065
num_trades = 365 * 5  # 5 trades per day for 1 year
amount_to_leave = 5000
profit_target = 20000
profit_taken = 0

# Set skew here: 0.0 (towards range_min), 1.0 (towards range_max), 0.5 (uniform)
skew = 0.7

alpha, beta_param = get_beta_params(skew)

for i in range(num_trades):
    r = random.betavariate(alpha, beta_param)
    trade_return = range_min + (range_max - range_min) * r
    final_balance *= trade_return
    print(f"Trade {i+1}: {trade_return:.2f} -> {final_balance:,.2f}")
    if final_balance > profit_target:
        curr_profit_to_take = final_balance - amount_to_leave
        profit_taken += curr_profit_to_take
        final_balance -= curr_profit_to_take

print(f"Starting balance: {starting_balance:,.2f}")
print(f"Final balance: {final_balance:,.2f}")
print(f"Profit taken: {profit_taken:,.2f}")
