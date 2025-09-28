TIMERANGE=20240101-20250701
# TIMERANGE=20240625-20240715
# TIMERANGE=20240715-20250701

freqtrade backtesting \
    --config   user_data/config_tao.json \
    --strategy GaussianChannelStochRSIStrategy \
    --timeframe 1m \
    --timerange $TIMERANGE \
    -d /Users/salmanshahid/Code/Projects/trading-bot/freqtrade/ft_userdata/user_data/data/kraken

python generate_graph.py

# –––––––––––––––––––––––––––––– Helpful Plotting Commands ––––––––––––––––––––––––––––––

# # Equity curve, draw-down, etc.
# docker compose run --rm \
#     freqtrade plot-profit \
#     --strategy GaussianChannelStochRSIStrategy \
#     --timeframe 1d \
#     --timerange $TIMERANGE

# # Inspect a single trade on the price chart
# docker compose run --rm freqtrade plot-dataframe \
#     --strategy GaussianChannelStochRSIStrategy \
#     --pair BTC/USDT \
#     --timeframe 1d \
#     --timerange $TIMERANGE
