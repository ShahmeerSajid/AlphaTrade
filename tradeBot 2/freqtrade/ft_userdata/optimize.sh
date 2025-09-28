TIMERANGE=20240101-20250701

freqtrade hyperopt \
    --config         user_data/config.json \
    --strategy       GaussianChannelStochRSIStrategy \
    --hyperopt-loss  MultiMetricHyperOptLoss \
    --spaces         buy sell \
    --timeframe      1m \
    --timerange      $TIMERANGE \
    --epochs         2500 \
    --random-state   42 \
    -vvv
