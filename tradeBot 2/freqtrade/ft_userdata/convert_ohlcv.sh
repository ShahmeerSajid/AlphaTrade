TOKENS=("XRP" "BNB" "ETH" "LINK" "LTC" "SOL" "BTC" "FARTCOIN" "DOGE" "TAO")

for TOKEN in "${TOKENS[@]}"; do
    freqtrade trades-to-ohlcv \
        --pair $TOKEN/USD \
        --exchange kraken \
        --timeframes 1m 15m 1h 1d \
        -vv
done
