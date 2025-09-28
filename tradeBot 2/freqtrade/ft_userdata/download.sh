# DATA_PREFIX="/freqtrade/user_data"
DATA_PREFIX="user_data"
DATA_PREFIX_LOCAL="user_data"
DATA_DIR="historical_data"
TIMERANGES=("20250601-20250701")
TOKENS=("XRP" "BNB" "ETH" "LINK" "LTC" "SOL" "BTC" "FARTCOIN" "DOGE" "TAO")
PAIRS=$(printf "%s/USD " "${TOKENS[@]}")

for TIMERANGE in "${TIMERANGES[@]}"; do
    # if [[ -f "$DATA_PREFIX_LOCAL/$DATA_DIR/$TIMERANGE/${token}_USD-trades.feather" ]]; then
    #     echo "Skipping, ${token} ($TIMERANGE), file already exists"
    #     continue
    # fi

    freqtrade \
        download-data \
        --exchange kraken \
        --pairs $PAIRS \
        --timeframes 1m \
        --erase \
        --timerange $TIMERANGE \
        -d "$DATA_PREFIX/$DATA_DIR/$TIMERANGE" 2>&1 | tee -a logs_10.txt
    
    echo $?
done
