import os
import pandas as pd
from collections import defaultdict

DATA_DIR = "historical_data"
SUFFIX = "_USD-trades.feather"
OUTPUT_DIR = "user_data/data/kraken"

def is_valid_file(file: str) -> bool:
    return file != ".DS_Store"

def merge_data(data_list: list[pd.DataFrame], ticker: str) -> pd.DataFrame:
    df = pd.concat(data_list)
    df = df.drop_duplicates(subset=["id"])
    print(f"ticker: {ticker}, df.timestamp.is_monotonic_increasing={df.timestamp.is_monotonic_increasing}")
    return df

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timeranges = [timerange for timerange in os.listdir(DATA_DIR) if is_valid_file(timerange)]
    timeranges = sorted(timeranges, key=lambda x: int(x.split("-")[0]))
    ticker_data_map = defaultdict(list)
    for time in timeranges:
        if not is_valid_file(time):
            continue
        files = os.listdir(os.path.join(DATA_DIR, time))
        files = [file for file in files if file.endswith(SUFFIX)]
        for file in files:
            if not is_valid_file(file):
                continue
            ticker = file.split(SUFFIX)[0]
            ticker_data_map[ticker].append(pd.read_feather(os.path.join(DATA_DIR, time, file)))

    for ticker in ticker_data_map:
        df = merge_data(ticker_data_map[ticker], ticker)
        df.to_feather(os.path.join(OUTPUT_DIR, f"{ticker}{SUFFIX}"))


if __name__ == "__main__":
    main()
