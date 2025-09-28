import os
import json
import zipfile
from typing import Tuple
import matplotlib.pyplot as plt
import subprocess
from datetime import datetime

BACKTESTS_DIR = "user_data/backtest_results"

def get_latest_backtest() -> Tuple[str, dict]:
    backtest_files = [f for f in os.listdir(BACKTESTS_DIR) if f.endswith(".zip")]
    backtest_files = sorted(backtest_files, key=lambda x: os.path.getmtime(os.path.join(BACKTESTS_DIR, x)))
    latest_backtest_file = backtest_files[-1]
    zip_file_path = os.path.join(BACKTESTS_DIR, latest_backtest_file)
    curr_dir_name = latest_backtest_file.replace(".zip", "")
    curr_dir_path = os.path.join(BACKTESTS_DIR, curr_dir_name)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(curr_dir_path)
    with open(os.path.join(curr_dir_path, f"{curr_dir_name}.json"), "r") as f:
        results = json.load(f)
    with open(os.path.join(curr_dir_path, f"{curr_dir_name}.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(curr_dir_path)
    # Open the JSON file in the default editor (using 'open' command on macOS)
    json_path = os.path.join(curr_dir_path, f"{curr_dir_name}.json")
    try:
        subprocess.run(['open', '-a', 'Cursor', json_path], check=True)
    except Exception as e:
        print(f"Could not open JSON file: {e}")
    return curr_dir_path, results

def show_graph(output_path):
    try:
        subprocess.run(['open', '-a', 'Preview', output_path], check=True)
    except Exception as e:
        print(f"Could not open image in Preview: {e}")

def generate_graph(backtest_dir, backtest_json):
    trades = backtest_json['strategy'][list(backtest_json['strategy'].keys())[0]]['trades']
    balances = [trade['stake_amount'] for trade in trades]
    dates = [datetime.strptime(trade['open_date'], '%Y-%m-%d %H:%M:%S%z') for trade in trades]
    plt.figure(figsize=(10, 6))
    plt.plot(dates, balances, marker='o')
    plt.title('Balances Over Time')
    plt.xlabel('Trade Number')
    plt.ylabel('Balance')
    plt.grid(True)
    output_path = os.path.join(backtest_dir, "balances_over_time.png")
    plt.savefig(output_path)
    plt.close()
    show_graph(output_path)

def main():
    backtest_dir, backtest_json = get_latest_backtest()
    generate_graph(backtest_dir, backtest_json)
    print(f"Graph saved to {backtest_dir}/balances_over_time.png")


if __name__ == "__main__":
    main()    
