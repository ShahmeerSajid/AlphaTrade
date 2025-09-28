


# # FINAL ---> ACTS LIKE A JUDGE TO EXAMINE THE TRADING STRATEGY ..... 

import sys
import os
import subprocess
import json
import tempfile
import shutil
from pathlib import Path
import traceback
import re
import time

"""
This evaluator serves as the fitness function for OpenEvolve to optimize trading strategies automatically.
It takes evolved trading strategy files, runs them through freqtrade backtesting using Docker, and extracts
performance metrics to create fitness scores. 
"""

def evaluate(strategy_path):
    """
    Main evaluation function that processes a single trading strategy and returns fitness metrics.
    This function orchestrates the complete evaluation pipeline from strategy loading to metric extraction.
    """
    try:
        strategy_class_name = extract_strategy_class_name(strategy_path)
        if not strategy_class_name:
            return create_failure_result("No valid IStrategy class found")
        
        env_result = setup_temp_environment(strategy_path, strategy_class_name)
        freqtrade_dir = env_result[0]
        strategy_file = env_result[1]
        
        try:
            backtest_results = run_backtest(freqtrade_dir, strategy_class_name)
            metrics = extract_performance_metrics(backtest_results)
            return metrics
            
        finally:
            cleanup_temp_environment(strategy_file)
            
    except Exception as e:
        return create_failure_result(f"Evaluation error: {str(e)}")

"""
This function dynamically extracts the strategy class name from any Python file by parsing the source code.
It uses regex pattern matching to find classes that inherit from IStrategy, ensuring compatibility with
any naming convention that OpenEvolve might generate during the evolution process.
"""
def extract_strategy_class_name(strategy_path):
    try:
        with open(strategy_path, 'r') as f:
            content = f.read()
        
        pattern = r'class\s+(\w+)\s*\([^)]*IStrategy[^)]*\):'
        match = re.search(pattern, content)
        
        if match:
            return match.group(1)
        return None
        
    except Exception:
        return None

"""
This function sets up the evaluation environment by copying the strategy to your existing freqtrade directory.
Instead of creating isolated environments, it leverages your working freqtrade setup to ensure compatibility
with Docker containers and existing data. The function modifies the existing config_tao.json to use the
evolved strategy name, ensuring we use all the same settings that your boss has configured and tested.
"""
def setup_temp_environment(strategy_path, strategy_class_name):
    
    freqtrade_dir = Path("/Users/shahmeer/Desktop/TradeBot_summer 2025/trading-bot/freqtrade/ft_userdata")
    strategies_dir = freqtrade_dir / "user_data" / "strategies"
    
    unique_name = f"eval_strategy_{int(time.time())}.py"
    strategy_dest = strategies_dir / unique_name
    shutil.copy2(strategy_path, strategy_dest)
    
    # Use existing config_tao.json and modify only the strategy name
    config = modify_existing_config(freqtrade_dir, strategy_class_name)
    # config_path = freqtrade_dir / "user_data" / "eval_config.json"
    config_path = freqtrade_dir / "user_data" / "config_tao.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return freqtrade_dir, strategy_dest



"""
This function reads the existing config_tao.json file and modifies only the strategy name to use the
evolved strategy. This ensures we use all the same exchange settings, trading parameters, and
configurations that your boss has set up and tested, while only changing the strategy being evaluated.

# not important fucntion 
"""
def modify_existing_config(freqtrade_dir, strategy_class_name):
    config_tao_path = freqtrade_dir / "user_data" / "config_tao.json"
    
    with open(config_tao_path, 'r') as f:
        config = json.load(f)
    
    # Modify settings for evaluation
    config["strategy"] = strategy_class_name
    config["timeframe"] = "1m"  
    
    return config

"""
This function executes the actual backtesting using docker-compose as specified in your setup.
The docker-compose.yml shows that freqtrade runs in a container with proper volume mounting
and configuration. This approach matches your existing setup exactly and ensures compatibility
with your data files and directory structure.
"""
def run_backtest(freqtrade_dir, strategy_class_name):
    # cmd = [
    #     "docker-compose", "run", "--rm", "freqtrade",
    #     "backtesting",
    #     "--config", "/freqtrade/user_data/eval_config.json",
    #     "--strategy", strategy_class_name,
    #     "--timeframe", "1h",
    #     "--timerange", "20240625-20241001", 
    #     "--export", "trades"
    # ]
    
    ## Format from backtest.sh in freqtrade / ft_userdata
    docker_working_dir = "/Users/shahmeer/Desktop/TradeBot_summer 2025/trading-bot/freqtrade/ft_userdata"
    
    cmd = [
        "docker-compose", "run", "--rm", "freqtrade",
        "backtesting",
        "--config", "/freqtrade/user_data/config_tao.json",
        "--strategy", strategy_class_name,
        "--timeframe", "1m",
        "--timerange", "20250101-20250401",
        "-d", "/freqtrade/user_data/data/kraken",
        "--export", "trades"
    ]
    
    result = subprocess.run(
        cmd, 
        cwd=docker_working_dir,  # This is the key fix!
        capture_output=True, 
        text=True, 
        timeout=300
    )
    
    if result.returncode != 0:
        raise Exception(f"Backtesting failed: {result.stderr}")
    
    return load_backtest_results(freqtrade_dir)


"""
This function locates and parses the backtest results from freqtrade's output files. It handles multiple
possible file locations and formats that freqtrade might generate, including zipped results. The function
looks for the most recent backtest results and extracts them from zip files if necessary.
"""
def load_backtest_results(freqtrade_dir):
    import zipfile
    
    try:
        backtest_results_dir = freqtrade_dir / "user_data" / "backtest_results"
        
        if backtest_results_dir.exists():
            zip_files = list(backtest_results_dir.glob("backtest-result-*.zip"))
            
            if zip_files:
                # Get the most recent zip file
                latest_zip = max(zip_files, key=lambda f: f.stat().st_mtime)
                
                # wew want the main config file in the zip files ...
                with zipfile.ZipFile(latest_zip, 'r') as zip_ref:
                    # Look for the main backtest result file in the zip
                    for file_name in zip_ref.namelist():
                        if file_name.startswith('backtest-result-') and file_name.endswith('.json') and ('_config.json' not in file_name):
                            with zip_ref.open(file_name) as result_file:
                                return json.load(result_file)
            
            # in terms of exception ...
            json_files = [f for f in backtest_results_dir.glob("*.json") if not f.name.endswith('.meta.json')]
            
            if json_files:
                latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
                with open(latest_file, 'r') as f:
                    return json.load(f)
        
        raise Exception("No backtest results found")
        
    except Exception as e:
        raise Exception(f"Failed to load results: {str(e)}")

"""
This is the core function that transforms raw backtesting results into normalized fitness scores for OpenEvolve.
It extracts key performance indicators including profitability, risk metrics, trade consistency, and efficiency
measures. The function normalizes all metrics to a 0-1 scale and creates a weighted composite score that
balances profit generation with risk management, encouraging the evolution of robust trading strategies.
"""
def extract_performance_metrics(backtest_data):
    try:
        # Try to get from strategy_comparison first (your preferred format)
        if 'strategy_comparison' in backtest_data and len(backtest_data['strategy_comparison']) > 0:
            strategy_stats = backtest_data['strategy_comparison'][0]
            print("Using strategy_comparison data")
        else:
            # Fallback to strategy section
            strategy_stats = list(backtest_data['strategy'].values())[0]
            print("Using strategy data")
        
        # Helper function to safely extract numeric values
        def safe_numeric(value, default=0):
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, list) and len(value) > 0:
                return float(value[0]) if isinstance(value[0], (int, float)) else default
            elif isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            return default
        
        # Extract values using the exact field names from your strategy_comparison
        trades = int(safe_numeric(strategy_stats.get('trades', 0)))
        if trades <= 0:
            print("================== >>>>>>>>>> ❌❌❌❌❌❌❌❌❌❌  Number of trades were LESS THAN OR EQUAL TO ZERO❌❌❌❌❌❌❌❌ <<<<<<<<<< ======================")
            return {
                "IsSuccessfulEval": 0.0000,  # ← This should make it fail
                "Strategy": "REJECTED_ZERO_TRADES",
                "profit_total_pct": -999.0,  # Worst possible
                "profitability_score": 0.0,
                "risk_score": 0.0,
                "consistency_score": 0.0,
                "efficiency_score": 0.0,
                "overall_fitness": 0.0,  # Worst possible
                "profit_total_abs": -999.0, # worst possible (fake) 
                "trades": trades,
                "wins": 0,
                "winrate": 0.0,
                "max_drawdown_account": 1.0,
            }
        
        
        profit_total_abs = safe_numeric(strategy_stats.get('profit_total_abs', 0))
        profit_total_pct = safe_numeric(strategy_stats.get('profit_total_pct', 0))
        wins = safe_numeric(strategy_stats.get('wins', 0))
        winrate = safe_numeric(strategy_stats.get('winrate', 0))
        max_drawdown_account = safe_numeric(strategy_stats.get('max_drawdown_account', 1.0))
        sharpe = safe_numeric(strategy_stats.get('sharpe', 0))
        profit_factor = safe_numeric(strategy_stats.get('profit_factor', 0))
        profit_mean_pct = safe_numeric(strategy_stats.get('profit_mean_pct', 0))
        
        strategy_type = strategy_stats.get("key", "None")
        
        # Additional metrics from your data
        cagr = safe_numeric(strategy_stats.get('cagr', 0))
        sortino = safe_numeric(strategy_stats.get('sortino', 0))
        calmar = safe_numeric(strategy_stats.get('calmar', 0))
        expectancy = safe_numeric(strategy_stats.get('expectancy', 0))
        
        profitability_score = normalize_profit_score(profit_total_pct)
        risk_score = normalize_risk_score(max_drawdown_account)
        consistency_score = normalize_consistency_score(winrate, trades)
        efficiency_score = normalize_efficiency_score(sharpe)
        
        # overall_fitness = (
        #     profitability_score * 0.4 +
        #     risk_score * 0.25 +
        #     consistency_score * 0.2 +
        #     efficiency_score * 0.15
        # )
        
        overall_fitness = (
            profitability_score * 0.8 +      # heavier 80% weight to profit (was 40%)
            risk_score * 0.1 +               # 10% weight to risk (was 25%) 
            consistency_score * 0.05 +       # 5% weight to consistency (was 20%)
            efficiency_score * 0.05          # 5% weight to efficiency (was 15%)
        )
        
        #         # CRITICAL DEBUG: Print detailed fitness breakdown
        # print(f"=== FITNESS DEBUG for strategy below ===")
        # print(f"Profit: {profit_total_pct:.2f}% → Profitability Score: {profitability_score:.4f}")
        # print(f"Drawdown: {max_drawdown_account*100:.1f}% → Risk Score: {risk_score:.4f}")
        # print(f"Win Rate: {winrate*100:.1f}% → Consistency Score: {consistency_score:.4f}")
        # print(f"OVERALL FITNESS: {overall_fitness:.4f}")
        # print(f"IsSuccessfulEval: {True}")
        # print("=" * 20)
        
        result= {
            ""
            "IsSuccessfulEval": 1.0000, 
            "Strategy": strategy_type,
            "overall_fitness": overall_fitness,
            "profit_total_pct": profit_total_pct,
            "trades": trades,
            "profitability_score": profitability_score,
            "risk_score": risk_score,
            "consistency_score": consistency_score,
            "efficiency_score": efficiency_score,
            "profit_total_abs": profit_total_abs,
            
            "wins": wins,
            "winrate": winrate,
            "max_drawdown_account": max_drawdown_account,
            # "sharpe": sharpe,
            # "profit_factor": profit_factor,
            # "profit_mean_pct": profit_mean_pct,
            # "cagr": cagr,
            # "sortino": sortino,
            # "calmar": calmar,
            # "expectancy": expectancy,
            # "runs_successfully": 1.0,
            # "execution_time": safe_numeric(strategy_stats.get('backtest_run_end_ts', 0)) - safe_numeric(strategy_stats.get('backtest_run_start_ts', 0)),
            # # Legacy field names for backward compatibility
            # "total_return": profit_total_abs,
            # "profit_percent": profit_total_pct,
            # "total_trades": trades,
            # "win_rate": winrate,
            # "max_drawdown": max_drawdown_account,
            # "sharpe_approximation": sharpe
        }
        
                # DOUBLE CHECK: Print what we're returning
        print(f"RETURNING: overall_fitness = {result['overall_fitness']}")
        
        return result
        
    except Exception as e:
        return create_failure_result(f"Metrics extraction failed: {str(e)}")

def calculate_sharpe_approximation(profit_total_pct, max_drawdown_account):
    try:
        if max_drawdown_account <= 0:
            return 0
        return max(0, float(profit_total_pct) / (float(max_drawdown_account) * 100))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0

def calculate_profit_factor(stats):
    try:
        def safe_numeric(value, default=0):
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, list) and len(value) > 0:
                return float(value[0]) if isinstance(value[0], (int, float)) else default
            elif isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    return default
            return default
        
        profit_total_abs = safe_numeric(stats.get('profit_total_abs', 0))
        # Calculate loss total from profit data if not available
        loss_total = abs(min(0, profit_total_abs))
        if loss_total == 0:
            loss_total = 0.01  # Prevent division by zero
        return max(0, profit_total_abs) / loss_total
    except (TypeError, ValueError, ZeroDivisionError):
        return 0

# def normalize_profit_score(profit_total_pct):
#     if profit_total_pct <= 0:
#         return 0.0
#     return min(1.0, profit_total_pct / (profit_total_pct + 10))


def normalize_profit_score(profit_total_pct):
    if profit_total_pct >= 0:
        return min(1.0, (profit_total_pct + 100) / 200)
    else:
        return max(0.0, (100 + profit_total_pct) / 100)
    
    

def normalize_risk_score(max_drawdown_account):
    if max_drawdown_account >= 1.0:
        return 0.0
    return max(0.0, 1.0 - max_drawdown_account)

def normalize_consistency_score(winrate, trades):
    if trades == 0:
        return 0.0
    
    trade_penalty = min(1.0, trades / 10)
    return winrate * trade_penalty

def normalize_efficiency_score(sharpe):
    if sharpe <= 0:
        return 0.0
    return min(1.0, max(0.0, sharpe / 2.0))

"""
This function creates standardized failure responses when strategy evaluation encounters errors. It ensures
that OpenEvolve receives consistent feedback format for failed strategies, allowing the evolution process
to properly eliminate problematic strategies while maintaining the expected data structure for fitness scoring.
"""
def create_failure_result(error_message):
    return {
        "IsSuccessfulEval": 0.0000, 
        "profitability_score": 0.0,
        "risk_score": 0.0,
        "consistency_score": 0.0,
        "efficiency_score": 0.0,
        "overall_fitness": 0.0,
        "profit_total_abs": 0.0,
        "profit_total_pct": 0.0,
        "trades": 0,
        "winrate": 0.0,
        "max_drawdown_account": 1.0,
        # "sharpe": 0.0,
        # "profit_factor": 0.0,
        # "runs_successfully": 0.0,
        # "execution_time": 999.0,
        # "error": error_message,
        # # Legacy field names for backward compatibility
        # "total_return": 0.0,
        # "profit_percent": 0.0,
        # "total_trades": 0,
        # "win_rate": 0.0,
        # "max_drawdown": 1.0,
        # "sharpe_approximation": 0.0
    }

def cleanup_temp_environment(strategy_file):
    try:
        if strategy_file.exists():
            strategy_file.unlink()
    except Exception:
        pass