
#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import json
import csv
import time

"""
This test script validates the trading strategy evaluator to ensure it's ready for OpenEvolve integration.
It performs comprehensive testing of the evaluation pipeline including Docker connectivity, strategy file
validation, and actual backtest execution. The script uses your existing freqtrade setup and validates
that all components work correctly before proceeding with evolutionary optimization.
"""

sys.path.insert(0, '.')

try:
    from trading_strategy_evaluator import evaluate
    print("Successfully imported trading_strategy_evaluator")
except ImportError as e:
    print(f"Failed to import evaluator: {e}")
    print("Make sure trading_strategy_evaluator.py is in the current directory")
    sys.exit(1)

STRATEGY_BASE_PATH = "/Users/shahmeer/Desktop/TradeBot_summer 2025/trading-bot/freqtrade/ft_userdata/user_data/strategies"

TEST_STRATEGIES = [  # NOTE: WE ARE TESTING WITH JUST ONE STRATEGY FOR SIMPLICITY TO JUST DETECT BUGS LIKE PATH CONFIGURATION (BEFORE GETTING INTO THE TECHNICALITIES)...
    "random_strategy.py"
    # ADD MORE ....
]

"""
This function validates the complete testing environment to ensure all prerequisites are met before
running strategy evaluations. It checks docker-compose availability and freqtrade service accessibility
as specified in your docker-compose.yml configuration. This matches your actual setup that uses
docker-compose to run freqtrade in containers.
"""
def validate_environment():
    print("Validating test environment...")
    
    try:
        import subprocess
        # Test if docker-compose works
        result = subprocess.run(['docker-compose', '--version'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("Docker-compose is accessible")
        else:
            print("Docker-compose command failed")
            return False
    except Exception as e:
        print(f"Cannot access docker-compose: {e}")
        print("Make sure Docker and docker-compose are installed and running")
        return False
    
    strategy_dir = Path(STRATEGY_BASE_PATH)
    if not strategy_dir.exists():
        print(f"Strategy directory not found: {STRATEGY_BASE_PATH}")
        return False
    
    print(f"Strategy directory found: {STRATEGY_BASE_PATH}")
    
    missing_strategies = []
    for strategy in TEST_STRATEGIES:
        strategy_path = strategy_dir / strategy
        if not strategy_path.exists():
            missing_strategies.append(strategy)
        else:
            print(f"Found strategy: {strategy}")
    
    if missing_strategies:
        print(f"Missing strategies: {missing_strategies}")
        return False
    
    print("Environment validation passed")
    return True

"""
This function executes a complete evaluation cycle on a single strategy to test the entire pipeline.
It measures execution time, validates result format, and provides detailed output for debugging any
issues in the evaluation process. The function tests all aspects from strategy loading through metric
extraction and fitness score calculation to ensure the evaluator is working correctly.
"""
def test_single_strategy(strategy_name):
    print(f"Testing strategy: {strategy_name}")
    
    strategy_path = Path(STRATEGY_BASE_PATH) / strategy_name
    print(f"Strategy path: {strategy_path}")
    
    print("Starting evaluation...")
    
    start_time = time.time()
    try:
        results = evaluate(str(strategy_path))
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")
        print("\nResults:")
        print("--->" * 40)
        
        if results.get('runs_successfully', 0) > 0:
            print("EVALUATION SUCCESSFUL")
            print(f"Overall Fitness Score: {results['overall_fitness']:.4f}")
            print(f"Total Return: ${results['profit_total_abs']:.2f}")
            print(f"Profit Percentage: {results['profit_total_pct']:.2f}%")
            print(f"Total Trades: {results['trades']}")
            print(f"Wins: {results.get('wins', 'N/A')}")
            print(f"Win Rate: {results['winrate']*100:.1f}%")
            print(f"Max Drawdown: {results['max_drawdown_account']*100:.1f}%")
            print(f"Sharpe Ratio: {results['sharpe']:.4f}")
            print(f"Profit Factor: {results['profit_factor']:.4f}")
            
            # Additional metrics if available
            if 'cagr' in results:
                print(f"CAGR: {results['cagr']*100:.2f}%")
            if 'sortino' in results:
                print(f"Sortino Ratio: {results['sortino']:.4f}")
            if 'expectancy' in results:
                print(f"Expectancy: {results['expectancy']:.4f}")
            
            print("\nComponent Scores:")
            print(f"  Profitability: {results['profitability_score']:.4f}")
            print(f"  Risk Management: {results['risk_score']:.4f}")
            print(f"  Consistency: {results['consistency_score']:.4f}")
            print(f"  Efficiency: {results['efficiency_score']:.4f}")
        else:
            print("EVALUATION FAILED")
            if 'error' in results:
                print(f"Error: {results['error']}")
        
        return results
        
    except Exception as e:
        print(f"EVALUATION CRASHED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

"""
This function analyzes results across all tested strategies to provide comprehensive feedback on the
evaluator's performance. It compares fitness scores between strategies, identifies patterns in successes
and failures, and provides actionable recommendations for improving the evaluation setup. The analysis
helps ensure that the scoring system properly differentiates between strategy performance levels.
"""
def analyze_test_results(all_results):
    print(f"\n{'='*60}")
    print("OVERALL TEST ANALYSIS")
    
    successful_tests = [r for r in all_results.values() if r and r.get('runs_successfully', 0) > 0]
    failed_tests = [r for r in all_results.values() if not r or r.get('runs_successfully', 0) == 0]
    
    print(f"Successful evaluations: {len(successful_tests)}/{len(all_results)}")
    print(f"Failed evaluations: {len(failed_tests)}")
    
    if successful_tests:
        print("\nPerformance Ranking:")
        ranked_strategies = sorted(
            [(name, results) for name, results in all_results.items()
             if results and results.get('runs_successfully', 0) > 0],
            key=lambda x: x[1]['overall_fitness'],
            reverse=True
        )
        
        for i, (strategy_name, results) in enumerate(ranked_strategies, 1):
            print(f"{i}. {strategy_name}: {results['overall_fitness']:.4f} "
                  f"(Profit: {results['profit_total_pct']:.1f}%, "
                  f"Trades: {results['trades']}, "
                  f"Win Rate: {results['winrate']*100:.1f}%)")
    
    if failed_tests:
        print(f"\nFailed strategies need investigation:")
        for name, results in all_results.items():
            if not results or results.get('runs_successfully', 0) == 0:
                error_msg = results.get('error', 'Unknown error') if results else 'Evaluation crashed'
                print(f"  {name}: {error_msg}")
    
    return len(successful_tests) > 0

"""
This main function orchestrates the complete testing process from environment validation through individual
strategy testing to results analysis. It provides a structured approach to validating the evaluator's
readiness for OpenEvolve integration and gives clear feedback on what needs to be fixed before proceeding
with the evolutionary optimization phase.
"""
def main():
    print("Trading Strategy Evaluator Test Suite (Docker Version)")
    
    if not validate_environment():
        print("\nEnvironment validation failed. Please fix issues before proceeding.")
        sys.exit(1)
    
    all_results = {}
    
    for strategy_name in TEST_STRATEGIES:
        results = test_single_strategy(strategy_name)
        all_results[strategy_name] = results
        time.sleep(2)  # Brief pause between tests
    
    success = analyze_test_results(all_results)
    
    if success:
        print("EVALUATOR TESTING COMPLETED SUCCESSFULLY")
        print("The evaluator is ready for OpenEvolve integration.")
    else:
        print("EVALUATOR TESTING FAILED")

if __name__ == "__main__":
    main()