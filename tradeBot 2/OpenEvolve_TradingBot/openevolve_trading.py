

# -2% profit ... 
#!/usr/bin/env python3
# import os
# import sys
# import json
# import yaml
# import time
# import shutil
# import random
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple
# import anthropic
# from dotenv import load_dotenv

# load_dotenv()

# class TrueOpenEvolve:
#     """
#     True OpenEvolve system where AI creates completely new trading strategies,
#     not just parameter optimization.
#     """
    
#     def __init__(self, config_path: str = "true_evolve_config.yaml"):
#         self.config = self.load_config(config_path)
#         self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
#         self.generation = 0
#         self.population = []
#         self.best_strategies = []
#         self.best_fitness = 0.0
#         self.evolution_log = []
#         self.setup_directories()
        
#     def load_config(self, config_path: str) -> Dict:
#         with open(config_path, 'r') as f:
#             return yaml.safe_load(f)
    
#     def setup_directories(self):
#         self.output_dir = Path(self.config['output']['strategy_output_dir'])
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.best_strategy_dir = self.output_dir / "best_strategy"
#         self.best_strategy_dir.mkdir(exist_ok=True)
    
#     def create_completely_new_strategy(self, variant_id: int, previous_strategies: List[Dict] = None) -> str:
#         prompt = self.create_strategy_evolution_prompt(variant_id, previous_strategies)
        
#         try:
#             print(f"AI creating completely new strategy variant {variant_id}...")
#             response = self.client.messages.create(
#                 model=self.config['llm']['primary_model'],
#                 max_tokens=8192,
#                 temperature=0.7,  # Higher creativity
#                 messages=[{"role": "user", "content": prompt}]
#             )
            
#             strategy_code = response.content[0].text
            
#             if "```python" in strategy_code:
#                 strategy_code = strategy_code.split("```python")[1].split("```")[0].strip()
#             elif "```" in strategy_code:
#                 strategy_code = strategy_code.split("```")[1].split("```")[0].strip()
            
#             print(f"AI successfully created new strategy concept for variant {variant_id}")
#             return strategy_code
            
#         except Exception as e:
#             print(f"Error creating strategy: {e}")
#             return self.create_fallback_strategy(variant_id)
    
#     def create_strategy_evolution_prompt(self, variant_id: int, previous_strategies: List[Dict] = None) -> str:
#         performance_context = ""
#         evolution_guidance = ""
        
#         if previous_strategies:
#             # Analyze what worked and what didn't
#             successful_patterns = []
#             failed_patterns = []
            
#             for strategy in previous_strategies[-3:]:  # Last 3 strategies
#                 if strategy.get('fitness', 0) > 0.3:
#                     successful_patterns.append(f"Strategy with {strategy.get('profit', 0):.1f}% profit")
#                 else:
#                     failed_patterns.append(f"Strategy lost {abs(strategy.get('profit', 0)):.1f}%")
            
#             performance_context = f"""
# EVOLUTION CONTEXT:
# Previous generation performance:
# Successful patterns: {successful_patterns}
# Failed patterns: {failed_patterns}

# EVOLUTION DIRECTIVE: Create a completely NEW trading approach that improves on previous failures.
# """
#             evolution_guidance = """
# EVOLUTION FOCUS:
# - If previous strategies were too conservative, create more aggressive entry conditions
# - If previous strategies overtrade, create more selective entry logic
# - If previous strategies had poor exits, innovate new exit mechanisms
# - Combine successful concepts in novel ways
# """
#         else:
#             performance_context = f"Create innovative strategy variant {variant_id} with unique trading logic."
#             evolution_guidance = f"""
# INNOVATION GOALS FOR VARIANT {variant_id}:
# - Variant 0-1: Focus on momentum-based strategies
# - Variant 2-3: Focus on mean-reversion strategies  
# - Variant 4-5: Focus on volatility-based strategies
# - Variant 6-7: Focus on volume-based strategies
# """
        
#         prompt = f"""You are an expert algorithmic trading strategist creating completely NEW trading strategies through evolutionary innovation.

# {performance_context}

# {evolution_guidance}

# STRATEGY CREATION TASK:
# Create a completely new freqtrade trading strategy with innovative entry/exit logic.

# INNOVATION REQUIREMENTS:
# 1. **Unique Entry Logic**: Don't just use "RSI < 30". Create novel combinations like:
#    - RSI crossovers with moving averages
#    - Volume surge detection with price momentum
#    - Bollinger band squeezes with RSI divergence
#    - MACD histogram patterns with volume confirmation
#    - Multiple timeframe confirmations

# 2. **Creative Exit Logic**: Beyond "RSI > 70", innovate:
#    - Trailing stops based on ATR
#    - Dynamic profit targets based on volatility
#    - Time-based exits with profit thresholds
#    - Trend reversal detection
#    - Support/resistance level exits

# 3. **Novel Indicator Combinations**: Mix indicators creatively:
#    - RSI + Bollinger Bands + Volume
#    - MACD + EMA + ATR
#    - Stochastic + Williams %R + Price Action
#    - Custom indicator combinations

# 4. **Smart Risk Management**: 
#    - Adaptive stoploss based on market conditions
#    - Position sizing based on volatility
#    - Dynamic ROI tables

# TECHNICAL REQUIREMENTS:
# - Class name: EvolvedStrategy_V{variant_id}_Gen{self.generation}
# - Use freqtrade IStrategy framework
# - Include comprehensive strategy description explaining the innovation
# - Use available indicators: RSI, MACD, EMA, SMA, Bollinger Bands, ATR, Stochastic, Williams %R
# - Ensure volume > 0 checks
# - Make entry/exit logic significantly different from simple RSI strategies

# INNOVATION EXAMPLES:
# ```python
# # Example 1: Momentum Burst Strategy
# dataframe["enter_long"] = (
#     (dataframe['rsi'] > 50) &  # Momentum building
#     (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2) &  # Volume surge
#     (dataframe['close'] > dataframe['bb_upperband']) &  # Breakout
#     (dataframe['macd'] > dataframe['macdsignal'])  # MACD confirmation
# ).astype(int)

# # Example 2: Mean Reversion with Momentum Exit
# dataframe["enter_long"] = (
#     (dataframe['rsi'] < 25) &  # Oversold
#     (dataframe['close'] < dataframe['bb_lowerband']) &  # Below lower BB
#     (dataframe['volume'] > dataframe['volume'].rolling(10).mean())  # Volume confirmation
# ).astype(int)

# dataframe["exit_long"] = (
#     (dataframe['rsi'] > 65) |  # Overbought OR
#     (dataframe['close'] > dataframe['ema_20'] * 1.02)  # 2% above EMA
# ).astype(int)
# ```

# CREATE A COMPLETELY NEW AND INNOVATIVE TRADING STRATEGY. 
# Return ONLY the complete Python strategy code with detailed explanations in the docstring.
# """
        
#         return prompt
    
#     def create_fallback_strategy(self, variant_id: int) -> str:
#         return f'''"""
# Fallback Strategy V{variant_id} - Basic RSI Strategy

# This is a simple fallback RSI strategy when AI generation fails.
# Uses basic RSI oversold/overbought conditions with volume confirmation.
# """

# import numpy as np
# import pandas as pd
# from pandas import DataFrame
# from freqtrade.strategy import IStrategy, IntParameter, RealParameter
# import talib.abstract as ta

# class EvolvedStrategy_V{variant_id}_Gen{self.generation}(IStrategy):
#     INTERFACE_VERSION = 3
#     can_short: bool = False
    
#     minimal_roi = {{"0": 0.03}}
#     stoploss = -0.08
#     timeframe = "1m"
    
#     def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
#         return dataframe
    
#     def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe["enter_long"] = (
#             (dataframe['rsi'] < 30) &
#             (dataframe['volume'] > 0)
#         ).astype(int)
#         return dataframe
    
#     def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe["exit_long"] = (dataframe['rsi'] > 70).astype(int)
#         return dataframe
# '''
    
#     def initialize_population(self) -> List[Dict]:
#         population = []
        
#         for i in range(self.config['population']['population_size']):
#             print(f"Creating innovative strategy variant {i}...")
#             strategy_code = self.create_completely_new_strategy(i)
            
#             population.append({
#                 'id': i,
#                 'code': strategy_code,
#                 'fitness': 0.0,
#                 'profit': 0.0,
#                 'trades': 0,
#                 'generation': 0,
#                 'strategy_type': self.analyze_strategy_type(strategy_code)
#             })
        
#         return population
    
#     def analyze_strategy_type(self, strategy_code: str) -> str:
#         if 'volume' in strategy_code and 'surge' in strategy_code.lower():
#             return 'volume_based'
#         elif 'bb_' in strategy_code or 'bollinger' in strategy_code.lower():
#             return 'volatility_based'  
#         elif 'macd' in strategy_code:
#             return 'momentum_based'
#         elif 'rsi' in strategy_code and '< 30' in strategy_code:
#             return 'mean_reversion'
#         else:
#             return 'hybrid'
    
#     def evolve_strategy(self, parent_strategy: Dict, variant_id: int) -> str:
#         evolution_prompt = f"""Evolve this trading strategy to create a better version:

# PARENT STRATEGY PERFORMANCE:
# - Fitness: {parent_strategy.get('fitness', 0):.4f}
# - Profit: {parent_strategy.get('profit', 0):.2f}%
# - Trades: {parent_strategy.get('trades', 0)}
# - Strategy Type: {parent_strategy.get('strategy_type', 'unknown')}

# PARENT STRATEGY CODE:
# ```python
# {parent_strategy['code']}
# ```

# EVOLUTION TASK:
# Create an improved version by:
# 1. Analyzing what might be causing poor performance
# 2. Adding new innovative elements
# 3. Modifying entry/exit logic intelligently
# 4. Keeping successful elements, improving weak ones
# 5. Change class name to: EvolvedStrategy_V{variant_id}_Gen{self.generation}

# If parent was losing money: Make entry conditions more selective
# If parent had few trades: Relax entry conditions slightly  
# If parent had poor win rate: Improve exit timing
# If parent had high drawdown: Add better risk management

# Create a meaningfully DIFFERENT strategy, not just parameter tweaks.
# Return ONLY the complete Python code.
# """
        
#         try:
#             response = self.client.messages.create(
#                 model=self.config['llm']['primary_model'],
#                 max_tokens=8192,
#                 temperature=0.6,
#                 messages=[{"role": "user", "content": evolution_prompt}]
#             )
            
#             evolved_code = response.content[0].text
            
#             if "```python" in evolved_code:
#                 evolved_code = evolved_code.split("```python")[1].split("```")[0].strip()
#             elif "```" in evolved_code:
#                 evolved_code = evolved_code.split("```")[1].split("```")[0].strip()
            
#             return evolved_code
            
#         except Exception as e:
#             print(f"Evolution failed: {e}")
#             return parent_strategy['code']  # Return parent if evolution fails
    
#     def evaluate_strategy_variant(self, variant: Dict) -> Dict:
#         import tempfile
        
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
#             temp_file.write(variant['code'])
#             temp_path = temp_file.name
        
#         try:
#             sys.path.insert(0, '.')
#             from trading_strategy_evaluator import evaluate
#             results = evaluate(temp_path)
            
#             variant['fitness'] = results.get('overall_fitness', 0.0)
#             variant['profit'] = results.get('profit_total_pct', 0.0)
#             variant['trades'] = results.get('trades', 0)
#             variant['winrate'] = results.get('winrate', 0.0)
#             variant['max_drawdown'] = results.get('max_drawdown_account', 1.0)
#             variant['evaluation_success'] = results.get('runs_successfully', 0) > 0
            
#             return variant
            
#         except Exception as e:
#             print(f"Error evaluating variant {variant['id']}: {e}")
#             variant['fitness'] = 0.0
#             variant['profit'] = 0.0
#             variant['evaluation_success'] = False
#             return variant
        
#         finally:
#             try:
#                 os.unlink(temp_path)
#             except:
#                 pass
    
#     def save_generation_best(self, population: List[Dict]) -> Path:
#         successful_variants = [v for v in population if v.get('evaluation_success', False)]
#         if not successful_variants:
#             return None
        
#         best_variant = max(successful_variants, key=lambda x: x['fitness'])
        
#         filename = f"Generation_{self.generation}_Best.py"
#         filepath = self.output_dir / filename
        
#         with open(filepath, 'w') as f:
#             f.write(best_variant['code'])
        
#         print(f"Saved Generation {self.generation} best strategy: {filename}")
#         print(f"  Strategy Type: {best_variant.get('strategy_type', 'unknown')}")
#         print(f"  Fitness: {best_variant['fitness']:.4f}, Profit: {best_variant['profit']:.2f}%")
        
#         return filepath
    
#     def create_next_generation(self, selected_strategies: List[Dict]) -> List[Dict]:
#         new_population = []
#         population_size = self.config['population']['population_size']
        
#         # Keep 1 best strategy (elitism)
#         if selected_strategies:
#             elite = selected_strategies[0].copy()
#             elite['id'] = 0
#             elite['generation'] = self.generation + 1
#             new_population.append(elite)
        
#         # Generate offspring through evolution
#         while len(new_population) < population_size:
#             if len(selected_strategies) >= 1:
#                 parent = random.choice(selected_strategies)
#                 offspring_code = self.evolve_strategy(parent, len(new_population))
                
#                 offspring = {
#                     'id': len(new_population),
#                     'code': offspring_code,
#                     'generation': self.generation + 1,
#                     'fitness': 0.0,
#                     'profit': 0.0,
#                     'trades': 0,
#                     'strategy_type': self.analyze_strategy_type(offspring_code)
#                 }
#                 new_population.append(offspring)
#             else:
#                 # Create completely new strategy if no good parents
#                 strategy_code = self.create_completely_new_strategy(len(new_population))
#                 offspring = {
#                     'id': len(new_population),
#                     'code': strategy_code,
#                     'generation': self.generation + 1,
#                     'fitness': 0.0,
#                     'strategy_type': self.analyze_strategy_type(strategy_code)
#                 }
#                 new_population.append(offspring)
        
#         return new_population
    
#     def check_stopping_criteria(self, population: List[Dict]) -> bool:
#         criteria = self.config['stopping_criteria']
        
#         if self.generation >= criteria['max_generations']:
#             print(f"Reached maximum generations ({criteria['max_generations']})")
#             return True
        
#         valid_strategies = [s for s in population if s.get('evaluation_success', False)]
#         if not valid_strategies:
#             return False
        
#         best_strategy = max(valid_strategies, key=lambda x: x['fitness'])
        
#         if best_strategy['fitness'] >= criteria.get('target_fitness', 0.8):
#             print(f"Reached target fitness: {best_strategy['fitness']:.4f}")
#             return True
        
#         if best_strategy['profit'] >= criteria.get('target_profit', 10.0):
#             print(f"Reached target profit: {best_strategy['profit']:.2f}%")
#             return True
        
#         return False
    
#     def run_evolution(self):
#         print("Starting TRUE OpenEvolve - Complete Strategy Evolution...")
#         print("AI will create entirely new trading strategies, not just tune parameters")
        
#         # Initialize with completely new strategies
#         self.population = self.initialize_population()
        
#         while True:
#             self.generation += 1
#             print(f"\n{'='*60}")
#             print(f"GENERATION {self.generation} - Strategy Innovation")
#             print(f"{'='*60}")
            
#             # Evaluate all strategies
#             for i, variant in enumerate(self.population):
#                 print(f"Evaluating {variant.get('strategy_type', 'unknown')} strategy {variant['id']}...")
#                 self.population[i] = self.evaluate_strategy_variant(variant)
            
#             # Save best from this generation
#             self.save_generation_best(self.population)
            
#             # Report results
#             successful_variants = [v for v in self.population if v.get('evaluation_success', False)]
            
#             if successful_variants:
#                 best_variant = max(successful_variants, key=lambda x: x['fitness'])
                
#                 print(f"Generation {self.generation} Results:")
#                 print(f"  Successful variants: {len(successful_variants)}/{len(self.population)}")
#                 print(f"  Best strategy type: {best_variant.get('strategy_type', 'unknown')}")
#                 print(f"  Best fitness: {best_variant['fitness']:.4f}")
#                 print(f"  Best profit: {best_variant['profit']:.2f}%")
#                 print(f"  Best trades: {best_variant.get('trades', 0)}")
                
#                 if best_variant['fitness'] > self.best_fitness:
#                     self.best_fitness = best_variant['fitness']
#                     self.best_strategies.append(best_variant.copy())
#                     print(f"NEW INNOVATION! Strategy type: {best_variant.get('strategy_type')}")
            
#             # Check stopping criteria
#             if self.check_stopping_criteria(self.population):
#                 break
            
#             # Create next generation through evolution
#             print("Evolving strategies to create new innovations...")
#             selected_strategies = [v for v in self.population if v.get('evaluation_success', False)]
#             selected_strategies.sort(key=lambda x: x['fitness'], reverse=True)
            
#             self.population = self.create_next_generation(selected_strategies[:3])  # Top 3 parents
            
#             time.sleep(2)
        
#         # Save best strategy
#         if self.best_strategies:
#             best_overall = max(self.best_strategies, key=lambda x: x['fitness'])
            
#             strategy_filepath = self.best_strategy_dir / "InnovativeStrategy.py"
#             with open(strategy_filepath, 'w') as f:
#                 f.write(best_overall['code'])
            
#             print(f"\n{'='*60}")
#             print("TRUE OPENEVOLVE COMPLETED")
#             print(f"Best innovative strategy type: {best_overall.get('strategy_type', 'unknown')}")
#             print(f"Best fitness: {best_overall['fitness']:.4f}")
#             print(f"Best profit: {best_overall['profit']:.2f}%")
#             print(f"Saved as: {strategy_filepath}")
#             print(f"{'='*60}")

# def main():
#     script_dir = Path(__file__).parent
#     config_path = script_dir / "config.yaml"  # get all the params from the config.yaml file 
    
#     if not config_path.exists():
#         print("Make sure config.yaml is in the same directory as this script")
#         sys.exit(1)
    
#     evolver = TrueOpenEvolve(str(config_path))
#     evolver.run_evolution()

# if __name__ == "__main__":
#     main()







#!/usr/bin/env python3

# import os
# import sys
# import json
# import yaml
# import time
# import shutil
# import random
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple
# import anthropic
# from dotenv import load_dotenv

# load_dotenv()

# class TrueOpenEvolve:
#     """
#     True OpenEvolve system where AI creates completely new trading strategies,
#     not just parameter optimization.
#     """
    
#     def __init__(self, config_path: str = "true_evolve_config.yaml"):
#         """
#         This function starts up the evolution system by loading the configuration file
#         and setting up all the necessary components. It connects to the claude 3.5 sonnet API,
#         creates directories for saving strategies, and initializes tracking variables
#         for the best strategies found.
#         """
#         self.config = self.load_config(config_path)
#         self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
#         self.generation = 0
#         self.population = []
#         self.best_strategies = []
#         self.best_fitness = 0.0
#         self.evolution_log = []
#         self.setup_directories()
        
#     def load_config(self, config_path: str) -> Dict:
#         with open(config_path, 'r') as f:
#             return yaml.safe_load(f)
    
#     def setup_directories(self):
#         self.output_dir = Path(self.config['output']['strategy_output_dir'])
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.best_strategy_dir = self.output_dir / "best_strategy"
#         self.best_strategy_dir.mkdir(exist_ok=True)
    
    
#     def create_completely_new_strategy(self, variant_id: int, previous_strategies: List[Dict] = None) -> str:
#         """
#         This function asks the AI to create a brand new trading strategy from scratch.
#         It's not just tweaking existing parameters but actually inventing new ways
#         to decide when to buy and sell. The AI looks at what worked or failed before
#         and tries to come up with creative solutions.
#         """
#         prompt = self.create_strategy_evolution_prompt(variant_id, previous_strategies)
        
#         try:
#             print(f"AI creating completely new strategy variant {variant_id}...")
#             response = self.client.messages.create(
#                 model=self.config['llm']['primary_model'],
#                 max_tokens=8192,
#                 temperature=0.7,
#                 messages=[{"role": "user", "content": prompt}]
#             )
            
#             strategy_code = response.content[0].text
            
#             if "```python" in strategy_code:
#                 strategy_code = strategy_code.split("```python")[1].split("```")[0].strip()
#             elif "```" in strategy_code:
#                 strategy_code = strategy_code.split("```")[1].split("```")[0].strip()
            
#             print(f"AI successfully created new strategy concept for variant {variant_id}")
#             return strategy_code
            
#         except Exception as e:
#             print(f"Error creating strategy: {e}")
#             return self.create_fallback_strategy(variant_id)
    
    
    
#     #### ASK ABOUT THIS ..... (do we have to specify this to the AI??)
#     def create_strategy_evolution_prompt(self, variant_id: int, previous_strategies: List[Dict] = None) -> str:
#         """
#         This function creates detailed instructions for the AI about what kind of
#         trading strategy to create. It analyzes what worked well in previous strategies
#         and what failed, then gives the AI specific guidance on how to improve.
#         This helps the AI understand the context and create better strategies each time.
#         """
#         performance_context = ""
#         evolution_guidance = ""
        
#         if previous_strategies:
#             successful_patterns = []
#             failed_patterns = []
            
#             for strategy in previous_strategies[-3:]:
#                 if strategy.get('fitness', 0) > 0.3:
#                     successful_patterns.append(f"Strategy with {strategy.get('profit', 0):.1f}% profit")
#                 else:
#                     failed_patterns.append(f"Strategy lost {abs(strategy.get('profit', 0)):.1f}%")
            
#             performance_context = f"""
# EVOLUTION CONTEXT:
# Previous generation performance:
# Successful patterns: {successful_patterns}
# Failed patterns: {failed_patterns}

# EVOLUTION DIRECTIVE: Create a completely NEW trading approach that improves on previous failures.
# """
#             evolution_guidance = """
# EVOLUTION FOCUS:
# - If previous strategies were too conservative, create more aggressive entry conditions
# - If previous strategies overtrade, create more selective entry logic
# - If previous strategies had poor exits, innovate new exit mechanisms
# - Combine successful concepts in novel ways
# """
#         else:
#             performance_context = f"Create innovative strategy variant {variant_id} with unique trading logic."
#             evolution_guidance = f"""
# INNOVATION GOALS FOR VARIANT {variant_id}:
# - Variant 0-1: Focus on momentum-based strategies
# - Variant 2-3: Focus on mean-reversion strategies  
# - Variant 4-5: Focus on volatility-based strategies
# - Variant 6-7: Focus on volume-based strategies
# """
        
#         prompt = f"""You are an expert algorithmic trading strategist creating completely NEW trading strategies through evolutionary innovation.

# {performance_context}

# {evolution_guidance}

# STRATEGY CREATION TASK:
# Create a completely new freqtrade trading strategy with innovative entry/exit logic.

# INNOVATION REQUIREMENTS:
# 1. **Unique Entry Logic**: Don't just use "RSI < 30". Create novel combinations like:
#    - RSI crossovers with moving averages
#    - Volume surge detection with price momentum
#    - Bollinger band squeezes with RSI divergence
#    - MACD histogram patterns with volume confirmation
#    - Multiple timeframe confirmations

# 2. **Creative Exit Logic**: Beyond "RSI > 70", innovate:
#    - Trailing stops based on ATR
#    - Dynamic profit targets based on volatility
#    - Time-based exits with profit thresholds
#    - Trend reversal detection
#    - Support/resistance level exits

# 3. **Novel Indicator Combinations**: Mix indicators creatively:
#    - RSI + Bollinger Bands + Volume
#    - MACD + EMA + ATR
#    - Stochastic + Williams %R + Price Action
#    - Custom indicator combinations

# 4. **Smart Risk Management**: 
#    - Adaptive stoploss based on market conditions
#    - Position sizing based on volatility
#    - Dynamic ROI tables

# TECHNICAL REQUIREMENTS:
# - Class name: EvolvedStrategy_V{variant_id}_Gen{self.generation}
# - Use freqtrade IStrategy framework
# - Include comprehensive strategy description explaining the innovation
# - Use available indicators: RSI, MACD, EMA, SMA, Bollinger Bands, ATR, Stochastic, Williams %R
# - Ensure volume > 0 checks
# - Make entry/exit logic significantly different from simple RSI strategies

# INNOVATION EXAMPLES:
# ```python
# # Example 1: Momentum Burst Strategy
# dataframe["enter_long"] = (
#     (dataframe['rsi'] > 50) &  # Momentum building
#     (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 2) &  # Volume surge
#     (dataframe['close'] > dataframe['bb_upperband']) &  # Breakout
#     (dataframe['macd'] > dataframe['macdsignal'])  # MACD confirmation
# ).astype(int)

# # Example 2: Mean Reversion with Momentum Exit
# dataframe["enter_long"] = (
#     (dataframe['rsi'] < 25) &  # Oversold
#     (dataframe['close'] < dataframe['bb_lowerband']) &  # Below lower BB
#     (dataframe['volume'] > dataframe['volume'].rolling(10).mean())  # Volume confirmation
# ).astype(int)

# dataframe["exit_long"] = (
#     (dataframe['rsi'] > 65) |  # Overbought OR
#     (dataframe['close'] > dataframe['ema_20'] * 1.02)  # 2% above EMA
# ).astype(int)
# ```

# CREATE A COMPLETELY NEW AND INNOVATIVE TRADING STRATEGY. 
# Return ONLY the complete Python strategy code with detailed explanations in the docstring.
# """
        
#         return prompt
    
#     def create_fallback_strategy(self, variant_id: int) -> str:
#         """
#         This function creates a simple backup trading strategy when the AI fails
#         to generate a proper strategy. Ensures that we always have something to evolve .... 
#         """
#         return f'''"""
# Fallback Strategy V{variant_id} - Basic RSI Strategy

# This is a simple fallback RSI strategy when AI generation fails.
# Uses basic RSI oversold/overbought conditions with volume confirmation.
# """

# import numpy as np
# import pandas as pd
# from pandas import DataFrame
# from freqtrade.strategy import IStrategy, IntParameter, RealParameter
# import talib.abstract as ta

# class EvolvedStrategy_V{variant_id}_Gen{self.generation}(IStrategy):
#     INTERFACE_VERSION = 3
#     can_short: bool = False
    
#     minimal_roi = {{"0": 0.03}}
#     stoploss = -0.08
#     timeframe = "1m"
    
#     def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
#         return dataframe
    
#     def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe["enter_long"] = (
#             (dataframe['rsi'] < 30) &
#             (dataframe['volume'] > 0)
#         ).astype(int)
#         return dataframe
    
#     def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe["exit_long"] = (dataframe['rsi'] > 70).astype(int)
#         return dataframe
# '''
    
#     def initialize_population(self) -> List[Dict]:
#         """
#         This function creates the first generation of trading strategies to start
#         the evolution process. It asks the AI to generate several completely different
#         strategy variants, each with unique approaches to trading. 
#         """
#         population = []
        
#         for i in range(self.config['population']['population_size']):
#             print(f"Creating innovative strategy variant {i}...")
#             strategy_code = self.create_completely_new_strategy(i)
            
#             population.append({
#                 'id': i,
#                 'code': strategy_code,
#                 'fitness': 0.0,
#                 'profit': 0.0,
#                 'trades': 0,
#                 'generation': 0,
#                 'strategy_type': self.analyze_strategy_type(strategy_code)
#             })
        
#         return population
    
#     def analyze_strategy_type(self, strategy_code: str) -> str:
#         if 'volume' in strategy_code and 'surge' in strategy_code.lower():
#             return 'volume_based'
#         elif 'bb_' in strategy_code or 'bollinger' in strategy_code.lower():
#             return 'volatility_based'  
#         elif 'macd' in strategy_code:
#             return 'momentum_based'
#         elif 'rsi' in strategy_code and '< 30' in strategy_code:
#             return 'mean_reversion'
#         else:
#             return 'hybrid'
    
#     def evolve_strategy(self, parent_strategy: Dict, variant_id: int) -> str:
#         """
#         This function takes a successful trading strategy and asks the AI to improve
#         it by creating a better version. It analyzes what the parent strategy did
#         well or poorly, then gives specific instructions to the AI on how to enhance
#         it. 
#         """
#         evolution_prompt = f"""Evolve this trading strategy to create a SIGNIFICANTLY DIFFERENT version:

# PARENT STRATEGY PERFORMANCE:
# - Fitness: {parent_strategy.get('fitness', 0):.4f}
# - Profit: {parent_strategy.get('profit', 0):.2f}%
# - Trades: {parent_strategy.get('trades', 0)}
# - Strategy Type: {parent_strategy.get('strategy_type', 'unknown')}

# PARENT STRATEGY CODE:
# ```python
# {parent_strategy['code']}
# ```

# CRITICAL: Create a MEANINGFULLY DIFFERENT strategy variant {variant_id}:

# REQUIRED CHANGES (pick 2-3):
# 1. **Change indicator focus**: If parent uses RSI, switch to MACD or Bollinger Bands
# 2. **Reverse approach**: If parent is mean-reversion, make it momentum-based  
# 3. **Add new conditions**: Include volume surges, trend filters, or volatility checks
# 4. **Modify timeframes**: Change from sensitive (RSI 14) to smooth (RSI 25) or vice versa
# 5. **Change exit strategy**: From fixed targets to trailing stops or trend reversal detection

# EVOLUTION VARIANTS:
# - Variant 0-1: Focus on momentum strategies (RSI > 50, MACD positive)
# - Variant 2-3: Focus on mean reversion (RSI < 30, below Bollinger lower band)  
# - Variant 4-5: Focus on volatility breakouts (close > upper band, volume surge)

# MANDATORY: Change class name to: EvolvedStrategy_V{variant_id}_Gen{self.generation}

# CREATE A GENUINELY DIFFERENT STRATEGY - not just parameter tweaks!
# Return ONLY the complete Python code.
# """
        
#         try:
#             response = self.client.messages.create(
#                 model=self.config['llm']['primary_model'],
#                 max_tokens=8192,
#                 temperature=0.6,
#                 messages=[{"role": "user", "content": evolution_prompt}]
#             )
            
#             evolved_code = response.content[0].text
            
#             if "```python" in evolved_code:
#                 evolved_code = evolved_code.split("```python")[1].split("```")[0].strip()
#             elif "```" in evolved_code:
#                 evolved_code = evolved_code.split("```")[1].split("```")[0].strip()
            
#             return evolved_code
            
#         except Exception as e:
#             print(f"Evolution failed: {e}")
#             return parent_strategy['code']
    
#     def evaluate_strategy_variant(self, variant: Dict) -> Dict:
#         """
#         This function tests how well a trading strategy performs by running it
#         through historical market data. It creates a temporary file with the strategy
#         code, then uses a separate evaluation system to see how much profit or loss
#         the strategy would have made. The function measures things like total profit,
#         number of trades, win rate, and maximum drawdown.
#         """
#         import tempfile
#         import signal
        
#         def timeout_handler(signum, frame):
#             raise TimeoutError("Evaluation timeout")
        
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
#             temp_file.write(variant['code'])
#             temp_path = temp_file.name
        
#         try:
#             with open(temp_path, 'r') as f:
#                 code = f.read()
#             compile(code, temp_path, 'exec')
            
#             signal.signal(signal.SIGALRM, timeout_handler)
#             signal.alarm(120)
            
#             sys.path.insert(0, '.')
#             from trading_strategy_evaluator import evaluate
#             results = evaluate(temp_path)
            
#             signal.alarm(0)
            
#             variant['fitness'] = results.get('overall_fitness', 0.0)
#             variant['profit'] = results.get('profit_total_pct', 0.0)
#             variant['trades'] = results.get('trades', 0)
#             variant['winrate'] = results.get('winrate', 0.0)
#             variant['max_drawdown'] = results.get('max_drawdown_account', 1.0)
#             variant['evaluation_success'] = results.get('runs_successfully', 0) > 0
            
#             return variant
            
#         except TimeoutError:
#             print(f"  TIMEOUT: Variant {variant['id']} evaluation took too long")
#             variant['fitness'] = 0.0
#             variant['profit'] = 0.0
#             variant['evaluation_success'] = False
#             variant['error'] = "Evaluation timeout"
#             return variant
#         except SyntaxError as e:
#             print(f"  SYNTAX ERROR in variant {variant['id']}: {e}")
#             variant['fitness'] = 0.0
#             variant['profit'] = 0.0
#             variant['evaluation_success'] = False
#             variant['error'] = f"Syntax error: {e}"
#             return variant
#         except Exception as e:
#             print(f"  EVALUATION ERROR in variant {variant['id']}: {e}")
#             variant['fitness'] = 0.0
#             variant['profit'] = 0.0
#             variant['evaluation_success'] = False
#             variant['error'] = f"Evaluation error: {e}"
#             return variant
        
#         finally:
#             signal.alarm(0)
#             try:
#                 os.unlink(temp_path)
#             except:
#                 pass
    
#     def save_generation_best(self, population: List[Dict]) -> Path:
#         """
#         This function finds the best performing strategy from the current generation
#         and saves it to a file for future reference. It looks through all the
#         strategies that were successfully evaluated, picks the one with the highest
#         fitness score, and writes its code to a file. This is like keeping a
#         record of your best cooking recipe from each attempt, so you can always
#         go back to it or use it as inspiration for future improvements. The saved
#         strategies help track progress over generations.
#         """
#         successful_variants = [v for v in population if v.get('evaluation_success', False)]
#         if not successful_variants:
#             return None
        
#         best_variant = max(successful_variants, key=lambda x: x['fitness'])
        
#         filename = f"Generation_{self.generation}_Best.py"
#         filepath = self.output_dir /  filename
        
#         with open(filepath, 'w') as f:
#             f.write(best_variant['code'])
        
#         print(f"Saved Generation {self.generation} best strategy: {filename}")
#         print(f"  Strategy Type: {best_variant.get('strategy_type', 'unknown')}")
#         print(f"  Fitness: {best_variant['fitness']:.4f}, Profit: {best_variant['profit']:.2f}%")
        
#         return filepath
    
#     def create_next_generation(self, selected_strategies: List[Dict]) -> List[Dict]:
#         """
#         This function creates a new generation of trading strategies by evolving
#         the best performers from the previous generation. It keeps the very best
#         strategy unchanged (elitism) and creates new strategies by asking the AI
#         to improve upon successful parent strategies. This is like breeding the
#         best plants in your garden to create an improved next generation - you
#         keep your prize winner and create offspring that hopefully inherit the
#         good traits while adding new improvements.
#         """
#         new_population = []
#         population_size = self.config['population']['population_size']
        
#         if selected_strategies:
#             elite = selected_strategies[0].copy()
#             elite['id'] = 0
#             elite['generation'] = self.generation + 1
#             new_population.append(elite)
        
#         while len(new_population) < population_size:
#             if len(selected_strategies) >= 1:
#                 parent = random.choice(selected_strategies)
#                 offspring_code = self.evolve_strategy(parent, len(new_population))
                
#                 offspring = {
#                     'id': len(new_population),
#                     'code': offspring_code,
#                     'generation': self.generation + 1,
#                     'fitness': 0.0,
#                     'profit': 0.0,
#                     'trades': 0,
#                     'strategy_type': self.analyze_strategy_type(offspring_code)
#                 }
#                 new_population.append(offspring)
#             else:
#                 strategy_code = self.create_completely_new_strategy(len(new_population))
#                 offspring = {
#                     'id': len(new_population),
#                     'code': strategy_code,
#                     'generation': self.generation + 1,
#                     'fitness': 0.0,
#                     'strategy_type': self.analyze_strategy_type(strategy_code)
#                 }
#                 new_population.append(offspring)
        
#         return new_population
    
#     def check_stopping_criteria(self, population: List[Dict]) -> bool:
#         """
#         This function decides whether the evolution process should stop by checking
#         if any of the success conditions have been met. It looks at things like
#         whether we've reached the maximum number of generations, if a strategy
#         has achieved the target fitness score, or if profit goals have been reached.
#         """
#         criteria = self.config['stopping_criteria']
        
#         if self.generation >= criteria['max_generations']:
#             print(f"Reached maximum generations ({criteria['max_generations']})")
#             return True
        
#         valid_strategies = [s for s in population if s.get('evaluation_success', False)]
#         if not valid_strategies:
#             return False
        
#         best_strategy = max(valid_strategies, key=lambda x: x['fitness'])
        
#         if best_strategy['fitness'] >= criteria.get('target_fitness', 0.8):
#             print(f"Reached target fitness: {best_strategy['fitness']:.4f}")
#             return True
        
#         if best_strategy['profit'] >= criteria.get('target_profit', 10.0):
#             print(f"Reached target profit: {best_strategy['profit']:.2f}%")
#             return True
        
#         return False
    
#     def run_evolution(self):
#         """
#         This is the main function that runs the entire evolution process from start
#         to finish. It creates the first generation of strategies, then repeatedly
#         evaluates them, saves the best ones, and creates improved versions for the
#         next generation. The process continues until stopping criteria are met or
#         a highly successful strategy is found.
#         """
#         print("Starting TRUE OpenEvolve - Complete Strategy Evolution...")
#         print("AI will create entirely new trading strategies, not just tune parameters")
        
#         self.population = self.initialize_population()
        
#         while True:
#             self.generation += 1
#             print(f"\n{'='*60}")
#             print(f"GENERATION {self.generation} - Strategy Innovation")
#             print(f"{'='*60}")
            
#             for i, variant in enumerate(self.population):
#                 print(f"Evaluating {variant.get('strategy_type', 'unknown')} strategy {variant['id']}...")
#                 self.population[i] = self.evaluate_strategy_variant(variant)
                
#                 result = self.population[i]
#                 if result.get('evaluation_success', False):
#                     print(f"  SUCCESS: Fitness={result['fitness']:.4f}, Profit={result['profit']:.2f}%, Trades={result['trades']}")
#                 else:
#                     print(f"  FAILED: Strategy evaluation failed")
            
#             self.save_generation_best(self.population)
            
#             successful_variants = [v for v in self.population if v.get('evaluation_success', False)]
#             failed_variants = [v for v in self.population if not v.get('evaluation_success', False)]
            
#             print(f"\nGENERATION {self.generation} DETAILED RESULTS:")
#             print(f"Successful evaluations: {len(successful_variants)}/{len(self.population)}")
#             print(f"Failed evaluations: {len(failed_variants)}")
            
#             if successful_variants:
#                 best_variant = max(successful_variants, key=lambda x: x['fitness'])
#                 avg_fitness = sum(v['fitness'] for v in successful_variants) / len(successful_variants)
#                 avg_profit = sum(v['profit'] for v in successful_variants) / len(successful_variants)
                
#                 print(f"BEST STRATEGY:")
#                 print(f"  Strategy type: {best_variant.get('strategy_type', 'unknown')}")
#                 print(f"  Fitness: {best_variant['fitness']:.4f}")
#                 print(f"  Profit: {best_variant['profit']:.2f}%")
#                 print(f"  Trades: {best_variant.get('trades', 0)}")
#                 print(f"  Win Rate: {best_variant.get('winrate', 0)*100:.1f}%")
                
#                 print(f"AVERAGE PERFORMANCE:")
#                 print(f"  Average fitness: {avg_fitness:.4f}")
#                 print(f"  Average profit: {avg_profit:.2f}%")
                
#                 if best_variant['fitness'] > self.best_fitness:
#                     self.best_fitness = best_variant['fitness']
#                     self.best_strategies.append(best_variant.copy())
#                     print(f"NEW INNOVATION! Best strategy type: {best_variant.get('strategy_type')}")
#             else:
#                 print("ALL STRATEGIES FAILED - No successful evaluations in this generation")
#                 print("This could indicate:")
#                 print("  - AI generated invalid code")
#                 print("  - Strategies are too conservative (no trades)")
#                 print("  - Evaluation system errors")
                
#                 for i, variant in enumerate(failed_variants[:3]):
#                     print(f"Failed Strategy {i}: {variant.get('strategy_type', 'unknown')}")
            
#             if successful_variants:
#                 avg_profit = sum(v['profit'] for v in successful_variants) / len(successful_variants)
#                 if avg_profit < -30:
#                     print(f"POOR PERFORMANCE DETECTED: Average profit {avg_profit:.1f}%")
#                     print("Strategies are losing too much money - will create more conservative variants next generation")
            
#             if self.check_stopping_criteria(self.population):
#                 break
            
#             print("Evolving strategies to create new innovations...")
#             selected_strategies = [v for v in self.population if v.get('evaluation_success', False)]
#             selected_strategies.sort(key=lambda x: x['fitness'], reverse=True)
            
#             if not selected_strategies or all(s['profit'] < -20 for s in selected_strategies):
#                 print("RESETTING POPULATION - All strategies performing poorly")
#                 print("Creating completely new population with more conservative approach...")
#                 self.population = self.initialize_population()
#             else:
#                 self.population = self.create_next_generation(selected_strategies[:3])
            
#             time.sleep(2)
        
#         if self.best_strategies:
#             best_overall = max(self.best_strategies, key=lambda x: x['fitness'])
            
#             strategy_filepath = self.best_strategy_dir / "InnovativeStrategy.py"
#             with open(strategy_filepath, 'w') as f:
#                 f.write(best_overall['code'])
            
#             print(f"\n{'='*60}")
#             print("TRUE OPENEVOLVE COMPLETED")
#             print(f"Best innovative strategy type: {best_overall.get('strategy_type', 'unknown')}")
#             print(f"Best fitness: {best_overall['fitness']:.4f}")
#             print(f"Best profit: {best_overall['profit']:.2f}%")
#             print(f"Saved as: {strategy_filepath}")
#             print(f"{'='*60}")

# def main():
#     script_dir = Path(__file__).parent
#     config_path = script_dir / "config.yaml"  # get all the params from the config.yaml file 
    
#     if not config_path.exists():
#         print("Make sure config.yaml is in the same directory as this script")
#         sys.exit(1)
    
#     evolver = TrueOpenEvolve(str(config_path))
#     evolver.run_evolution()

# if __name__ == "__main__":
#     main()










#!/usr/bin/env python3

# import os
# import sys
# import json
# import yaml
# import time
# import shutil
# import random
# import hashlib
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple
# import anthropic
# from dotenv import load_dotenv

# load_dotenv()

# class TrueOpenEvolve:
#     """
#     Enhanced OpenEvolve system with strict diversity enforcement and strategy type tracking
#     """
    
#     def __init__(self, config_path: str = "config.yaml"):
#         self.config = self.load_config(config_path)
#         self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
#         self.generation = 0
#         self.population = []
#         self.best_strategies = []
#         self.best_fitness = 0.0
#         self.evolution_log = []
#         self.strategy_hashes = set()  # Track code similarity
#         self.strategy_types_used = set()  # Track strategy types
#         self.setup_directories()
        
#     def load_config(self, config_path: str) -> Dict:
#         with open(config_path, 'r') as f:
#             return yaml.safe_load(f)
    
#     def setup_directories(self):
#         self.output_dir = Path(self.config['output']['strategy_output_dir'])
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         self.best_strategy_dir = self.output_dir / "best_strategy"
#         self.best_strategy_dir.mkdir(exist_ok=True)
    
#     def get_strategy_hash(self, code: str) -> str:
#         """
#         Create a hash of the strategy logic to detect duplicates
#         """
#         # Extract only the logic parts, ignore class names and comments
#         logic_parts = []
#         lines = code.split('\n')
#         for line in lines:
#             line = line.strip()
#             if ('enter_long' in line or 'exit_long' in line or 
#                 'dataframe[' in line or 'ta.' in line):
#                 logic_parts.append(line)
        
#         logic_string = ''.join(logic_parts)
#         return hashlib.md5(logic_string.encode()).hexdigest()[:8]
    
#     def is_strategy_duplicate(self, code: str) -> bool:
#         """
#         Check if strategy is too similar to existing ones
#         """
#         strategy_hash = self.get_strategy_hash(code)
#         if strategy_hash in self.strategy_hashes:
#             return True
#         self.strategy_hashes.add(strategy_hash)
#         return False
    
#     def get_mandated_strategy_type(self, variant_id: int) -> str:
#         """
#         Force specific strategy types based on variant ID to ensure diversity
#         """
#         strategy_types = [
#             "momentum_macd",
#             "mean_reversion_rsi", 
#             "volatility_bollinger",
#             "volume_surge",
#             "trend_following_ema",
#             "oscillator_stochastic"
#         ]
#         return strategy_types[variant_id % len(strategy_types)]
    
#     def create_completely_new_strategy(self, variant_id: int, previous_strategies: List[Dict] = None) -> str:
#         """
#         Create strategy with mandatory diversity requirements
#         """
#         mandated_type = self.get_mandated_strategy_type(variant_id)
#         max_attempts = 3
        
#         for attempt in range(max_attempts):
#             prompt = self.create_strategy_evolution_prompt(variant_id, previous_strategies, mandated_type)
            
#             try:
#                 print(f"AI creating {mandated_type} strategy variant {variant_id} (attempt {attempt + 1})...")
#                 response = self.client.messages.create(
#                     model=self.config['llm']['primary_model'],
#                     max_tokens=8192,
#                     temperature=0.8,  # Higher creativity for diversity
#                     messages=[{"role": "user", "content": prompt}]
#                 )
                
#                 strategy_code = response.content[0].text
                
#                 if "```python" in strategy_code:
#                     strategy_code = strategy_code.split("```python")[1].split("```")[0].strip()
#                 elif "```" in strategy_code:
#                     strategy_code = strategy_code.split("```")[1].split("```")[0].strip()
                
#                 # Check for duplicates
#                 if not self.is_strategy_duplicate(strategy_code):
#                     print(f"AI successfully created unique {mandated_type} strategy for variant {variant_id}")
#                     return strategy_code
#                 else:
#                     print(f"  Duplicate detected, retrying...")
                    
#             except Exception as e:
#                 print(f"Error creating strategy: {e}")
        
#         # If all attempts failed, create fallback
#         return self.create_fallback_strategy(variant_id, mandated_type)
    
#     def create_strategy_evolution_prompt(self, variant_id: int, previous_strategies: List[Dict] = None, mandated_type: str = None) -> str:
#         """
#         Enhanced prompt with strict diversity requirements and mandated strategy types
#         """
#         performance_context = self.analyze_previous_performance(previous_strategies)
        
#         strategy_type_instructions = {
#             "momentum_macd": """
# MANDATORY: Create a MACD-based momentum strategy
# - Primary indicator: MACD crossovers and histogram
# - Entry: MACD line crosses above signal line + momentum confirmation
# - Use MACD histogram for trend strength
# - Add volume confirmation
# - NO Bollinger Bands, NO RSI as primary logic
# """,
#             "mean_reversion_rsi": """
# MANDATORY: Create an RSI mean reversion strategy  
# - Primary indicator: RSI oversold/overbought levels
# - Entry: RSI < 30 (oversold) with reversal confirmation
# - Exit: RSI > 70 (overbought)
# - Add support/resistance levels
# - NO MACD, NO Bollinger Bands as primary logic
# """,
#             "volatility_bollinger": """
# MANDATORY: Create a Bollinger Bands volatility strategy
# - Primary indicator: Bollinger Band squeezes and expansions
# - Entry: Price touches lower band + squeeze release
# - Use BB width for volatility measurement
# - NO RSI, NO MACD as primary logic
# """,
#             "volume_surge": """
# MANDATORY: Create a volume-based surge strategy
# - Primary indicator: Volume spikes and volume moving averages
# - Entry: Volume > 3x average + price momentum
# - Use volume oscillators and on-balance volume
# - NO Bollinger Bands, NO RSI as primary logic
# """,
#             "trend_following_ema": """
# MANDATORY: Create an EMA trend following strategy
# - Primary indicator: EMA crossovers (fast/slow)
# - Entry: Fast EMA crosses above slow EMA + trend confirmation
# - Use multiple timeframe EMA alignment
# - NO Bollinger Bands, NO MACD as primary logic
# """,
#             "oscillator_stochastic": """
# MANDATORY: Create a Stochastic oscillator strategy
# - Primary indicator: Stochastic %K and %D lines
# - Entry: Stochastic oversold + crossover
# - Use Williams %R for confirmation
# - NO RSI, NO Bollinger Bands as primary logic
# """
#         }
        
#         type_instruction = strategy_type_instructions.get(mandated_type, "")
        
#         prompt = f"""You are creating a COMPLETELY UNIQUE trading strategy. Previous strategies have been too similar.

# {performance_context}

# {type_instruction}

# CRITICAL DIVERSITY REQUIREMENTS:
# 1. You MUST create a {mandated_type} strategy - no exceptions
# 2. Strategy MUST be fundamentally different from previous attempts
# 3. Use ONLY the mandated indicators as primary logic
# 4. Create unique entry/exit combinations not seen before
# 5. Class name: EvolvedStrategy_V{variant_id}_Gen{self.generation}

# INNOVATION EXAMPLES FOR {mandated_type}:

# ```python
# # UNIQUE approach examples:
# # - Multi-timeframe confirmations
# # - Divergence detection
# # - Rate of change filters
# # - Custom threshold combinations
# # - Time-based filters
# # - Price pattern recognition
# ```

# TECHNICAL REQUIREMENTS:
# - Use freqtrade IStrategy framework
# - Include volume > 0 checks
# - Add comprehensive docstring explaining the unique approach
# - Make it SIGNIFICANTLY different from volatility breakout strategies

# BANNED PATTERNS (do not copy these):
# - Generic "close > bb_upperband" entries
# - Standard "volume_surge > 2.0" conditions  
# - Basic "rsi > 70" exits
# - Copy-paste Bollinger Band logic from previous generations

# CREATE A GENUINELY UNIQUE {mandated_type} STRATEGY.
# Return ONLY the complete Python strategy code.
# """
        
#         return prompt
    
#     def analyze_previous_performance(self, previous_strategies: List[Dict]) -> str:
#         """
#         Analyze what went wrong with previous strategies
#         """
#         if not previous_strategies:
#             return "No previous strategies to analyze."
        
#         recent_strategies = previous_strategies[-3:]
        
#         failures = []
#         patterns = []
        
#         for strategy in recent_strategies:
#             profit = strategy.get('profit', 0)
#             fitness = strategy.get('fitness', 0)
#             strategy_type = strategy.get('strategy_type', 'unknown')
            
#             if profit < -5:
#                 failures.append(f"{strategy_type} lost {abs(profit):.1f}%")
            
#             patterns.append(strategy_type)
        
#         pattern_analysis = f"""
# PREVIOUS STRATEGY ANALYSIS:
# Failed strategies: {failures}
# Strategy types attempted: {patterns}
# Problem: Too many similar volatility/breakout strategies

# MANDATORY CHANGES NEEDED:
# - Stop creating similar volatility breakout strategies
# - Focus on completely different indicator logic
# - Avoid copying previous entry/exit patterns
# """
        
#         return pattern_analysis
    
#     def create_fallback_strategy(self, variant_id: int, strategy_type: str = "momentum_macd") -> str:
#         """
#         Create diverse fallback strategies based on type
#         """
#         fallbacks = {
#             "momentum_macd": f'''
# import talib.abstract as ta
# from freqtrade.strategy import IStrategy
# from pandas import DataFrame

# class EvolvedStrategy_V{variant_id}_Gen{self.generation}(IStrategy):
#     INTERFACE_VERSION = 3
#     can_short: bool = False
#     minimal_roi = {{"0": 0.04}}
#     stoploss = -0.08
#     timeframe = "1h"
    
#     def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         macd = ta.MACD(dataframe)
#         dataframe['macd'] = macd['macd']
#         dataframe['macdsignal'] = macd['macdsignal']
#         dataframe['macdhist'] = macd['macdhist']
#         return dataframe
    
#     def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe["enter_long"] = (
#             (dataframe['macd'] > dataframe['macdsignal']) &
#             (dataframe['macdhist'] > 0) &
#             (dataframe['volume'] > 0)
#         ).astype(int)
#         return dataframe
    
#     def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe["exit_long"] = (
#             dataframe['macd'] < dataframe['macdsignal']
#         ).astype(int)
#         return dataframe
# ''',
            
#             "mean_reversion_rsi": f'''
# import talib.abstract as ta
# from freqtrade.strategy import IStrategy
# from pandas import DataFrame

# class EvolvedStrategy_V{variant_id}_Gen{self.generation}(IStrategy):
#     INTERFACE_VERSION = 3
#     can_short: bool = False
#     minimal_roi = {{"0": 0.03}}
#     stoploss = -0.08
#     timeframe = "1h"
    
#     def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
#         return dataframe
    
#     def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe["enter_long"] = (
#             (dataframe['rsi'] < 30) &
#             (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
#             (dataframe['volume'] > 0)
#         ).astype(int)
#         return dataframe
    
#     def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
#         dataframe["exit_long"] = (dataframe['rsi'] > 70).astype(int)
#         return dataframe
# '''
#         }
        
#         return fallbacks.get(strategy_type, fallbacks["momentum_macd"])
    
#     def evolve_strategy(self, parent_strategy: Dict, variant_id: int) -> str:
#         """
#         Enhanced evolution with diversity enforcement
#         """
#         mandated_type = self.get_mandated_strategy_type(variant_id)
        
#         evolution_prompt = f"""EVOLVE this strategy into a DIFFERENT {mandated_type} approach:

# PARENT PERFORMANCE:
# - Fitness: {parent_strategy.get('fitness', 0):.4f}
# - Profit: {parent_strategy.get('profit', 0):.2f}%
# - Type: {parent_strategy.get('strategy_type', 'unknown')}

# EVOLUTION DIRECTIVE:
# 1. You MUST change this to a {mandated_type} strategy
# 2. COMPLETELY change the indicator logic
# 3. Use different entry/exit conditions
# 4. Make it fundamentally different from the parent

# PARENT STRATEGY CODE:
# ```python
# {parent_strategy['code']}
# ```

# MANDATORY: Transform this into a {mandated_type} strategy.
# Change class name to: EvolvedStrategy_V{variant_id}_Gen{self.generation}

# Return ONLY the complete Python code for the new {mandated_type} strategy.
# """
        
#         try:
#             response = self.client.messages.create(
#                 model=self.config['llm']['primary_model'],
#                 max_tokens=8192,
#                 temperature=0.7,
#                 messages=[{"role": "user", "content": evolution_prompt}]
#             )
            
#             evolved_code = response.content[0].text
            
#             if "```python" in evolved_code:
#                 evolved_code = evolved_code.split("```python")[1].split("```")[0].strip()
#             elif "```" in evolved_code:
#                 evolved_code = evolved_code.split("```")[1].split("```")[0].strip()
            
#             # Check for duplicates
#             if not self.is_strategy_duplicate(evolved_code):
#                 return evolved_code
#             else:
#                 print(f"  Evolution created duplicate, using fallback")
#                 return self.create_fallback_strategy(variant_id, mandated_type)
            
#         except Exception as e:
#             print(f"Evolution failed: {e}")
#             return self.create_fallback_strategy(variant_id, mandated_type)
    
#     def initialize_population(self) -> List[Dict]:
#         """
#         Initialize with guaranteed diverse strategy types
#         """
#         population = []
        
#         for i in range(self.config['population']['population_size']):
#             mandated_type = self.get_mandated_strategy_type(i)
#             print(f"Creating {mandated_type} strategy variant {i}...")
#             strategy_code = self.create_completely_new_strategy(i)
            
#             population.append({
#                 'id': i,
#                 'code': strategy_code,
#                 'fitness': 0.0,
#                 'profit': 0.0,
#                 'trades': 0,
#                 'generation': 0,
#                 'strategy_type': self.analyze_strategy_type(strategy_code),
#                 'mandated_type': mandated_type
#             })
        
#         return population
    
#     def analyze_strategy_type(self, strategy_code: str) -> str:
#         """
#         Enhanced strategy type detection
#         """
#         code_lower = strategy_code.lower()
        
#         if 'macd' in code_lower and ('macd' in code_lower and 'signal' in code_lower):
#             return 'momentum_macd'
#         elif 'rsi' in code_lower and ('< 30' in code_lower or '> 70' in code_lower):
#             return 'mean_reversion_rsi'
#         elif 'bb_' in code_lower or 'bollinger' in code_lower:
#             return 'volatility_bollinger'
#         elif 'volume' in code_lower and ('surge' in code_lower or 'volume_ma' in code_lower):
#             return 'volume_surge'
#         elif 'ema' in code_lower and 'crossover' in code_lower:
#             return 'trend_following_ema'
#         elif 'stoch' in code_lower or 'williams' in code_lower:
#             return 'oscillator_stochastic'
#         else:
#             return 'hybrid'
    
#     # ... (rest of the methods remain the same as in original)
#     def evaluate_strategy_variant(self, variant: Dict) -> Dict:
#         import tempfile
#         import signal
        
#         def timeout_handler(signum, frame):
#             raise TimeoutError("Evaluation timeout")
        
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
#             temp_file.write(variant['code'])
#             temp_path = temp_file.name
        
#         try:
#             with open(temp_path, 'r') as f:
#                 code = f.read()
#             compile(code, temp_path, 'exec')
            
#             signal.signal(signal.SIGALRM, timeout_handler)
#             signal.alarm(120)
            
#             sys.path.insert(0, '.')
#             from trading_strategy_evaluator import evaluate
#             results = evaluate(temp_path)
            
#             signal.alarm(0)
            
#             variant['fitness'] = results.get('overall_fitness', 0.0)
#             variant['profit'] = results.get('profit_percent', 0.0)
#             variant['trades'] = results.get('total_trades', 0)
#             variant['winrate'] = results.get('win_rate', 0.0)
#             variant['max_drawdown'] = results.get('max_drawdown', 1.0)
#             variant['evaluation_success'] = results.get('runs_successfully', 0) > 0
            
#             return variant
            
#         except Exception as e:
#             print(f"  EVALUATION ERROR in variant {variant['id']}: {e}")
#             variant['fitness'] = 0.0
#             variant['profit'] = 0.0
#             variant['evaluation_success'] = False
#             variant['error'] = f"Evaluation error: {e}"
#             return variant
        
#         finally:
#             signal.alarm(0)
#             try:
#                 os.unlink(temp_path)
#             except:
#                 pass
    
#     def save_generation_best(self, population: List[Dict]) -> Path:
#         successful_variants = [v for v in population if v.get('evaluation_success', False)]
#         if not successful_variants:
#             return None
        
#         best_variant = max(successful_variants, key=lambda x: x['fitness'])
        
#         filename = f"Generation_{self.generation}_Best_{best_variant.get('strategy_type', 'unknown')}.py"
#         filepath = self.output_dir / filename
        
#         with open(filepath, 'w') as f:
#             f.write(best_variant['code'])
        
#         print(f"Saved Generation {self.generation} best strategy: {filename}")
#         print(f"  Strategy Type: {best_variant.get('strategy_type', 'unknown')}")
#         print(f"  Mandated Type: {best_variant.get('mandated_type', 'unknown')}")
#         print(f"  Fitness: {best_variant['fitness']:.4f}, Profit: {best_variant['profit']:.2f}%")
        
#         return filepath
    
#     def create_next_generation(self, selected_strategies: List[Dict]) -> List[Dict]:
#         new_population = []
#         population_size = self.config['population']['population_size']
        
#         # Keep best strategy if it's good
#         if selected_strategies and selected_strategies[0]['fitness'] > 0.3:
#             elite = selected_strategies[0].copy()
#             elite['id'] = 0
#             elite['generation'] = self.generation + 1
#             new_population.append(elite)
        
#         # Generate diverse offspring
#         while len(new_population) < population_size:
#             variant_id = len(new_population)
            
#             if len(selected_strategies) >= 1 and random.random() < 0.7:
#                 # Evolve from parent but force diversity
#                 parent = random.choice(selected_strategies)
#                 offspring_code = self.evolve_strategy(parent, variant_id)
#             else:
#                 # Create completely new strategy
#                 offspring_code = self.create_completely_new_strategy(variant_id)
            
#             mandated_type = self.get_mandated_strategy_type(variant_id)
            
#             offspring = {
#                 'id': variant_id,
#                 'code': offspring_code,
#                 'generation': self.generation + 1,
#                 'fitness': 0.0,
#                 'profit': 0.0,
#                 'trades': 0,
#                 'strategy_type': self.analyze_strategy_type(offspring_code),
#                 'mandated_type': mandated_type
#             }
#             new_population.append(offspring)
        
#         return new_population
    
#     def check_stopping_criteria(self, population: List[Dict]) -> bool:
#         criteria = self.config['stopping_criteria']
        
#         if self.generation >= criteria['max_generations']:
#             print(f"Reached maximum generations ({criteria['max_generations']})")
#             return True
        
#         valid_strategies = [s for s in population if s.get('evaluation_success', False)]
#         if not valid_strategies:
#             return False
        
#         best_strategy = max(valid_strategies, key=lambda x: x['fitness'])
        
#         if best_strategy['fitness'] >= criteria.get('target_fitness', 0.8):
#             print(f"Reached target fitness: {best_strategy['fitness']:.4f}")
#             return True
        
#         if best_strategy['profit'] >= criteria.get('target_profit', 10.0):
#             print(f"Reached target profit: {best_strategy['profit']:.2f}%")
#             return True
        
#         return False
    
#     def run_evolution(self):
#         print("Starting ENHANCED OpenEvolve - Guaranteed Strategy Diversity...")
#         print("Forcing different strategy types: MACD, RSI, Bollinger, Volume, EMA, Stochastic")
        
#         self.population = self.initialize_population()
        
#         while True:
#             self.generation += 1
#             print(f"\n{'='*60}")
#             print(f"GENERATION {self.generation} - Diverse Strategy Testing")
#             print(f"{'='*60}")
            
#             # Show strategy diversity
#             types_in_generation = [p.get('mandated_type', 'unknown') for p in self.population]
#             print(f"Strategy types in this generation: {set(types_in_generation)}")
            
#             # Evaluate all strategies
#             for i, variant in enumerate(self.population):
#                 mandated_type = variant.get('mandated_type', 'unknown')
#                 print(f"Evaluating {mandated_type} strategy {variant['id']}...")
#                 self.population[i] = self.evaluate_strategy_variant(variant)
                
#                 result = self.population[i]
#                 if result.get('evaluation_success', False):
#                     print(f"  SUCCESS: Fitness={result['fitness']:.4f}, Profit={result['profit']:.2f}%, Trades={result['trades']}")
#                 else:
#                     print(f"  FAILED: Strategy evaluation failed")
            
#             self.save_generation_best(self.population)
            
#             # Detailed performance analysis
#             successful_variants = [v for v in self.population if v.get('evaluation_success', False)]
            
#             if successful_variants:
#                 print(f"\nDIVERSITY ANALYSIS:")
#                 for strategy_type in set(v.get('mandated_type', 'unknown') for v in successful_variants):
#                     type_strategies = [v for v in successful_variants if v.get('mandated_type') == strategy_type]
#                     if type_strategies:
#                         best_of_type = max(type_strategies, key=lambda x: x['fitness'])
#                         avg_profit_of_type = sum(v['profit'] for v in type_strategies) / len(type_strategies)
#                         print(f"  {strategy_type}: Best={best_of_type['fitness']:.4f}, Avg Profit={avg_profit_of_type:.2f}%")
                
#                 best_variant = max(successful_variants, key=lambda x: x['fitness'])
                
#                 if best_variant['fitness'] > self.best_fitness:
#                     self.best_fitness = best_variant['fitness']
#                     self.best_strategies.append(best_variant.copy())
#                     print(f"NEW BEST STRATEGY! Type: {best_variant.get('mandated_type')}")
            
#             if self.check_stopping_criteria(self.population):
#                 break
            
#             # Create next generation with enforced diversity
#             selected_strategies = [v for v in self.population if v.get('evaluation_success', False)]
#             selected_strategies.sort(key=lambda x: x['fitness'], reverse=True)
            
#             print("Creating next generation with enforced diversity...")
#             self.population = self.create_next_generation(selected_strategies[:3])
            
#             time.sleep(2)
        
#         # Save final best strategy
#         if self.best_strategies:
#             best_overall = max(self.best_strategies, key=lambda x: x['fitness'])
            
#             strategy_filepath = self.best_strategy_dir / "InnovativeStrategy.py"
#             with open(strategy_filepath, 'w') as f:
#                 f.write(best_overall['code'])
            
#             print(f"\n{'='*60}")
#             print("ENHANCED OPENEVOLVE COMPLETED")
#             print(f"Best strategy type: {best_overall.get('mandated_type', 'unknown')}")
#             print(f"Best fitness: {best_overall['fitness']:.4f}")
#             print(f"Best profit: {best_overall['profit']:.2f}%")
#             print(f"Saved as: {strategy_filepath}")
#             print(f"{'='*60}")

# def main():
#     script_dir = Path(__file__).parent
#     config_path = script_dir / "config.yaml"
    
#     if not config_path.exists():
#         print("Make sure config.yaml is in the same directory as this script")
#         sys.exit(1)
    
#     evolver = TrueOpenEvolve(str(config_path))
#     evolver.run_evolution()

# if __name__ == "__main__":
#     main()