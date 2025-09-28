


#!/usr/bin/env python3
"""
Final OpenEvolve Integration (to be further worked upon .... )

Latest result: 9.67% profit on 12 trades 
"""

# import os
# import sys
# import yaml
# import json
# import datetime
# import json
# import csv
# import shutil
# import asyncio
# from pathlib import Path
# from typing import Dict, Any, Optional

# from dotenv import load_dotenv
# load_dotenv() 






# # Import the real OpenEvolve
# try:
#     from openevolve.openevolve import OpenEvolve
#     from openevolve.openevolve.config import Config
# except ImportError:
#     print("OpenEvolve not installed. Install with:")
#     print("git clone https://github.com/codelion/openevolve.git")
#     print("cd openevolve && pip install -e .")
    
#     print("-------> Make sure to keep openevolve folder in the same directory as OpenEvolve_TradingBot folder.")
#     sys.exit(1)


    




import logging        
import re             
import math           
import random         
import time           
import functools      
import itertools     
import hashlib        # HTTP requests (GET, POST, APIs) --->  useful for the frontend integration ... 
import httpx          
import pandas as pd   
import numpy as np    

import os
import sys
import yaml
import json
import datetime
import csv
import shutil
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Force Python to use local OpenEvolve instead of global installation
script_dir = Path(__file__).parent
local_openevolve_path = script_dir / "openevolve"

print("Path:", local_openevolve_path)

# Add local path to the beginning of sys.path (highest priority)
if local_openevolve_path.exists():
    sys.path.insert(0, str(local_openevolve_path))
    print(f"DEBUG: Using local OpenEvolve from {local_openevolve_path}")
else:
    print(f"WARNING: Local OpenEvolve not found at {local_openevolve_path}")
    print("Using global OpenEvolve installation")

# Import the real OpenEvolve
try:
    # Try the correct nested import structure
    from openevolve.controller import OpenEvolve
    from openevolve.config import Config
    print("✅ Successfully imported OpenEvolve and Config from local installation!")
    
except ImportError as e:
    print(f"❌ Local import failed: {e}")
    try:
        # Fallback: try different import patterns
        from openevolve.openevolve.controller import OpenEvolve
        from openevolve.openevolve.config import Config
        print("✅ Successfully imported with nested path!")
    except ImportError as e2:
        print(f"❌ Nested import also failed: {e2}")
        print("OpenEvolve not installed or local version has issues.")
        print("Install with:")
        print("git clone https://github.com/codelion/openevolve.git")
        print("cd openevolve && pip install -e .")
        print("-------> Make sure to keep openevolve folder in the same directory as OpenEvolve_TradingBot folder.")
        sys.exit(1)
        
        
        
class TradingOpenEvolveIntegration:
    """
    Final integration using only OpenEvolve config
    """
    
    def __init__(self):
        self.setup_paths()
    
    def setup_paths(self):
        """Setup paths to your existing files"""
        script_dir = Path(__file__).parent.parent
        self.existing_strategy_path = script_dir / "freqtrade" / "ft_userdata" / "user_data" / "strategies" / "random_strategy.py"
        self.evaluator_path = Path(__file__).parent / "trading_strategy_evaluator.py"
        self.openevolve_config_path = Path(__file__).parent / "openevolve_config.yaml"
        
        
        
        if not self.existing_strategy_path.exists():
            print(f"Strategy file not found: {self.existing_strategy_path}")
            sys.exit(1)
        if not self.evaluator_path.exists():
            print(f"Evaluator file not found: {self.evaluator_path}")
            sys.exit(1)
        if not self.openevolve_config_path.exists():
            print(f"OpenEvolve config not found: {self.openevolve_config_path}")
            print("Please create the openevolve_config.yaml file")
            sys.exit(1)
        
        print(f"Found strategy: {self.existing_strategy_path}")
        print(f"Found evaluator: {self.evaluator_path}")
        print(f"Found config: {self.openevolve_config_path}")
        
        # Output directory from OpenEvolve config location
        self.output_dir = self.openevolve_config_path.parent / "openevolve_output"
        self.output_dir.mkdir(exist_ok=True)
        
        config_copy_path = self.output_dir / "openevolve_config.yaml"
        shutil.copy2(self.openevolve_config_path, config_copy_path)
    
    
    def check_evolve_blocks(self):
        """Check for EVOLVE-BLOCK markers"""
        with open(self.existing_strategy_path, 'r') as f:
            content = f.read()
        
        has_blocks = 'EVOLVE-BLOCK-START' in content and 'EVOLVE-BLOCK-END' in content
        if not has_blocks:
            print("WARNING: No EVOLVE-BLOCK markers found in strategy file")
            print("Add markers around code you want to evolve")
            return False
        print("Found EVOLVE-BLOCK markers in strategy")
        return True
    
    async def run_evolution(self):
        """Run OpenEvolve evolution"""
        if not self.check_evolve_blocks():
            return None
        
        print("Starting OpenEvolve Evolution")
        
        try:
            # Load OpenEvolve config
            config = Config.from_yaml(self.openevolve_config_path)
            
            # Set API key at runtime (secure)
            api_key = os.environ.get('ANTHROPIC_API_KEY') or os.environ.get('OPENAI_API_KEY')
            if not api_key:
                print("ERROR: No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
                return None
            
            # Set API base based on key type
            if os.environ.get('ANTHROPIC_API_KEY'):
                api_base = "https://api.anthropic.com/v1"
            else:
                api_base = "https://api.openai.com/v1"
            
            config.llm.update_model_params({
                "api_key": api_key,
                "api_base": api_base
            })
            
            print(f"Using API: {api_base}")
            print(f"Model: {config.llm.primary_model}")
            print(f"Max iterations: {config.max_iterations}")
            
            # Initialize OpenEvolve
            # I got it from the git repo explanations ... 
            
            # this is the most important part of the execution of this program ...
            
            # All the other important files are being utilized through this line of code, which are present inside the openevolve folder ...
            # controller.py ---> the main brain behind using the (num_iterations) evolution loop.
            # database.py ---> Stores/samples evolved strategies during the loop
            # evaluator.py ---> Calls my trading_strategy_evaluator.py, to handle timeouts, retries, and cascade evaluation
            # evaluation_result.py - Used to structure the results returned from your evaluator into OpenEvolve's internal format
            
            
            evolve = OpenEvolve(
                initial_program_path=str(self.existing_strategy_path),
                evaluation_file=str(self.evaluator_path),
                
                ## ----> through this, we utilize the sophisticated prompt settings automatically ?? 
                # The right config is my openevolve_config.yaml, and the left side is the parameter name
                # that OpenEvolve expects ... 
                config=config,   
                output_dir=str(self.output_dir)
            )
            
            # Run evolution
            print("Starting evolution process...")
            OpenEvolveBeststrategy = await evolve.run(iterations=config.max_iterations)
            

            # if OpenEvolveBeststrategy:
            #     print("=== ORIGINAL STRATEGY ===")
            #     with open(self.existing_strategy_path, 'r') as f:
            #         original = f.read()
            #     print(original[:500])
                
            #     print("\n=== EVOLVED STRATEGY ===")
            #     print(OpenEvolveBeststrategy.code[:500])
                
            #     print("\n=== CODE CHANGED? ===")
            #     print("Different code:", original != OpenEvolveBeststrategy.code)
            await self.process_results(OpenEvolveBeststrategy)
            return OpenEvolveBeststrategy
            
            
            
            
            
        except Exception as e:
            print(f"Evolution failed: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # async def process_results(self, OpenEvolveBeststrategy):
    #     """Save evolved strategy"""
    #     if OpenEvolveBeststrategy:
    #         # Save evolved strategy next to original
    #         original_dir = self.existing_strategy_path.parent
    #         evolved_path = original_dir / "evolved_strategies" / "OpenEvolveBeststrategy.py"
            
    #         with open(evolved_path, 'w') as f:
    #             f.write(OpenEvolveBeststrategy.code)
            
    #         # Save the best  metrics
    #         metrics_path = original_dir / "evolved_strategies" / "BestEvolutionMetrics.json"
    #         with open(metrics_path, 'w') as f:
    #             json.dump({
    #                 "program_id": OpenEvolveBeststrategy.id,
    #                 "generation": OpenEvolveBeststrategy.generation,
    #                 "metrics": OpenEvolveBeststrategy.metrics,
    #                 "timestamp": OpenEvolveBeststrategy.timestamp
    #             }, f, indent=2)
            
    #         print("\nEVOLUTION COMPLETED")
    #         print(f"Original: {self.existing_strategy_path}")
    #         print(f"Evolved: {evolved_path}")
    #         print(f"Metrics: {metrics_path}")
            
    #         # Performance summary
    #         metrics = OpenEvolveBeststrategy.metrics
    #         print(f"\nPERFORMANCE:")
    #         print(f"  Overall Fitness: {metrics.get('overall_fitness', 0):.4f}")
    #         print(f"  Profit: {metrics.get('profit_total_pct', 0):.2f}%")
    #         print(f"  Trades: {metrics.get('trades', 0)}")
    #         print(f"  Win Rate: {metrics.get('winrate', 0)*100:.1f}%")
    #         print(f"  Max Drawdown: {metrics.get('max_drawdown_account', 1)*100:.1f}%")
            
    #     else:
    #         print("No successful evolution found")
            
            
            
    async def process_results(self, OpenEvolveBeststrategy):
        """
        Save evolved strategy with validation to ensure we never save failed strategies as best.
        This function validates that the returned "best" strategy is actually successful before
        saving it to prevent zero-trade or failed strategies from being marked as optimal.
        """
        if OpenEvolveBeststrategy:
            # CRITICAL: Validate the "best" strategy before saving
            is_successful = OpenEvolveBeststrategy.metrics.get('IsSuccessfulEval')
            trades = OpenEvolveBeststrategy.metrics.get('trades', 0)
            fitness = OpenEvolveBeststrategy.metrics.get('overall_fitness', 0)
            profit_pct = OpenEvolveBeststrategy.metrics.get('profit_total_pct', -100)
            
            print(f"DEBUG: Final strategy validation:")
            print(f"  Strategy ID: {OpenEvolveBeststrategy.id}")
            print(f"  IsSuccessfulEval: {is_successful}")
            print(f"  Trades: {trades}")
            print(f"  Fitness: {fitness:.4f}")
            print(f"  Profit: {profit_pct:.2f}%")
            
            # REJECT if the "best" strategy is actually failed
            if is_successful < 0.5 or trades == 0:
                print("🚨 ERROR: OpenEvolve returned a failed strategy as 'best'!")
                print("This indicates the final selection logic has a bug.")
                print("The zero-trade rejection is working, but final selection is broken.")
                
                # Don't save failed strategies
                print("❌ Refusing to save failed strategy as 'best', so no creation of 'BestEvolutionMetrics.json' file ")
                return None
            
            # Only save if strategy is truly successful
            print(f"✅ Confirmed: Saving successful strategy")
            
            # Save evolved strategy next to original
            original_dir = self.existing_strategy_path.parent
            evolved_strategies_dir = original_dir / "evolved_strategies"
            evolved_strategies_dir.mkdir(exist_ok=True)
            
            evolved_path = evolved_strategies_dir / "OpenEvolveBeststrategy.py"
            
            with open(evolved_path, 'w') as f:
                f.write(OpenEvolveBeststrategy.code)
            
            # Save the best metrics with additional context
            metrics_path = evolved_strategies_dir / "BestEvolutionMetrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    "program_id": OpenEvolveBeststrategy.id,
                    "generation": OpenEvolveBeststrategy.generation,
                    "metrics": OpenEvolveBeststrategy.metrics,
                    "timestamp": OpenEvolveBeststrategy.timestamp,
                    "validation": {
                        "is_successful": is_successful,
                        "has_trades": trades > 0,
                        "fitness_score": fitness,
                        "profit_improvement": f"Strategy achieved {profit_pct:.2f}% profit",
                        "risk_metrics": f"Drawdown: {OpenEvolveBeststrategy.metrics.get('max_drawdown_account', 1)*100:.1f}%",
                        "trading_activity": f"{trades} trades, {OpenEvolveBeststrategy.metrics.get('winrate', 0)*100:.1f}% win rate"
                    }
                }, f, indent=2)
            
            print("\nEVOLUTION COMPLETED SUCCESSFULLY")
            print(f"Original: {self.existing_strategy_path}")
            print(f"Evolved: {evolved_path}")
            print(f"Metrics: {metrics_path}")
            
            # Performance summary with validated data
            metrics = OpenEvolveBeststrategy.metrics
            print(f"\nVALIDATED BEST PERFORMANCE:")
            print(f"  Overall Fitness: {fitness:.4f}")
            print(f"  Profit: {profit_pct:.2f}%")
            print(f"  Trades: {trades}")
            print(f"  Win Rate: {metrics.get('winrate', 0)*100:.1f}%")
            print(f"  Max Drawdown: {metrics.get('max_drawdown_account', 1)*100:.1f}%")
            
        else:
            print("❌ No successful evolution found - OpenEvolve returned None")
            return None
            
            
            

async def main():
    """Main entry point"""

    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    openai_key = os.environ.get('OPENAI_API_KEY')
    
    if not anthropic_key and not openai_key:
        print("ERROR: No API key found. Please set one.")
        sys.exit(1)
    
    try:
        integration = TradingOpenEvolveIntegration()
        await integration.run_evolution()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())