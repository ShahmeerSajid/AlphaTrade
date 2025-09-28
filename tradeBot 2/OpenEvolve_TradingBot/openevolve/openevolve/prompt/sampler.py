"""
Prompt sampling for OpenEvolve
"""


print("🔥 DEBUG: Using LOCAL sampler.py.....")

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

from openevolve.config import PromptConfig
from openevolve.prompt.templates import TemplateManager 
from openevolve.utils.format_utils import format_metrics_safe
from openevolve.utils.metrics_utils import safe_numeric_average

logger = logging.getLogger(__name__)


class PromptSampler:
    """Generates prompts for code evolution"""

    def __init__(self, config: PromptConfig):
        self.config = config  #  my config.prompt section (including my system_message inside openevolve_config.yaml file .... )
        self.template_manager = TemplateManager(config.template_dir)

        # Initialize the random number generator
        random.seed()

        # Store custom template mappings
        self.system_template_override = None
        self.user_template_override = None

        logger.info("Initialized prompt sampler")

    def set_templates(
        self, system_template: Optional[str] = None, user_template: Optional[str] = None
    ) -> None:
        """
        Set custom templates to use for this sampler

        Args:
            system_template: Template name for system message
            user_template: Template name for user message
        """
        self.system_template_override = system_template
        self.user_template_override = user_template
        logger.info(f"Set custom templates: system={system_template}, user={user_template}")

    def build_prompt(
        self,
        current_program: str = "",
        parent_program: str = "",
        program_metrics: Dict[str, float] = {},
        previous_programs: List[Dict[str, Any]] = [],
        top_programs: List[Dict[str, Any]] = [],
        inspirations: List[Dict[str, Any]] = [],  # Add inspirations parameter
        language: str = "python",
        evolution_round: int = 0,
        diff_based_evolution: bool = True,
        template_key: Optional[str] = None,
        program_artifacts: Optional[Dict[str, Union[str, bytes]]] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Build a prompt for the LLM

        Args:
            current_program: Current program code
            parent_program: Parent program from which current was derived
            program_metrics: Dictionary of metric names to values
            previous_programs: List of previous program attempts
            top_programs: List of top-performing programs (best by fitness)
            inspirations: List of inspiration programs (diverse/creative examples)
            language: Programming language
            evolution_round: Current evolution round
            diff_based_evolution: Whether to use diff-based evolution (True) or full rewrites (False)
            template_key: Optional override for template key
            program_artifacts: Optional artifacts from program evaluation
            **kwargs: Additional keys to replace in the user prompt

        Returns:
            Dictionary with 'system' and 'user' keys
        """
        # Select template based on evolution mode (with overrides)
        if template_key:
            # Use explicitly provided template key
            user_template_key = template_key
        elif self.user_template_override:
            # Use the override set with set_templates
            user_template_key = self.user_template_override
        else:
            # Default behavior: diff-based vs full rewrite
            user_template_key = "diff_user" if diff_based_evolution else "full_rewrite_user"

        # Get the template from templates.py ... added 
        user_template = self.template_manager.get_template(user_template_key)

        # Use system template override if set
        if self.system_template_override:
            system_message = self.template_manager.get_template(self.system_template_override)
        else:
            system_message = self.config.system_message # This is my yaml prompt ... added 
            # If system_message is a template name rather than content, get the template
            if system_message in self.template_manager.templates:
                system_message = self.template_manager.get_template(system_message)

        # Format metrics
        metrics_str = self._format_metrics(program_metrics)

        # Identify areas for improvement (Very important program ---> I refined the code ... added )
        improvement_areas = self._identify_improvement_areas(
            current_program, parent_program, program_metrics, previous_programs
        )

        # Format evolution history
        evolution_history = self._format_evolution_history(
            previous_programs, top_programs, inspirations, language
        )

        # Format artifacts section if enabled and available
        artifacts_section = ""
        if self.config.include_artifacts and program_artifacts:
            artifacts_section = self._render_artifacts(program_artifacts)

        # Apply stochastic template variations if enabled
        if self.config.use_template_stochasticity:
            user_template = self._apply_template_variations(user_template)

        # Format the final user message
        user_message = user_template.format(
            metrics=metrics_str,
            improvement_areas=improvement_areas,
            evolution_history=evolution_history,
            current_program=current_program,
            language=language,
            artifacts=artifacts_section,
            **kwargs,
        )

        # print("System message:" )
        # print(system_message)
        
        ## DEBUG: Check the prompt to llm (only user_message)
        print("user message:🦁🦁🦁🦁🦁🦁🦁🦁🦁🦁🦁v ")
        print(user_message)
        
        return {
            "system": system_message,
            "user": user_message,
        }

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for the prompt using safe formatting"""
        # Use safe formatting to handle mixed numeric and string values
        formatted_parts = []
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                try:
                    formatted_parts.append(f"- {name}: {value:.4f}")
                except (ValueError, TypeError):
                    formatted_parts.append(f"- {name}: {value}")
            else:
                formatted_parts.append(f"- {name}: {value}")
        return "\n".join(formatted_parts)




    # def _identify_improvement_areas(
    #     self,
    #     current_program: str,
    #     parent_program: str,
    #     metrics: Dict[str, float],
    #     previous_programs: List[Dict[str, Any]],
    # ) -> str:
    #     """Identify potential areas for improvement"""
        
    #     # This method could be expanded to include more sophisticated analysis
    #     # For now, we'll use a simple approach

    #     improvement_areas = []

    #     # Check program length
    #     if len(current_program) > 500:
    #         improvement_areas.append(
    #             "Consider simplifying the code to improve readability and maintainability"
    #         )

    #     # Check for performance patterns in previous attempts
    #     if len(previous_programs) >= 2:
    #         recent_attempts = previous_programs[-2:]
    #         metrics_improved = []
    #         metrics_regressed = []

    #         for metric, value in metrics.items():
    #             # Only compare numeric metrics
    #             if not isinstance(value, (int, float)) or isinstance(value, bool):
    #                 continue

    #             improved = True
    #             regressed = True

    #             for attempt in recent_attempts:
    #                 attempt_value = attempt["metrics"].get(metric, 0)
    #                 # Only compare if both values are numeric
    #                 if isinstance(value, (int, float)) and isinstance(attempt_value, (int, float)):
    #                     if attempt_value <= value:
    #                         regressed = False
    #                     if attempt_value >= value:
    #                         improved = False
    #                 else:
    #                     # If either value is non-numeric, skip comparison
    #                     improved = False
    #                     regressed = False

    #             if improved and metric not in metrics_improved:
    #                 metrics_improved.append(metric)
    #             if regressed and metric not in metrics_regressed:
    #                 metrics_regressed.append(metric)

    #         if metrics_improved:
    #             improvement_areas.append(
    #                 f"Metrics showing improvement: {', '.join(metrics_improved)}. "
    #                 "Consider continuing with similar changes."
    #             )

    #         if metrics_regressed:
    #             improvement_areas.append(
    #                 f"Metrics showing regression: {', '.join(metrics_regressed)}. "
    #                 "Consider reverting or revising recent changes in these areas."
    #             )

    #     # If we don't have specific improvements to suggest
    #     if not improvement_areas:
    #         improvement_areas.append(
    #             "Focus on optimizing the code for better performance on the target metrics"
    #         )

    #     return "\n".join([f"- {area}" for area in improvement_areas])
    
    
    
    
    
    
    
    
    
    # # my own coded identify_improvement_areas function:
    # def _identify_improvement_areas(
    #     self,
    #     current_program: str,
    #     parent_program: str,    
    #     metrics: Dict[str, float],
    #     previous_programs: List[Dict[str, Any]],
    #     ) -> str:
    #     """
    #     Enhanced trading-specific improvement area identification that analyzes performance patterns,
    #     risk metrics, trading behavior, and strategy effectiveness to provide sophisticated guidance
    #     for algorithmic trading strategy evolution.
    #     """
    #     improvement_areas = []
        
    #     # Extract key trading metrics safely
    #     def safe_get_metric(metrics_dict, key, default=0):
    #         value = metrics_dict.get(key, default)
    #         return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else default
        
    #     current_profit = safe_get_metric(metrics, 'profit_total_pct')
    #     current_winrate = safe_get_metric(metrics, 'winrate')
    #     current_drawdown = safe_get_metric(metrics, 'max_drawdown_account')
    #     current_trades = safe_get_metric(metrics, 'trades')
    #     current_sharpe = safe_get_metric(metrics, 'sharpe', 0)
        
    #     # 1. RISK-ADJUSTED PERFORMANCE ANALYSIS
    #     if current_profit > 0 and current_drawdown > 0.15:
    #         improvement_areas.append(
    #             f"HIGH RISK CONCERN: Strategy shows {current_profit:.1f}% profit but {current_drawdown*100:.1f}% drawdown. "
    #             "Focus on implementing stricter stop-losses, position sizing, or volatility-based risk management."
    #         )
    #     elif current_profit > 5 and current_drawdown < 0.05:
    #         improvement_areas.append(
    #             f"EXCELLENT RISK PROFILE: {current_profit:.1f}% profit with only {current_drawdown*100:.1f}% drawdown. "
    #             "Consider slightly increasing position sizes or entry frequency to capitalize on this risk-efficient approach."
    #         )
        
    #     # 2. SHARPE RATIO AND RISK-ADJUSTED ANALYSIS
    #     if current_sharpe < 0.5 and current_profit > 0:
    #         improvement_areas.append(
    #             f"LOW RISK-ADJUSTED RETURNS: Sharpe ratio of {current_sharpe:.2f} indicates poor risk-adjusted performance. "
    #             "Focus on reducing trade frequency, improving entry timing, or implementing volatility filters."
    #         )
    #     elif current_sharpe > 1.5:
    #         improvement_areas.append(
    #             f"EXCEPTIONAL SHARPE RATIO: {current_sharpe:.2f} indicates excellent risk-adjusted returns. "
    #             "Maintain current approach and consider scaling position sizes or adding similar entry conditions."
    #         )
        
    #     # 3. TRADING FREQUENCY AND EFFICIENCY ANALYSIS
    #     if current_trades < 5:
    #         improvement_areas.append(
    #             f"LOW TRADING ACTIVITY: Only {current_trades} trades executed. "
    #             "Consider relaxing entry conditions, reducing indicator thresholds, or adding alternative entry signals to increase opportunity capture."
    #         )
    #     elif current_trades > 900:
    #         improvement_areas.append(
    #             f"POTENTIAL OVERTRADING: {current_trades} trades may indicate excessive activity. "
    #             "Consider tightening entry conditions, adding cooldown periods, or implementing minimum profit targets to reduce noise trades."
    #         )
        
    #     # 4. WIN RATE OPTIMIZATION GUIDANCE
    #     if current_winrate < 0.4:
    #         improvement_areas.append(
    #             f"LOW WIN RATE: {current_winrate*100:.1f}% win rate suggests poor entry timing. "
    #             "Focus on improving entry signal quality, adding trend confirmation, or implementing better market condition filters."
    #         )
    #     elif current_winrate > 0.7 and current_profit < 2:
    #         improvement_areas.append(
    #             f"HIGH WIN RATE BUT LOW PROFIT: {current_winrate*100:.1f}% win rate with {current_profit:.1f}% profit suggests small wins, large losses. "
    #             "Focus on profit-taking strategies, trailing stops, or reducing stop-loss distances to improve profit capture."
    #         )
        
    #     # 5. ADVANCED HISTORICAL PATTERN ANALYSIS
    #     if len(previous_programs) >= 3:
    #         # Analyze last 5 attempts for deeper patterns
    #         recent_attempts = previous_programs[-5:] if len(previous_programs) >= 5 else previous_programs
            
    #         # Calculate performance trends
    #         profit_trend = []
    #         drawdown_trend = []
    #         winrate_trend = []
    #         trades_trend = []
            
    #         for attempt in recent_attempts:
    #             attempt_metrics = attempt.get("metrics", {})
    #             profit_trend.append(safe_get_metric(attempt_metrics, 'profit_total_pct'))
    #             drawdown_trend.append(safe_get_metric(attempt_metrics, 'max_drawdown_account'))
    #             winrate_trend.append(safe_get_metric(attempt_metrics, 'winrate'))
    #             trades_trend.append(safe_get_metric(attempt_metrics, 'trades'))
            
    #         # Detect performance patterns
    #         if len(profit_trend) >= 3:
    #             recent_profit_changes = [profit_trend[i] - profit_trend[i-1] for i in range(1, len(profit_trend))]
                
    #             if all(change > 0 for change in recent_profit_changes[-2:]):
    #                 improvement_areas.append(
    #                     "POSITIVE MOMENTUM: Profit has been consistently improving over recent iterations. "
    #                     "Continue with current optimization direction and consider amplifying successful changes."
    #                 )
    #             elif all(change < 0 for change in recent_profit_changes[-2:]):
    #                 improvement_areas.append(
    #                     "NEGATIVE TREND DETECTED: Profit declining over recent attempts. "
    #                     "Consider reverting to earlier successful approaches or trying completely different strategy elements."
    #                 )
            
    #         # Risk trend analysis
    #         if len(drawdown_trend) >= 3:
    #             if drawdown_trend[-1] > max(drawdown_trend[:-1]) * 1.5:
    #                 improvement_areas.append(
    #                     "ESCALATING RISK: Recent changes significantly increased drawdown. "
    #                     "Immediately focus on risk management - implement stricter position sizing or stop-losses."
    #                 )
        
    #     # 6. STRATEGY TYPE AND MARKET CONDITION ANALYSIS
    #     code_lower = current_program.lower()
        
    #     # Detect strategy characteristics from code
    #     has_rsi = 'rsi' in code_lower
    #     has_ema = 'ema' in code_lower or 'sma' in code_lower
    #     has_volume = 'volume' in code_lower
    #     has_bollinger = 'bb_' in code_lower or 'bollinger' in code_lower
    #     has_macd = 'macd' in code_lower
        
    #     # Strategy-specific guidance
    #     if has_rsi and current_winrate < 0.45:
    #         improvement_areas.append(
    #             "RSI-BASED STRATEGY UNDERPERFORMING: Consider combining RSI with trend confirmation (EMA crossover) "
    #             "or adding volume validation to filter false RSI signals in choppy markets."
    #         )
        
    #     if has_ema and current_profit < 1:
    #         improvement_areas.append(
    #             "EMA STRATEGY LOW PROFIT: Moving average strategies often struggle in sideways markets. "
    #             "Consider adding volatility filters (ATR) or trend strength indicators to avoid whipsaw trades."
    #         )
        
    #     if not has_volume and current_trades > 50:
    #         improvement_areas.append(
    #             "MISSING VOLUME VALIDATION: High trade count without volume filters may include low-conviction signals. "
    #             "Add volume confirmation (volume > average) to filter out weak breakouts and improve signal quality."
    #         )
        
    #     # 7. PERFORMANCE THRESHOLD ANALYSIS
    #     if current_profit > 0 and current_profit < 1:
    #         improvement_areas.append(
    #             f"MARGINAL PROFITABILITY: {current_profit:.2f}% profit is barely above break-even. "
    #             "Focus on high-conviction trades only - tighten entry conditions and improve exit timing to increase profit per trade."
    #         )
    #     elif current_profit > 10:
    #         improvement_areas.append(
    #             f"EXCEPTIONAL PERFORMANCE: {current_profit:.1f}% profit is outstanding. "
    #             "Validate robustness by testing on different timeframes or market conditions. Consider partial profit-taking strategies."
    #         )
        
    #     # 8. CORRELATION AND METRIC RELATIONSHIP ANALYSIS
    #     if current_profit > 3 and current_winrate < 0.4:
    #         improvement_areas.append(
    #             "HIGH PROFIT, LOW WIN RATE: This suggests a few large wins compensate for many small losses. "
    #             "This can be sustainable but risky. Consider implementing partial profit-taking or trailing stops to secure gains."
    #         )
    #     elif current_profit < 1 and current_winrate > 0.6:
    #         improvement_areas.append(
    #             "HIGH WIN RATE, LOW PROFIT: Many small wins are being offset by large losses. "
    #             "Critically review stop-loss placement and consider reducing maximum loss per trade or implementing faster exit signals."
    #         )
        
    #     # 9. TRADING EFFICIENCY METRICS
    #     if current_trades > 0:
    #         avg_profit_per_trade = current_profit / current_trades
    #         if avg_profit_per_trade < 0.05:  # Less than 0.05% per trade
    #             improvement_areas.append(
    #                 f"LOW PROFIT PER TRADE: Averaging {avg_profit_per_trade:.3f}% per trade suggests inefficient trading. "
    #                 "Focus on higher-conviction signals, better entry timing, or improved profit-taking strategies."
    #             )
        
    #     # 10. CODE COMPLEXITY AND MAINTAINABILITY
    #     evolve_blocks = current_program.count('EVOLVE-BLOCK-START')
    #     if evolve_blocks > 3:
    #         improvement_areas.append(
    #             f"HIGH COMPLEXITY: {evolve_blocks} evolve blocks may indicate over-optimization. "
    #             "Consider simplifying to 1-2 core signal types and focus on perfecting those rather than adding complexity."
    #         )
    #     elif evolve_blocks == 0:
    #         improvement_areas.append(
    #             "NO EVOLVE BLOCKS: Add EVOLVE-BLOCK markers around entry/exit logic to enable targeted evolution. "
    #             "This will focus improvements on trading logic rather than structural changes."
    #         )
        
    #     # Default guidance if no specific areas identified
    #     if not improvement_areas:
    #         if current_profit <= 0:
    #             improvement_areas.append(
    #                 "BASELINE OPTIMIZATION NEEDED: Focus on fundamental improvements - better entry signal timing, "
    #                 "proper stop-loss implementation, and volume validation for trade confirmation."
    #             )
    #         else:
    #             improvement_areas.append(
    #                 "INCREMENTAL OPTIMIZATION: Current strategy shows promise. Focus on fine-tuning entry thresholds, "
    #                 "optimizing exit timing, and enhancing risk management for consistency."
    #             )
        
    #     # Format with priority indicators
    #     prioritized_areas = []
    #     for i, area in enumerate(improvement_areas[:6]):  # Limit to top 6 most important
    #         priority = "-----> CRITICAL" if "HIGH RISK" in area or "NEGATIVE TREND" in area else \
    #                 "-----> IMPORTANT" if "LOW" in area or "HIGH" in area else \
    #                 " ------> OPTIMIZE"
    #         prioritized_areas.append(f"{priority}: {area}")
        
    #     return "\n".join(prioritized_areas)
    
    
    
    
    
    
    
    
    def _identify_improvement_areas(
    self,
    current_program: str,
    parent_program: str,
    metrics: Dict[str, float],
    previous_programs: List[Dict[str, Any]],
    ) -> str:
        """
        Identify specific improvement areas for trading strategy evolution.
        Provides concrete guidance on what aspects of the trading logic need
        modification to achieve better profitability and risk management.
        """
        improvement_areas = []
        
        # Get current performance metrics
        profit_pct = metrics.get('profit_total_pct', 0)
        trades = metrics.get('trades', 0)
        winrate = metrics.get('winrate', 0)
        max_drawdown = metrics.get('max_drawdown_account', 1)
        
        # CRITICAL: Reject strategies with zero trades
        if trades == 0:
            improvement_areas.append(
                "CRITICAL: Strategy makes no trades. Modify entry conditions to be less restrictive."
                "Consider: lowering RSI thresholds, reducing volume requirements, or relaxing Bollinger band conditions."
            )
            return "\n".join([f"- {area}" for area in improvement_areas])
        
        # Provide specific trading improvements based on current performance
        if profit_pct < -50:
            improvement_areas.append(
                "Strategy has severe losses. Focus on tightening entry conditions and improving exit timing. "
                "Consider: stricter volume filters, trend confirmation, or earlier profit-taking."
            )
        elif profit_pct < -20:
            improvement_areas.append(
                "Strategy shows significant losses. Improve risk management by adjusting stop-loss levels "
                "and optimizing entry/exit RSI thresholds for better timing."
            )
        elif profit_pct < 0:
            improvement_areas.append(
                "Strategy is close to breakeven. Fine-tune entry conditions for better selectivity. "
                "Consider: optimizing RSI levels, improving volume filters, or adding momentum indicators."
            )
        else:
            improvement_areas.append(
                "Strategy is profitable. Focus on optimizing position sizing and fine-tuning exit conditions "
                "to maximize returns while maintaining risk control."
            )
        
        # Risk management improvements
        if max_drawdown > 0.5:
            improvement_areas.append(
                "High drawdown detected. Implement stricter risk controls: tighter stop-losses, "
                "better trend filtering, or reduced position sizes during volatile periods."
            )
        elif max_drawdown > 0.3:
            improvement_areas.append(
                "Moderate drawdown. Consider improving exit logic with trailing stops or "
                "dynamic position sizing based on market volatility."
            )
        
        # Win rate improvements
        if winrate < 0.3:
            improvement_areas.append(
                "Low win rate suggests poor entry timing. Improve signal quality by adding "
                "confluence factors: combine RSI with MACD signals or trend confirmation."
            )
        elif winrate > 0.8:
            improvement_areas.append(
                "Very high win rate may indicate overly conservative strategy. Consider taking "
                "more selective but higher-probability trades with better risk/reward ratios."
            )
        
        # Trading frequency optimization
        if trades < 10:
            improvement_areas.append(
                "Very few trades. Strategy may be too restrictive. Consider relaxing entry conditions "
                "or expanding timeframe coverage to capture more opportunities."
            )
        elif trades > 200:
            improvement_areas.append(
                "High trading frequency detected. Focus on trade quality over quantity. "
                "Tighten entry conditions to filter out low-probability setups."
            )
        
        # Historical comparison improvements
        if len(previous_programs) >= 2:
            recent_profits = [p.get("metrics", {}).get("profit_total_pct", -100) for p in previous_programs[-3:]]
            if all(p < profit_pct for p in recent_profits):
                improvement_areas.append(
                    "Recent improvements detected. Continue evolving in the same direction: "
                    "refine current entry/exit logic while maintaining risk management principles."
                )
            elif profit_pct < max(recent_profits):
                improvement_areas.append(
                    "Performance regression detected. Revert to previous successful approach and "
                    "make smaller, incremental changes to preserve gains."
                )
        
        # Default guidance if no specific issues identified
        if not improvement_areas:
            improvement_areas.append(
                "Continue optimizing entry timing with RSI and volume filters. "
                "Focus on improving risk-adjusted returns through better exit strategies."
            )
        
        print("ADDED THE IMPROVEMENT AREA 🥶👺👺👺👺👺👺👺👺👺👺👺👺👺👺 ")
        # print(improvement_areas)
        return "\n".join([f"- {area}" for area in improvement_areas])
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def _format_evolution_history(
        self,
        previous_programs: List[Dict[str, Any]],
        top_programs: List[Dict[str, Any]],
        inspirations: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Format the evolution history for the prompt"""
        # Get templates
        history_template = self.template_manager.get_template("evolution_history")
        previous_attempt_template = self.template_manager.get_template("previous_attempt")
        top_program_template = self.template_manager.get_template("top_program")

        # Format previous attempts (most recent first)
        previous_attempts_str = ""
        selected_previous = previous_programs[-min(3, len(previous_programs)) :]

        for i, program in enumerate(reversed(selected_previous)):
            attempt_number = len(previous_programs) - i
            changes = program.get("changes", "Unknown changes")

            # Format performance metrics using safe formatting
            performance_parts = []
            for name, value in program.get("metrics", {}).items():
                if isinstance(value, (int, float)):
                    try:
                        performance_parts.append(f"{name}: {value:.4f}")
                    except (ValueError, TypeError):
                        performance_parts.append(f"{name}: {value}")
                else:
                    performance_parts.append(f"{name}: {value}")
            performance_str = ", ".join(performance_parts)

            # Determine outcome based on comparison with parent (only numeric metrics)
            parent_metrics = program.get("parent_metrics", {})
            outcome = "Mixed results"

            # Safely compare only numeric metrics
            program_metrics = program.get("metrics", {})

            # Check if all numeric metrics improved
            numeric_comparisons_improved = []
            numeric_comparisons_regressed = []

            for m in program_metrics:
                prog_value = program_metrics.get(m, 0)
                parent_value = parent_metrics.get(m, 0)

                # Only compare if both values are numeric
                if isinstance(prog_value, (int, float)) and isinstance(parent_value, (int, float)):
                    if prog_value > parent_value:
                        numeric_comparisons_improved.append(True)
                    else:
                        numeric_comparisons_improved.append(False)

                    if prog_value < parent_value:
                        numeric_comparisons_regressed.append(True)
                    else:
                        numeric_comparisons_regressed.append(False)

            # Determine outcome based on numeric comparisons
            if numeric_comparisons_improved and all(numeric_comparisons_improved):
                outcome = "Improvement in all metrics"
            elif numeric_comparisons_regressed and all(numeric_comparisons_regressed):
                outcome = "Regression in all metrics"

            previous_attempts_str += (
                previous_attempt_template.format(
                    attempt_number=attempt_number,
                    changes=changes,
                    performance=performance_str,
                    outcome=outcome,
                )
                + "\n\n"
            )

        # Format top programs
        top_programs_str = ""
        selected_top = top_programs[: min(self.config.num_top_programs, len(top_programs))]

        for i, program in enumerate(selected_top):
            # Extract a snippet (first 10 lines) for display
            program_code = program.get("code", "")
            program_snippet = "\n".join(program_code.split("\n")[:10])
            if len(program_code.split("\n")) > 10:
                program_snippet += "\n# ... (truncated for brevity)"

            # Calculate a composite score using safe numeric average
            score = safe_numeric_average(program.get("metrics", {}))

            # Extract key features (this could be more sophisticated)
            key_features = program.get("key_features", [])
            if not key_features:
                key_features = []
                for name, value in program.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        try:
                            key_features.append(f"Performs well on {name} ({value:.4f})")
                        except (ValueError, TypeError):
                            key_features.append(f"Performs well on {name} ({value})")
                    else:
                        key_features.append(f"Performs well on {name} ({value})")

            key_features_str = ", ".join(key_features)

            top_programs_str += (
                top_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    language=language,
                    program_snippet=program_snippet,
                    key_features=key_features_str,
                )
                + "\n\n"
            )

        # Format diverse programs using num_diverse_programs config
        diverse_programs_str = ""
        if (
            self.config.num_diverse_programs > 0
            and len(top_programs) > self.config.num_top_programs
        ):
            # Skip the top programs we already included
            remaining_programs = top_programs[self.config.num_top_programs :]

            # Sample diverse programs from the remaining
            num_diverse = min(self.config.num_diverse_programs, len(remaining_programs))
            if num_diverse > 0:
                # Use random sampling to get diverse programs
                diverse_programs = random.sample(remaining_programs, num_diverse)

                diverse_programs_str += "\n\n## Diverse Programs\n\n"

                for i, program in enumerate(diverse_programs):
                    # Extract a snippet (first 5 lines for diversity)
                    program_code = program.get("code", "")
                    program_snippet = "\n".join(program_code.split("\n")[:5])
                    if len(program_code.split("\n")) > 5:
                        program_snippet += "\n# ... (truncated)"

                    # Calculate a composite score using safe numeric average
                    score = safe_numeric_average(program.get("metrics", {}))

                    # Extract key features
                    key_features = program.get("key_features", [])
                    if not key_features:
                        key_features = [
                            f"Alternative approach to {name}"
                            for name in list(program.get("metrics", {}).keys())[
                                :2
                            ]  # Just first 2 metrics
                        ]

                    key_features_str = ", ".join(key_features)

                    diverse_programs_str += (
                        top_program_template.format(
                            program_number=f"D{i + 1}",
                            score=f"{score:.4f}",
                            language=language,
                            program_snippet=program_snippet,
                            key_features=key_features_str,
                        )
                        + "\n\n"
                    )

        # Combine top and diverse programs
        combined_programs_str = top_programs_str + diverse_programs_str

        # Format inspirations section
        inspirations_section_str = self._format_inspirations_section(inspirations, language)

        # Combine into full history
        return history_template.format(
            previous_attempts=previous_attempts_str.strip(),
            top_programs=combined_programs_str.strip(),
            inspirations_section=inspirations_section_str,
        )

    def _format_inspirations_section(
        self, inspirations: List[Dict[str, Any]], language: str
    ) -> str:
        """
        Format the inspirations section for the prompt
        
        Args:
            inspirations: List of inspiration programs
            language: Programming language
            
        Returns:
            Formatted inspirations section string
        """
        if not inspirations:
            return ""
            
        # Get templates
        inspirations_section_template = self.template_manager.get_template("inspirations_section")
        inspiration_program_template = self.template_manager.get_template("inspiration_program")
        
        inspiration_programs_str = ""
        
        for i, program in enumerate(inspirations):
            # Extract a snippet (first 8 lines) for display
            program_code = program.get("code", "")
            program_snippet = "\n".join(program_code.split("\n")[:8])
            if len(program_code.split("\n")) > 8:
                program_snippet += "\n# ... (truncated for brevity)"
            
            # Calculate a composite score using safe numeric average
            score = safe_numeric_average(program.get("metrics", {}))
            
            # Determine program type based on metadata and score
            program_type = self._determine_program_type(program)
            
            # Extract unique features (emphasizing diversity rather than just performance)
            unique_features = self._extract_unique_features(program)
            
            inspiration_programs_str += (
                inspiration_program_template.format(
                    program_number=i + 1,
                    score=f"{score:.4f}",
                    program_type=program_type,
                    language=language,
                    program_snippet=program_snippet,
                    unique_features=unique_features,
                )
                + "\n\n"
            )
        
        return inspirations_section_template.format(
            inspiration_programs=inspiration_programs_str.strip()
        )
        
    def _determine_program_type(self, program: Dict[str, Any]) -> str:
        """
        Determine the type/category of an inspiration program
        
        Args:
            program: Program dictionary
            
        Returns:
            String describing the program type
        """
        metadata = program.get("metadata", {})
        score = safe_numeric_average(program.get("metrics", {}))
        
        # Check metadata for explicit type markers
        if metadata.get("diverse", False):
            return "Diverse"
        if metadata.get("migrant", False):
            return "Migrant"
        if metadata.get("random", False):
            return "Random"
            
        # Classify based on score ranges
        if score >= 0.8:
            return "High-Performer"
        elif score >= 0.6:
            return "Alternative"
        elif score >= 0.4:
            return "Experimental"
        else:
            return "Exploratory"
            
    def _extract_unique_features(self, program: Dict[str, Any]) -> str:
        """
        Extract unique features of an inspiration program
        
        Args:
            program: Program dictionary
            
        Returns:
            String describing unique aspects of the program
        """
        features = []
        
        # Extract from metadata if available
        metadata = program.get("metadata", {})
        if "changes" in metadata:
            changes = metadata["changes"]
            if isinstance(changes, str) and len(changes) < 100:
                features.append(f"Modification: {changes}")
        
        # Analyze metrics for standout characteristics
        metrics = program.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if value >= 0.9:
                    features.append(f"Excellent {metric_name} ({value:.3f})")
                elif value <= 0.3:
                    features.append(f"Alternative {metric_name} approach")
        
        # Code-based features (simple heuristics)
        code = program.get("code", "")
        if code:
            code_lower = code.lower()
            if "class" in code_lower and "def __init__" in code_lower:
                features.append("Object-oriented approach")
            if "numpy" in code_lower or "np." in code_lower:
                features.append("NumPy-based implementation")
            if "for" in code_lower and "while" in code_lower:
                features.append("Mixed iteration strategies")
            if len(code.split("\n")) < 10:
                features.append("Concise implementation")
            elif len(code.split("\n")) > 50:
                features.append("Comprehensive implementation")
        
        # Default if no specific features found
        if not features:
            program_type = self._determine_program_type(program)
            features.append(f"{program_type} approach to the problem")
            
        return ", ".join(features[:3])  # Limit to top 3 features

    def _apply_template_variations(self, template: str) -> str:
        """Apply stochastic variations to the template"""
        result = template

        # Apply variations defined in the config
        for key, variations in self.config.template_variations.items():
            if variations and f"{{{key}}}" in result:
                chosen_variation = random.choice(variations)
                result = result.replace(f"{{{key}}}", chosen_variation)

        return result

    def _render_artifacts(self, artifacts: Dict[str, Union[str, bytes]]) -> str:
        """
        Render artifacts for prompt inclusion

        Args:
            artifacts: Dictionary of artifact name to content

        Returns:
            Formatted string for prompt inclusion (empty string if no artifacts)
        """
        if not artifacts:
            return ""

        sections = []

        # Process all artifacts using .items()
        for key, value in artifacts.items():
            content = self._safe_decode_artifact(value)
            # Truncate if too long
            if len(content) > self.config.max_artifact_bytes:
                content = content[: self.config.max_artifact_bytes] + "\n... (truncated)"

            sections.append(f"### {key}\n```\n{content}\n```")

        if sections:
            return "## Last Execution Output\n\n" + "\n\n".join(sections)
        else:
            return ""

    def _safe_decode_artifact(self, value: Union[str, bytes]) -> str:
        """
        Safely decode an artifact value to string

        Args:
            value: Artifact value (string or bytes)

        Returns:
            String representation of the value
        """
        if isinstance(value, str):
            # Apply security filter if enabled
            if self.config.artifact_security_filter:
                return self._apply_security_filter(value)
            return value
        elif isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="replace")
                if self.config.artifact_security_filter:
                    return self._apply_security_filter(decoded)
                return decoded
            except Exception:
                return f"<binary data: {len(value)} bytes>"
        else:
            return str(value)

    def _apply_security_filter(self, text: str) -> str:
        """
        Apply security filtering to artifact text

        Args:
            text: Input text

        Returns:
            Filtered text with potential secrets/sensitive info removed
        """
        import re

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        filtered = ansi_escape.sub("", text)

        # Basic patterns for common secrets (can be expanded)
        secret_patterns = [
            (r"[A-Za-z0-9]{32,}", "<REDACTED_TOKEN>"),  # Long alphanumeric tokens
            (r"sk-[A-Za-z0-9]{48}", "<REDACTED_API_KEY>"),  # OpenAI-style API keys
            (r"password[=:]\s*[^\s]+", "password=<REDACTED>"),  # Password assignments
            (r"token[=:]\s*[^\s]+", "token=<REDACTED>"),  # Token assignments
        ]

        for pattern, replacement in secret_patterns:
            filtered = re.sub(pattern, replacement, filtered, flags=re.IGNORECASE)

        return filtered
