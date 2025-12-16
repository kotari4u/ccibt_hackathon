"""
Scenario simulation engine for market analysis.
Implements Monte Carlo simulations and scenario analysis.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import norm
import structlog

logger = structlog.get_logger(__name__)


class ScenarioSimulator:
    """
    Simulates market scenarios using Monte Carlo and historical analog methods.
    
    Supports stress testing, scenario analysis, and liquidity-adjusted modeling.
    """
    
    def __init__(self, n_simulations: int = 10000):
        """
        Initialize scenario simulator.
        
        Args:
            n_simulations: Number of Monte Carlo simulations
        """
        self.n_simulations = n_simulations
        self.logger = logger
    
    def monte_carlo_simulation(
        self,
        current_price: float,
        volatility: float,
        drift: float = 0.0,
        days: int = 5,
        n_simulations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for price paths.
        
        Args:
            current_price: Current price
            volatility: Annualized volatility
            drift: Annualized drift (expected return)
            days: Number of days to simulate
            n_simulations: Number of simulation paths
            
        Returns:
            Dictionary with simulation results
        """
        n_sim = n_simulations or self.n_simulations
        
        # Convert to daily parameters
        dt = 1 / 252  # Daily time step
        daily_drift = drift * dt
        daily_vol = volatility * np.sqrt(dt)
        
        # Generate random shocks
        random_shocks = np.random.normal(0, 1, (n_sim, days))
        
        # Simulate price paths
        price_paths = np.zeros((n_sim, days + 1))
        price_paths[:, 0] = current_price
        
        for t in range(days):
            price_paths[:, t + 1] = price_paths[:, t] * np.exp(
                daily_drift - 0.5 * daily_vol**2 + daily_vol * random_shocks[:, t]
            )
        
        # Calculate statistics
        final_prices = price_paths[:, -1]
        
        results = {
            "current_price": float(current_price),
            "simulated_prices": final_prices.tolist(),
            "mean_price": float(np.mean(final_prices)),
            "median_price": float(np.median(final_prices)),
            "std_price": float(np.std(final_prices)),
            "percentiles": {
                "p5": float(np.percentile(final_prices, 5)),
                "p25": float(np.percentile(final_prices, 25)),
                "p50": float(np.percentile(final_prices, 50)),
                "p75": float(np.percentile(final_prices, 75)),
                "p95": float(np.percentile(final_prices, 95)),
            },
            "probability_up": float(np.mean(final_prices > current_price)),
            "probability_down": float(np.mean(final_prices < current_price)),
            "n_simulations": n_sim,
            "days": days,
        }
        
        logger.info(
            "Monte Carlo simulation completed",
            n_simulations=n_sim,
            mean_price=results["mean_price"],
        )
        
        return results
    
    def scenario_analysis(
        self,
        base_volatility: float,
        scenarios: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Analyze different market scenarios.
        
        Args:
            base_volatility: Base volatility level
            scenarios: Dictionary of scenario names and parameters
                       e.g., {"bull": {"vol_multiplier": 0.8, "drift": 0.1},
                              "bear": {"vol_multiplier": 1.5, "drift": -0.1}}
            
        Returns:
            Dictionary with scenario analysis results
        """
        results = {}
        
        for scenario_name, params in scenarios.items():
            vol_multiplier = params.get("vol_multiplier", 1.0)
            drift = params.get("drift", 0.0)
            adjusted_vol = base_volatility * vol_multiplier
            
            results[scenario_name] = {
                "volatility": float(adjusted_vol),
                "volatility_multiplier": float(vol_multiplier),
                "drift": float(drift),
                "expected_range": {
                    "lower": float(base_volatility * vol_multiplier * 0.8),
                    "upper": float(base_volatility * vol_multiplier * 1.2),
                },
            }
        
        logger.info("Scenario analysis completed", n_scenarios=len(scenarios))
        
        return results
    
    def find_historical_analogs(
        self,
        market_data: pd.DataFrame,
        current_features: Dict[str, float],
        n_analogs: int = 10,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        Find historical periods similar to current market conditions.
        
        Args:
            market_data: Historical market data
            current_features: Dictionary of current market features
            n_analogs: Number of analogs to return
            lookback_days: Days to look back
            
        Returns:
            DataFrame with similar historical periods
        """
        # Filter recent data
        if 'timestamp' in market_data.columns:
            market_data = market_data.copy()
            market_data['timestamp'] = pd.to_datetime(market_data['timestamp'])
            cutoff_date = market_data['timestamp'].max() - timedelta(days=lookback_days)
            historical_data = market_data[market_data['timestamp'] >= cutoff_date].copy()
        else:
            historical_data = market_data.tail(lookback_days).copy()
        
        # Calculate similarity scores
        similarities = []
        
        for idx, row in historical_data.iterrows():
            score = 0.0
            matches = 0
            
            for feature, current_value in current_features.items():
                if feature in row.index and not pd.isna(row[feature]):
                    # Normalized difference
                    feature_range = historical_data[feature].max() - historical_data[feature].min()
                    if feature_range > 0:
                        diff = abs(row[feature] - current_value) / feature_range
                        score += 1 - diff  # Higher score for smaller difference
                        matches += 1
            
            if matches > 0:
                similarities.append({
                    "index": idx,
                    "similarity_score": score / matches,
                    "timestamp": row.get('timestamp', idx),
                })
        
        # Sort by similarity and return top analogs
        similarities_df = pd.DataFrame(similarities)
        if len(similarities_df) == 0:
            return pd.DataFrame()
        
        analogs = similarities_df.nlargest(n_analogs, 'similarity_score')
        
        # Merge with original data
        result = historical_data.loc[analogs['index']].copy()
        result['similarity_score'] = analogs['similarity_score'].values
        
        logger.info(
            "Historical analogs found",
            n_analogs=len(result),
            avg_similarity=float(analogs['similarity_score'].mean()),
        )
        
        return result.sort_values('similarity_score', ascending=False)
    
    def stress_test(
        self,
        portfolio_value: float,
        positions: Dict[str, float],
        market_data: pd.DataFrame,
        stress_scenarios: List[Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Dictionary of {symbol: weight}
            market_data: Market data DataFrame
            stress_scenarios: List of stress scenario dictionaries
                            e.g., [{"name": "crash", "price_change": -0.2, "vol_multiplier": 2.0}]
            
        Returns:
            Dictionary with stress test results
        """
        results = {}
        
        for scenario in stress_scenarios:
            scenario_name = scenario.get("name", "unknown")
            price_change = scenario.get("price_change", 0.0)
            vol_multiplier = scenario.get("vol_multiplier", 1.0)
            
            # Calculate portfolio impact
            portfolio_change = 0.0
            for symbol, weight in positions.items():
                symbol_data = market_data[market_data['symbol'] == symbol]
                if len(symbol_data) > 0:
                    current_price = symbol_data['close'].iloc[-1]
                    new_price = current_price * (1 + price_change)
                    position_change = (new_price - current_price) / current_price * weight
                    portfolio_change += position_change
            
            new_portfolio_value = portfolio_value * (1 + portfolio_change)
            loss = portfolio_value - new_portfolio_value
            
            results[scenario_name] = {
                "price_change": float(price_change),
                "volatility_multiplier": float(vol_multiplier),
                "portfolio_change": float(portfolio_change),
                "new_portfolio_value": float(new_portfolio_value),
                "loss": float(loss),
                "loss_percentage": float(-portfolio_change * 100),
            }
        
        logger.info("Stress test completed", n_scenarios=len(stress_scenarios))
        
        return results
    
    def liquidity_adjusted_impact(
        self,
        trade_size: float,
        daily_volume: float,
        volatility: float,
        impact_model: str = "square_root",
    ) -> Dict[str, float]:
        """
        Calculate liquidity-adjusted market impact.
        
        Args:
            trade_size: Size of trade
            daily_volume: Average daily volume
            volatility: Current volatility
            impact_model: Impact model type ('square_root', 'linear')
            
        Returns:
            Dictionary with impact metrics
        """
        participation_rate = trade_size / daily_volume
        
        if impact_model == "square_root":
            # Square root model: impact ~ sqrt(participation_rate)
            impact = volatility * np.sqrt(participation_rate) * 0.1
        elif impact_model == "linear":
            # Linear model: impact ~ participation_rate
            impact = volatility * participation_rate * 0.5
        else:
            raise ValueError(f"Unknown impact model: {impact_model}")
        
        return {
            "participation_rate": float(participation_rate),
            "estimated_impact": float(impact),
            "impact_percentage": float(impact * 100),
            "model": impact_model,
        }

