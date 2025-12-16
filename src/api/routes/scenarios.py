"""
API routes for scenario simulation.
"""

from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import structlog

from src.api.schemas import ScenarioRequest, ScenarioResponse
from src.models.scenario_simulator import ScenarioSimulator
# Authentication disabled
# from src.api.auth import get_current_user

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/scenarios", tags=["scenarios"])

scenario_simulator = ScenarioSimulator()


@router.post("/", response_model=ScenarioResponse)
async def run_scenario(
    request: ScenarioRequest,
    # Authentication disabled
    # current_user: dict = Depends(get_current_user),
) -> ScenarioResponse:
    """
    Run scenario simulation.
    
    Args:
        request: Scenario simulation parameters
        current_user: Authenticated user
        
    Returns:
        Scenario simulation results
    """
    try:
        logger.info(
            "Scenario simulation request",
            symbol=request.symbol,
            user="anonymous",
        )
        
        # Run Monte Carlo simulation
        mc_results = scenario_simulator.monte_carlo_simulation(
            current_price=request.current_price,
            volatility=request.volatility,
            days=request.days,
            n_simulations=request.n_simulations,
        )
        
        # Run scenario analysis if scenarios provided
        scenario_results = {}
        if request.scenarios:
            scenario_results = scenario_simulator.scenario_analysis(
                base_volatility=request.volatility,
                scenarios=request.scenarios,
            )
        else:
            # Default scenarios
            scenario_results = scenario_simulator.scenario_analysis(
                base_volatility=request.volatility,
                scenarios={
                    "bull": {"vol_multiplier": 0.8, "drift": 0.1},
                    "base": {"vol_multiplier": 1.0, "drift": 0.0},
                    "bear": {"vol_multiplier": 1.5, "drift": -0.1},
                },
            )
        
        return ScenarioResponse(
            symbol=request.symbol,
            current_price=request.current_price,
            scenarios=scenario_results,
            monte_carlo_results=mc_results,
            timestamp=datetime.now().isoformat(),
        )
    
    except Exception as e:
        logger.error("Scenario simulation failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Scenario simulation failed: {str(e)}",
        )

