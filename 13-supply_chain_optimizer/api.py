"""
📦 Supply Chain Optimizer - FastAPI Backend

REST API for supply chain optimization operations. 
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, List
from pathlib import Path
from datetime import datetime
import asyncio
import json
import uuid


app = FastAPI(
    title="Supply Chain Optimizer API",
    description="📦 Hierarchical Multi-Agent Supply Chain Optimization",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Models
# =============================================================================

class OptimizationRequest(BaseModel):
    service_level_a: float = 0.98
    service_level_b: float = 0.95
    service_level_c: float = 0.90
    forecast_periods: int = 3


class ScenarioRequest(BaseModel):
    scenario_type: str  # "demand_shock" or "supply_disruption"
    demand_change_pct: Optional[float] = None
    supplier_id: Optional[str] = None
    disruption_days: Optional[int] = None


# =============================================================================
# State Management
# =============================================================================

class AppState:
    def __init__(self):
        self.optimization_jobs: Dict[str, dict] = {}
        self.latest_result: Optional[dict] = None


state = AppState()


# =============================================================================
# Routes
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend."""
    html_path = Path(__file__).parent / "frontend" / "index.html"
    if html_path.exists():
        return html_path.read_text()
    return """
    <html>
        <head><title>Supply Chain Optimizer</title></head>
        <body style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; font-family: sans-serif; padding: 40px; text-align: center;">
            <h1>📦 Supply Chain Optimizer</h1>
            <p>Visit <a href="/docs" style="color: #4ecdc4;">/docs</a> for API documentation</p>
        </body>
    </html>
    """


@app.get("/api/status")
async def get_status():
    """Get API status."""
    return {
        "status": "ready",
        "version": "1.0.0",
        "jobs_running": len([j for j in state.optimization_jobs.values() if j.get("status") == "running"]),
    }


@app.get("/api/products")
async def get_products():
    """Get all products."""
    from supply_chain.data import PRODUCTS
    
    return {
        "products": [
            {
                "sku": p.sku,
                "name": p.name,
                "category": p.category.value,
                "unit_cost": p.unit_cost,
                "lead_time": p.lead_time_days,
                "abc_class": p.abc_class,
            }
            for p in PRODUCTS.values()
        ]
    }


@app.get("/api/suppliers")
async def get_suppliers():
    """Get all suppliers with scores."""
    from supply_chain.data import SUPPLIERS
    
    return {
        "suppliers": [
            {
                "supplier_id": s.supplier_id,
                "name": s.name,
                "country": s.country,
                "tier": s.tier.value,
                "overall_score": round(s.overall_score, 1),
                "quality_score": s.quality_score,
                "delivery_score": s.delivery_score,
                "cost_score": s.cost_score,
                "risk": s.risk_indicator,
            }
            for s in SUPPLIERS.values()
        ]
    }


@app.get("/api/inventory")
async def get_inventory():
    """Get current inventory levels."""
    from supply_chain.data import generate_inventory_levels, PRODUCTS
    
    inventory = generate_inventory_levels()
    
    return {
        "inventory": [
            {
                "sku": inv.sku,
                "product_name": PRODUCTS[inv.sku].name if inv.sku in PRODUCTS else "Unknown",
                "on_hand": inv.on_hand,
                "on_order": inv.on_order,
                "safety_stock": inv.safety_stock,
                "reorder_point": inv.reorder_point,
                "available": inv.available,
                "status": inv.stock_status,
            }
            for inv in inventory.values()
        ]
    }


@app.get("/api/kpis")
async def get_kpis():
    """Get supply chain KPIs."""
    from supply_chain.data import generate_inventory_levels, PRODUCTS, SUPPLIERS
    
    inventory = generate_inventory_levels()
    
    # Calculate KPIs
    total_value = sum(
        inv.on_hand * PRODUCTS[inv.sku].unit_cost 
        for inv in inventory.values() 
        if inv.sku in PRODUCTS
    )
    
    stockout_items = sum(1 for inv in inventory.values() if inv.stock_status == "STOCKOUT")
    critical_items = sum(1 for inv in inventory.values() if inv.stock_status == "CRITICAL")
    
    avg_supplier_score = sum(s.overall_score for s in SUPPLIERS.values()) / len(SUPPLIERS)
    high_risk_suppliers = sum(1 for s in SUPPLIERS.values() if s.risk_indicator == "🔴")
    
    return {
        "kpis": {
            "inventory": {
                "total_value": total_value,
                "total_skus": len(inventory),
                "stockout_count": stockout_items,
                "critical_count": critical_items,
                "healthy_pct": (len(inventory) - stockout_items - critical_items) / len(inventory) * 100,
            },
            "suppliers": {
                "total_count": len(SUPPLIERS),
                "avg_score": round(avg_supplier_score, 1),
                "high_risk_count": high_risk_suppliers,
            },
            "performance": {
                "fill_rate": 94.5,  # Simulated
                "otif_rate": 91.2,  # Simulated
                "forecast_accuracy": 87.3,  # Simulated
            },
        }
    }


@app.post("/api/optimize")
async def start_optimization(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Start optimization in background."""
    from supply_chain import SupplyChainOptimizer, OptimizationConfig
    
    job_id = f"job-{uuid.uuid4().hex[:8]}"
    
    state.optimization_jobs[job_id] = {
        "status": "running",
        "started_at": datetime.now().isoformat(),
        "result": None,
    }
    
    def run_optimization():
        try:
            config = OptimizationConfig(
                target_service_level_a=request.service_level_a,
                target_service_level_b=request.service_level_b,
                target_service_level_c=request.service_level_c,
                forecast_periods=request.forecast_periods,
                verbose=False,
            )
            
            optimizer = SupplyChainOptimizer(config)
            result = optimizer.optimize()
            
            state.optimization_jobs[job_id] = {
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                "result": {
                    "optimization_id": result.optimization_id,
                    "current_inventory_value": result.current_inventory_value,
                    "recommended_inventory_value": result.recommended_inventory_value,
                    "inventory_reduction_pct": result.inventory_reduction_pct,
                    "working_capital_freed": result.working_capital_freed,
                    "stockout_risk_current": result.current_stockout_risk,
                    "stockout_risk_recommended": result.recommended_stockout_risk,
                    "routes_before": result.routes_before,
                    "routes_after": result.routes_after,
                    "shipping_cost_savings": result.shipping_cost_savings,
                    "carbon_reduction_pct": result.carbon_reduction_pct,
                    "recommendations_count": len(result.replenishment_recommendations),
                    "alerts": result.alerts,
                },
            }
            state.latest_result = state.optimization_jobs[job_id]["result"]
            
        except Exception as e:
            state.optimization_jobs[job_id] = {
                "status": "failed",
                "error": str(e),
            }
    
    background_tasks.add_task(run_optimization)
    
    return {"job_id": job_id, "status": "started"}


@app.get("/api/optimize/{job_id}")
async def get_optimization_status(job_id: str):
    """Get optimization job status."""
    if job_id not in state.optimization_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return state.optimization_jobs[job_id]


@app.post("/api/scenario")
async def run_scenario(request: ScenarioRequest):
    """Run what-if scenario analysis."""
    from supply_chain import (
        SupplyChainOptimizer, OptimizationConfig,
        create_demand_shock_scenario, create_supply_disruption_scenario
    )
    
    config = OptimizationConfig(verbose=False)
    optimizer = SupplyChainOptimizer(config)
    
    # Run demand planning to get forecasts
    optimizer.run_demand_planning()
    
    # Create scenario
    if request.scenario_type == "demand_shock" and request.demand_change_pct:
        scenario = create_demand_shock_scenario(request.demand_change_pct)
    elif request.scenario_type == "supply_disruption" and request.supplier_id:
        scenario = create_supply_disruption_scenario(
            request.supplier_id,
            request.disruption_days or 30
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid scenario parameters")
    
    results = optimizer.run_scenario(scenario)
    
    return {
        "scenario": {
            "name": scenario.name,
            "description": scenario.description,
        },
        "results": results,
    }


@app.get("/api/hierarchy")
async def get_agent_hierarchy():
    """Get agent hierarchy structure."""
    from supply_chain.agents import get_agent_hierarchy
    return get_agent_hierarchy()


@app.get("/api/recommendations")
async def get_recommendations():
    """Get latest replenishment recommendations."""
    if not state.latest_result:
        return {"recommendations": [], "message": "No optimization run yet"}
    
    # Return sample recommendations
    from supply_chain import SupplyChainOptimizer, OptimizationConfig
    
    config = OptimizationConfig(verbose=False)
    optimizer = SupplyChainOptimizer(config)
    
    demand_plan = optimizer.run_demand_planning()
    inventory_plan = optimizer.run_inventory_management(demand_plan)
    
    recs = inventory_plan.get("replenishment_recommendations", [])
    
    return {
        "recommendations": [
            {
                "sku": r.sku,
                "product_name": r.product_name,
                "current_stock": r.current_stock,
                "forecast_demand": r.forecast_demand,
                "recommended_qty": r.recommended_order_qty,
                "supplier": r.recommended_supplier,
                "estimated_cost": r.estimated_cost,
                "priority": r.priority,
            }
            for r in recs[:20]
        ]
    }


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
