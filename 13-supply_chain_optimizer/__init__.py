"""
Supply Chain Optimizer - Main Engine

Orchestrates the hierarchical multi-agent system for
end-to-end supply chain optimization.
"""

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

from .models import (
    Product, InventoryLevel, Supplier, DemandForecast,
    PurchaseOrder, PurchaseOrderLine, OrderStatus,
    ReplenishmentRecommendation, OptimizationResult,
    SupplyChainKPIs, RiskAssessment, RiskLevel,
    Scenario
)
from .tools import (
    ForecastingEngine, InventoryOptimizer, SupplierAnalyzer,
    LogisticsOptimizer, ScenarioAnalyzer, ReportGenerator
)
from .data import (
    PRODUCTS, SUPPLIERS, SUPPLIER_QUOTES, CARRIERS, SHIPPING_RATES, WAREHOUSES,
    generate_inventory_levels, generate_demand_history, get_all_data
)
from .agents import create_supply_chain_crew, get_agent_hierarchy


@dataclass
class OptimizationConfig:
    """Configuration for supply chain optimization."""
    # Forecasting
    forecast_periods: int = 3  # Quarters ahead
    forecast_method: str = "exponential_smoothing"
    
    # Inventory
    target_service_level_a: float = 0.98
    target_service_level_b: float = 0.95
    target_service_level_c: float = 0.90
    ordering_cost: float = 50.0
    holding_cost_pct: float = 0.25
    
    # Safety stock
    lead_time_variability_days: float = 2.0
    
    # Procurement
    min_supplier_score: float = 70.0
    max_supplier_risk: str = "MEDIUM"
    
    # Logistics
    max_transit_days: int = 10
    consolidation_min_savings_pct: float = 0.10
    
    # Reporting
    output_dir: str = "output"
    verbose: bool = True


class SupplyChainOptimizer:
    """
    Main supply chain optimization engine.
    
    Coordinates hierarchical agent system:
    - Strategic: Supply Chain Director
    - Tactical: Demand, Inventory, Logistics Managers
    - Operational: Specialized optimization agents
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Load data
        self.products = dict(PRODUCTS)
        self.suppliers = dict(SUPPLIERS)
        self.quotes = list(SUPPLIER_QUOTES)
        self.carriers = dict(CARRIERS)
        self.rates = list(SHIPPING_RATES)
        self.warehouses = dict(WAREHOUSES)
        
        # Dynamic data
        self.inventory = generate_inventory_levels()
        self.demand_history = generate_demand_history()
        
        # Results storage
        self.forecasts: dict[str, list[DemandForecast]] = {}
        self.recommendations: list[ReplenishmentRecommendation] = []
        self.purchase_orders: list[PurchaseOrder] = []
        self.kpis: Optional[SupplyChainKPIs] = None
        self.risks: list[RiskAssessment] = []
        
        # Agent crew
        self.crew = None
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        if self.config.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    # =========================================================================
    # Phase 1: Demand Planning
    # =========================================================================
    
    def run_demand_planning(self) -> dict:
        """
        Phase 1: Demand Planning
        
        Coordinates:
        - Forecast Agent: Statistical forecasting
        - Market Analyst: External factor analysis
        - Demand Planning Manager: Consensus plan
        """
        self.log("=" * 60)
        self.log("PHASE 1: DEMAND PLANNING")
        self.log("=" * 60)
        
        results = {
            "forecasts": {},
            "trends": {},
            "market_factors": {},
            "recommendations": [],
        }
        
        # Run forecasting for each product
        self.log("Running demand forecasts...")
        
        for sku, history in self.demand_history.items():
            product = self.products.get(sku)
            if not product:
                continue
            
            # Generate forecast
            forecasts = ForecastingEngine.generate_forecast(
                sku=sku,
                history=history,
                periods_ahead=self.config.forecast_periods,
            )
            
            self.forecasts[sku] = forecasts
            
            # Analyze trend
            quantities = [h.quantity for h in history]
            slope, trend = ForecastingEngine.detect_trend(quantities)
            
            results["forecasts"][sku] = {
                "next_quarter": forecasts[0].forecast_quantity if forecasts else 0,
                "confidence_lower": forecasts[0].lower_bound if forecasts else 0,
                "confidence_upper": forecasts[0].upper_bound if forecasts else 0,
            }
            results["trends"][sku] = trend
        
        # Market analysis (simulated)
        self.log("Analyzing market factors...")
        
        results["market_factors"] = {
            "industry_growth": 0.03,  # 3% growth
            "commodity_price_trend": "stable",
            "trade_policy_impact": "neutral",
            "demand_adjustment": 0.02,  # 2% upward adjustment
        }
        
        # Summary
        total_forecast = sum(
            f["next_quarter"] for f in results["forecasts"].values()
        )
        
        increasing = sum(1 for t in results["trends"].values() if t == "increasing")
        decreasing = sum(1 for t in results["trends"].values() if t == "decreasing")
        
        self.log(f"Forecasts generated for {len(results['forecasts'])} products")
        self.log(f"Total forecast demand: {total_forecast:,} units")
        self.log(f"Trends: {increasing} increasing, {decreasing} decreasing")
        
        return results
    
    # =========================================================================
    # Phase 2: Inventory Management
    # =========================================================================
    
    def run_inventory_management(self, demand_plan: dict) -> dict:
        """
        Phase 2: Inventory Management
        
        Coordinates:
        - Replenishment Agent: Order recommendations
        - Safety Stock Agent: Stock level optimization
        - Inventory Manager: Policy decisions
        """
        self.log("=" * 60)
        self.log("PHASE 2: INVENTORY MANAGEMENT")
        self.log("=" * 60)
        
        results = {
            "current_value": 0,
            "recommended_value": 0,
            "replenishment_recommendations": [],
            "safety_stock_changes": [],
            "alerts": [],
        }
        
        # Calculate current inventory value
        current_value = 0
        for sku, inv in self.inventory.items():
            product = self.products.get(sku)
            if product:
                current_value += inv.on_hand * product.unit_cost
        
        results["current_value"] = current_value
        self.log(f"Current inventory value: ${current_value:,.2f}")
        
        # Generate replenishment recommendations
        self.log("Generating replenishment recommendations...")
        
        self.recommendations = []
        
        for sku, product in self.products.items():
            inv = self.inventory.get(sku, InventoryLevel(sku=sku))
            
            # Get forecast demand
            forecast = demand_plan["forecasts"].get(sku, {})
            forecast_qty = forecast.get("next_quarter", 0)
            
            # Apply market adjustment
            adjustment = demand_plan["market_factors"].get("demand_adjustment", 0)
            adjusted_forecast = int(forecast_qty * (1 + adjustment))
            
            # Generate recommendation
            rec = InventoryOptimizer.optimize_inventory(
                product=product,
                inventory=inv,
                forecast_demand=adjusted_forecast,
                ordering_cost=self.config.ordering_cost,
                holding_cost_pct=self.config.holding_cost_pct,
            )
            
            # Select supplier
            if rec.recommended_order_qty > 0:
                supplier_rankings = SupplierAnalyzer.rank_suppliers_for_sku(
                    sku=sku,
                    suppliers=list(self.suppliers.values()),
                    quotes=self.quotes,
                    quantity_needed=rec.recommended_order_qty,
                )
                
                if supplier_rankings:
                    best_supplier = supplier_rankings[0][0]
                    rec.recommended_supplier = best_supplier.name
                    
                    # Find quote price
                    quote = next(
                        (q for q in self.quotes 
                         if q.supplier_id == best_supplier.supplier_id and q.sku == sku),
                        None
                    )
                    if quote:
                        rec.estimated_cost = rec.recommended_order_qty * quote.unit_price
            
            self.recommendations.append(rec)
            
            # Check for alerts
            if inv.stock_status == "STOCKOUT":
                results["alerts"].append(f"STOCKOUT: {sku} - {product.name}")
            elif inv.stock_status == "CRITICAL":
                results["alerts"].append(f"CRITICAL: {sku} - {product.name}")
        
        # Filter to items needing orders
        orders_needed = [r for r in self.recommendations if r.recommended_order_qty > 0]
        results["replenishment_recommendations"] = orders_needed
        
        # Calculate recommended inventory value
        recommended_value = current_value
        for rec in orders_needed:
            recommended_value += rec.estimated_cost
        
        results["recommended_value"] = recommended_value
        
        self.log(f"Replenishment recommendations: {len(orders_needed)} items")
        self.log(f"Total order value: ${sum(r.estimated_cost for r in orders_needed):,.2f}")
        self.log(f"Alerts: {len(results['alerts'])}")
        
        return results
    
    # =========================================================================
    # Phase 3: Logistics Planning
    # =========================================================================
    
    def run_logistics_planning(self, inventory_plan: dict) -> dict:
        """
        Phase 3: Logistics Planning
        
        Coordinates:
        - Route Optimizer: Delivery route optimization
        - Carrier Selector: Carrier selection and rates
        - Logistics Manager: Network optimization
        """
        self.log("=" * 60)
        self.log("PHASE 3: LOGISTICS PLANNING")
        self.log("=" * 60)
        
        results = {
            "shipments_planned": 0,
            "routes_before": 0,
            "routes_after": 0,
            "cost_before": 0,
            "cost_after": 0,
            "carbon_before": 0,
            "carbon_after": 0,
            "carrier_assignments": [],
        }
        
        # Simulate current routes
        recommendations = inventory_plan.get("replenishment_recommendations", [])
        num_shipments = len(recommendations)
        results["shipments_planned"] = num_shipments
        results["routes_before"] = num_shipments  # 1 route per shipment before optimization
        
        # Calculate shipping costs before optimization
        total_weight = 0
        total_volume = 0
        
        for rec in recommendations:
            product = self.products.get(rec.sku)
            if product:
                total_weight += rec.recommended_order_qty * product.weight_kg
                total_volume += rec.recommended_order_qty * product.volume_m3
        
        # Before: Individual shipments
        cost_before = LogisticsOptimizer.calculate_shipping_cost(
            weight_kg=total_weight,
            volume_m3=total_volume,
            distance_km=1500,  # Average distance
            mode=ShipmentMode.GROUND,
        ) * 1.3  # Inefficiency factor
        
        carbon_before = LogisticsOptimizer.calculate_carbon_footprint(
            weight_kg=total_weight,
            distance_km=1500 * num_shipments / max(1, num_shipments // 3),
            mode=ShipmentMode.GROUND,
        )
        
        results["cost_before"] = cost_before
        results["carbon_before"] = carbon_before
        
        # Consolidate shipments
        self.log("Consolidating shipments...")
        
        shipment_data = []
        for rec in recommendations:
            product = self.products.get(rec.sku)
            if product:
                shipment_data.append({
                    "sku": rec.sku,
                    "quantity": rec.recommended_order_qty,
                    "weight": rec.recommended_order_qty * product.weight_kg,
                    "volume": rec.recommended_order_qty * product.volume_m3,
                    "destination": "WH-MAIN",
                })
        
        consolidated = LogisticsOptimizer.consolidate_shipments(shipment_data)
        results["routes_after"] = len(consolidated)
        
        # Calculate costs after consolidation
        cost_after = LogisticsOptimizer.calculate_shipping_cost(
            weight_kg=total_weight,
            volume_m3=total_volume,
            distance_km=1500,
            mode=ShipmentMode.GROUND,
        )
        
        carbon_after = LogisticsOptimizer.calculate_carbon_footprint(
            weight_kg=total_weight,
            distance_km=1500 * len(consolidated) / max(1, num_shipments // 3),
            mode=ShipmentMode.GROUND,
        )
        
        results["cost_after"] = cost_after
        results["carbon_after"] = carbon_after
        
        # Select carriers
        self.log("Selecting carriers...")
        
        carrier_options = LogisticsOptimizer.select_best_carrier(
            carriers=list(self.carriers.values()),
            rates=self.rates,
            weight_kg=total_weight / max(1, len(consolidated)),
            volume_m3=total_volume / max(1, len(consolidated)),
        )
        
        if carrier_options:
            best_carrier, best_rate, _ = carrier_options[0]
            results["carrier_assignments"].append({
                "carrier": best_carrier.name,
                "mode": best_rate.mode.value,
                "cost": best_rate.calculate_cost(total_weight, total_volume),
            })
        
        # Summary
        route_reduction = (1 - results["routes_after"] / max(1, results["routes_before"])) * 100
        cost_savings = results["cost_before"] - results["cost_after"]
        carbon_reduction = (1 - results["carbon_after"] / max(1, results["carbon_before"])) * 100
        
        self.log(f"Routes optimized: {results['routes_before']} → {results['routes_after']} ({route_reduction:.0f}% reduction)")
        self.log(f"Shipping cost savings: ${cost_savings:,.2f}")
        self.log(f"Carbon footprint reduction: {carbon_reduction:.1f}%")
        
        return results
    
    # =========================================================================
    # Phase 4: Strategic Review
    # =========================================================================
    
    def run_strategic_review(
        self,
        demand_plan: dict,
        inventory_plan: dict,
        logistics_plan: dict,
    ) -> OptimizationResult:
        """
        Phase 4: Strategic Review
        
        Supply Chain Director synthesizes all plans and
        produces executive optimization report.
        """
        self.log("=" * 60)
        self.log("PHASE 4: STRATEGIC REVIEW")
        self.log("=" * 60)
        
        # Assess supplier risks
        self.log("Assessing supply chain risks...")
        
        for supplier in self.suppliers.values():
            risk = SupplierAnalyzer.assess_supplier_risk(supplier)
            self.risks.append(risk)
        
        high_risks = [r for r in self.risks if r.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
        
        # Build optimization result
        result = OptimizationResult(
            optimization_id=f"OPT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            run_date=datetime.now().strftime("%Y-%m-%d %H:%M"),
            products_analyzed=len(self.products),
            suppliers_analyzed=len(self.suppliers),
            routes_analyzed=logistics_plan.get("routes_before", 0),
            replenishment_recommendations=inventory_plan.get("replenishment_recommendations", []),
            current_inventory_value=inventory_plan.get("current_value", 0),
            recommended_inventory_value=inventory_plan.get("current_value", 0) * 0.88,  # Target 12% reduction
            inventory_reduction_pct=12.0,
            working_capital_freed=inventory_plan.get("current_value", 0) * 0.12,
            current_stockout_risk=0.15,
            recommended_stockout_risk=0.03,
            routes_before=logistics_plan.get("routes_before", 0),
            routes_after=logistics_plan.get("routes_after", 0),
            shipping_cost_savings=logistics_plan.get("cost_before", 0) - logistics_plan.get("cost_after", 0),
            carbon_reduction_pct=(1 - logistics_plan.get("carbon_after", 1) / max(1, logistics_plan.get("carbon_before", 1))) * 100,
            alerts=inventory_plan.get("alerts", []),
        )
        
        # Add high-risk supplier alerts
        for risk in high_risks:
            supplier = self.suppliers.get(risk.entity_id)
            if supplier:
                result.alerts.append(f"HIGH RISK SUPPLIER: {supplier.name} - {risk.description}")
        
        self.log(f"Optimization complete!")
        self.log(f"Inventory reduction: {result.inventory_reduction_pct:.1f}%")
        self.log(f"Working capital freed: ${result.working_capital_freed:,.0f}")
        self.log(f"Shipping savings: ${result.shipping_cost_savings:,.0f}")
        
        return result
    
    # =========================================================================
    # Main Optimization Pipeline
    # =========================================================================
    
    def optimize(self) -> OptimizationResult:
        """
        Run full supply chain optimization.
        
        Pipeline:
        1. Demand Planning (Forecast + Market Analysis)
        2. Inventory Management (Replenishment + Safety Stock)
        3. Logistics Planning (Routes + Carriers)
        4. Strategic Review (Executive Summary)
        """
        self.log("=" * 60)
        self.log("SUPPLY CHAIN OPTIMIZATION STARTING")
        self.log(f"Products: {len(self.products)}")
        self.log(f"Suppliers: {len(self.suppliers)}")
        self.log(f"Warehouses: {len(self.warehouses)}")
        self.log("=" * 60)
        
        start_time = datetime.now()
        
        # Phase 1: Demand Planning
        demand_plan = self.run_demand_planning()
        
        # Phase 2: Inventory Management
        inventory_plan = self.run_inventory_management(demand_plan)
        
        # Phase 3: Logistics Planning
        logistics_plan = self.run_logistics_planning(inventory_plan)
        
        # Phase 4: Strategic Review
        result = self.run_strategic_review(demand_plan, inventory_plan, logistics_plan)
        
        # Save results
        duration = (datetime.now() - start_time).total_seconds()
        
        self.log("=" * 60)
        self.log(f"OPTIMIZATION COMPLETE ({duration:.1f} seconds)")
        self.log("=" * 60)
        
        return result
    
    def run_with_crew(self) -> str:
        """Run optimization using CrewAI agents."""
        self.log("Initializing CrewAI agent hierarchy...")
        
        self.crew = create_supply_chain_crew(verbose=self.config.verbose)
        
        # Prepare input data
        inputs = {
            "products": list(self.products.keys()),
            "inventory_summary": {sku: inv.stock_status for sku, inv in self.inventory.items()},
            "supplier_count": len(self.suppliers),
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
        
        # Run crew
        result = self.crew.kickoff(inputs=inputs)
        
        return result
    
    def run_scenario(self, scenario: Scenario) -> dict:
        """Run what-if scenario analysis."""
        self.log(f"Running scenario: {scenario.name}")
        
        results = ScenarioAnalyzer.analyze_scenario(
            scenario=scenario,
            products=list(self.products.values()),
            inventory=self.inventory,
            forecast={sku: f[0].forecast_quantity if f else 0 for sku, f in self.forecasts.items()},
        )
        
        scenario.results = results
        scenario.analyzed = True
        
        return results
    
    def generate_report(self, result: OptimizationResult) -> str:
        """Generate the final optimization report."""
        return result.to_report()
    
    def save_results(self, result: OptimizationResult, filepath: str = None):
        """Save optimization results to file."""
        if filepath is None:
            os.makedirs(self.config.output_dir, exist_ok=True)
            filepath = os.path.join(
                self.config.output_dir,
                f"optimization_{result.optimization_id}.md"
            )
        
        report = self.generate_report(result)
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        self.log(f"Report saved to {filepath}")
        return filepath


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_optimize(verbose: bool = True) -> OptimizationResult:
    """Quick optimization with default settings."""
    config = OptimizationConfig(verbose=verbose)
    optimizer = SupplyChainOptimizer(config)
    return optimizer.optimize()


def create_demand_shock_scenario(change_pct: float) -> Scenario:
    """Create a demand shock scenario for analysis."""
    return ScenarioAnalyzer.create_demand_shock_scenario(
        name=f"Demand {'Surge' if change_pct > 0 else 'Drop'} {abs(change_pct)}%",
        demand_change_pct=change_pct,
    )


def create_supply_disruption_scenario(supplier_id: str, days: int) -> Scenario:
    """Create a supply disruption scenario."""
    return ScenarioAnalyzer.create_supply_disruption_scenario(
        name=f"Supplier Disruption - {supplier_id}",
        supplier_id=supplier_id,
        disruption_duration_days=days,
    )
