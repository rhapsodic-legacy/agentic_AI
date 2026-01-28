"""
Supply Chain Optimizer - Tools

Tools for:
- Demand forecasting
- Inventory optimization (EOQ, safety stock)
- Supplier scoring
- Route optimization
- Risk assessment
- Scenario analysis
"""

from typing import Optional
from datetime import datetime, timedelta
import math
import random
import statistics

from .models import (
    Product, InventoryLevel, Supplier, SupplierQuote,
    DemandHistory, DemandForecast, SeasonalityPattern,
    PurchaseOrder, PurchaseOrderLine, OrderStatus,
    Carrier, ShippingRate, Route, ShipmentMode,
    RiskAssessment, RiskLevel, Scenario,
    ReplenishmentRecommendation, OptimizationResult,
    ForecastMethod, SupplierTier
)


# =============================================================================
# Demand Forecasting Tools
# =============================================================================

class ForecastingEngine:
    """Demand forecasting algorithms."""
    
    @staticmethod
    def moving_average(history: list[int], periods: int = 3) -> int:
        """Simple moving average forecast."""
        if len(history) < periods:
            return int(sum(history) / len(history)) if history else 0
        return int(sum(history[-periods:]) / periods)
    
    @staticmethod
    def exponential_smoothing(history: list[int], alpha: float = 0.3) -> int:
        """Single exponential smoothing forecast."""
        if not history:
            return 0
        
        forecast = history[0]
        for actual in history[1:]:
            forecast = alpha * actual + (1 - alpha) * forecast
        
        return int(forecast)
    
    @staticmethod
    def weighted_moving_average(history: list[int], weights: list[float] = None) -> int:
        """Weighted moving average with more recent periods weighted higher."""
        if not history:
            return 0
        
        n = min(len(history), 6)
        recent = history[-n:]
        
        if weights is None:
            # Generate decreasing weights
            weights = [i + 1 for i in range(n)]
        
        total_weight = sum(weights[:n])
        weighted_sum = sum(recent[i] * weights[i] for i in range(n))
        
        return int(weighted_sum / total_weight)
    
    @staticmethod
    def seasonal_forecast(
        history: list[int],
        seasonality: dict[int, float],  # {period_index: seasonal_factor}
        base_forecast: int,
        target_period: int,
    ) -> int:
        """Apply seasonal adjustment to base forecast."""
        seasonal_factor = seasonality.get(target_period, 1.0)
        return int(base_forecast * seasonal_factor)
    
    @staticmethod
    def calculate_forecast_error(actuals: list[int], forecasts: list[int]) -> dict:
        """Calculate forecast accuracy metrics."""
        if len(actuals) != len(forecasts) or not actuals:
            return {}
        
        errors = [abs(a - f) for a, f in zip(actuals, forecasts)]
        pct_errors = [e / a if a > 0 else 0 for e, a in zip(errors, actuals)]
        
        return {
            "mae": statistics.mean(errors),  # Mean Absolute Error
            "mape": statistics.mean(pct_errors) * 100,  # Mean Absolute Percentage Error
            "rmse": math.sqrt(statistics.mean([e**2 for e in errors])),  # Root Mean Square Error
            "bias": statistics.mean([f - a for a, f in zip(actuals, forecasts)]),
        }
    
    @staticmethod
    def detect_trend(history: list[int]) -> tuple[float, str]:
        """Detect trend in historical data."""
        if len(history) < 3:
            return 0, "insufficient_data"
        
        # Simple linear regression slope
        n = len(history)
        x_mean = (n - 1) / 2
        y_mean = statistics.mean(history)
        
        numerator = sum((i - x_mean) * (history[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0, "flat"
        
        slope = numerator / denominator
        
        # Classify trend
        pct_change = slope / y_mean if y_mean > 0 else 0
        
        if pct_change > 0.05:
            return slope, "increasing"
        elif pct_change < -0.05:
            return slope, "decreasing"
        else:
            return slope, "stable"
    
    @staticmethod
    def generate_forecast(
        sku: str,
        history: list[DemandHistory],
        periods_ahead: int = 3,
        method: ForecastMethod = ForecastMethod.EXPONENTIAL_SMOOTHING,
    ) -> list[DemandForecast]:
        """Generate demand forecasts for multiple periods."""
        
        quantities = [h.quantity for h in history]
        
        if not quantities:
            return []
        
        # Calculate base forecast
        if method == ForecastMethod.MOVING_AVERAGE:
            base = ForecastingEngine.moving_average(quantities)
        elif method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            base = ForecastingEngine.exponential_smoothing(quantities)
        else:
            base = ForecastingEngine.weighted_moving_average(quantities)
        
        # Detect trend
        slope, trend = ForecastingEngine.detect_trend(quantities)
        
        # Calculate variability for confidence intervals
        if len(quantities) > 1:
            std_dev = statistics.stdev(quantities)
        else:
            std_dev = base * 0.2  # Assume 20% variability
        
        forecasts = []
        for i in range(periods_ahead):
            # Apply trend
            adjusted_forecast = int(base + slope * (i + 1))
            adjusted_forecast = max(0, adjusted_forecast)
            
            # Calculate confidence intervals (95%)
            margin = 1.96 * std_dev
            lower = max(0, int(adjusted_forecast - margin))
            upper = int(adjusted_forecast + margin)
            
            forecast = DemandForecast(
                forecast_id=f"FC-{sku}-{i+1}",
                sku=sku,
                period=f"Period+{i+1}",
                forecast_quantity=adjusted_forecast,
                lower_bound=lower,
                upper_bound=upper,
                confidence_level=0.95,
                method=method,
                created_at=datetime.now().isoformat(),
            )
            forecasts.append(forecast)
        
        return forecasts


# =============================================================================
# Inventory Optimization Tools
# =============================================================================

class InventoryOptimizer:
    """Inventory optimization calculations."""
    
    @staticmethod
    def calculate_eoq(
        annual_demand: int,
        ordering_cost: float,
        holding_cost_per_unit: float,
    ) -> int:
        """
        Calculate Economic Order Quantity.
        
        EOQ = sqrt(2 * D * S / H)
        D = Annual demand
        S = Ordering cost per order
        H = Holding cost per unit per year
        """
        if holding_cost_per_unit <= 0 or annual_demand <= 0:
            return 0
        
        eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_per_unit)
        return int(round(eoq))
    
    @staticmethod
    def calculate_safety_stock(
        avg_daily_demand: float,
        demand_std_dev: float,
        lead_time_days: int,
        lead_time_std_dev: float,
        service_level: float = 0.95,
    ) -> int:
        """
        Calculate safety stock using statistical method.
        
        SS = Z * sqrt(LT * σd² + d² * σLT²)
        Z = Service level factor
        LT = Lead time
        σd = Demand standard deviation
        d = Average demand
        σLT = Lead time standard deviation
        """
        # Z-score for service level
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.97: 1.88,
            0.99: 2.33,
        }
        z = z_scores.get(service_level, 1.65)
        
        # Combined variability
        demand_variance = lead_time_days * (demand_std_dev ** 2)
        lead_time_variance = (avg_daily_demand ** 2) * (lead_time_std_dev ** 2)
        
        safety_stock = z * math.sqrt(demand_variance + lead_time_variance)
        return int(math.ceil(safety_stock))
    
    @staticmethod
    def calculate_reorder_point(
        avg_daily_demand: float,
        lead_time_days: int,
        safety_stock: int,
    ) -> int:
        """
        Calculate reorder point.
        
        ROP = (Average Daily Demand × Lead Time) + Safety Stock
        """
        return int(math.ceil(avg_daily_demand * lead_time_days + safety_stock))
    
    @staticmethod
    def calculate_inventory_value(
        inventory_levels: list[tuple[int, float]],  # [(quantity, unit_cost)]
    ) -> float:
        """Calculate total inventory value."""
        return sum(qty * cost for qty, cost in inventory_levels)
    
    @staticmethod
    def calculate_days_of_supply(
        current_stock: int,
        avg_daily_demand: float,
    ) -> float:
        """Calculate days of supply."""
        if avg_daily_demand <= 0:
            return float('inf')
        return current_stock / avg_daily_demand
    
    @staticmethod
    def calculate_inventory_turnover(
        cost_of_goods_sold: float,
        average_inventory_value: float,
    ) -> float:
        """Calculate inventory turnover ratio."""
        if average_inventory_value <= 0:
            return 0
        return cost_of_goods_sold / average_inventory_value
    
    @staticmethod
    def abc_classification(
        products: list[tuple[str, float]],  # [(sku, annual_value)]
    ) -> dict[str, str]:
        """
        Perform ABC classification.
        
        A: Top 80% of value (typically 20% of items)
        B: Next 15% of value (typically 30% of items)
        C: Bottom 5% of value (typically 50% of items)
        """
        if not products:
            return {}
        
        # Sort by value descending
        sorted_products = sorted(products, key=lambda x: x[1], reverse=True)
        total_value = sum(p[1] for p in sorted_products)
        
        if total_value == 0:
            return {sku: "C" for sku, _ in products}
        
        classifications = {}
        cumulative_value = 0
        
        for sku, value in sorted_products:
            cumulative_value += value
            cumulative_pct = cumulative_value / total_value
            
            if cumulative_pct <= 0.80:
                classifications[sku] = "A"
            elif cumulative_pct <= 0.95:
                classifications[sku] = "B"
            else:
                classifications[sku] = "C"
        
        return classifications
    
    @staticmethod
    def optimize_inventory(
        product: Product,
        inventory: InventoryLevel,
        forecast_demand: int,
        ordering_cost: float = 50.0,
        holding_cost_pct: float = 0.25,
    ) -> ReplenishmentRecommendation:
        """Generate replenishment recommendation for a product."""
        
        # Calculate parameters
        annual_demand = forecast_demand * 4  # Quarterly to annual
        holding_cost = product.unit_cost * holding_cost_pct
        
        eoq = InventoryOptimizer.calculate_eoq(annual_demand, ordering_cost, holding_cost)
        
        # Daily demand
        daily_demand = forecast_demand / 90  # Quarter = 90 days
        demand_std = daily_demand * 0.3  # Assume 30% variability
        
        safety_stock = InventoryOptimizer.calculate_safety_stock(
            daily_demand, demand_std, product.lead_time_days, 2, product.service_level_target
        )
        
        reorder_point = InventoryOptimizer.calculate_reorder_point(
            daily_demand, product.lead_time_days, safety_stock
        )
        
        # Determine order quantity
        if inventory.projected <= reorder_point:
            # Need to order
            shortfall = forecast_demand - inventory.projected + safety_stock
            order_qty = max(eoq, shortfall)
            
            # Round to order multiple
            if product.order_multiple > 1:
                order_qty = math.ceil(order_qty / product.order_multiple) * product.order_multiple
            
            # Ensure minimum order quantity
            order_qty = max(order_qty, product.min_order_quantity)
            
            priority = "Critical" if inventory.stock_status == "STOCKOUT" else \
                       "High" if inventory.stock_status == "CRITICAL" else "Normal"
        else:
            order_qty = 0
            priority = "Low"
        
        # Calculate dates
        order_by = datetime.now() + timedelta(days=3)
        expected_receipt = order_by + timedelta(days=product.lead_time_days)
        
        return ReplenishmentRecommendation(
            sku=product.sku,
            product_name=product.name,
            current_stock=inventory.on_hand,
            safety_stock=safety_stock,
            reorder_point=reorder_point,
            forecast_demand=forecast_demand,
            forecast_period="Next Quarter",
            recommended_order_qty=order_qty,
            recommended_supplier="",  # Set by supplier selection
            estimated_cost=order_qty * product.unit_cost,
            order_by_date=order_by.strftime("%Y-%m-%d"),
            expected_receipt_date=expected_receipt.strftime("%Y-%m-%d"),
            priority=priority,
            reason=f"Projected inventory ({inventory.projected}) below ROP ({reorder_point})" if order_qty > 0 else "Inventory levels adequate",
        )


# =============================================================================
# Supplier Management Tools
# =============================================================================

class SupplierAnalyzer:
    """Supplier analysis and selection tools."""
    
    @staticmethod
    def calculate_supplier_score(
        quality_score: float,
        delivery_score: float,
        cost_score: float,
        responsiveness_score: float,
        weights: dict = None,
    ) -> float:
        """Calculate weighted supplier score."""
        if weights is None:
            weights = {
                "quality": 0.30,
                "delivery": 0.30,
                "cost": 0.25,
                "responsiveness": 0.15,
            }
        
        return (
            quality_score * weights["quality"] +
            delivery_score * weights["delivery"] +
            cost_score * weights["cost"] +
            responsiveness_score * weights["responsiveness"]
        )
    
    @staticmethod
    def calculate_total_cost_of_ownership(
        unit_price: float,
        shipping_cost: float,
        quality_cost: float,  # Cost of defects/returns
        lead_time_cost: float,  # Cost of carrying safety stock
        admin_cost: float,
    ) -> float:
        """Calculate total cost of ownership."""
        return unit_price + shipping_cost + quality_cost + lead_time_cost + admin_cost
    
    @staticmethod
    def rank_suppliers_for_sku(
        sku: str,
        suppliers: list[Supplier],
        quotes: list[SupplierQuote],
        quantity_needed: int,
    ) -> list[tuple[Supplier, float, str]]:
        """
        Rank suppliers for a specific SKU.
        
        Returns: [(supplier, score, reason)]
        """
        rankings = []
        
        for supplier in suppliers:
            # Find quote for this supplier
            quote = next(
                (q for q in quotes if q.supplier_id == supplier.supplier_id and q.sku == sku),
                None
            )
            
            if not quote:
                continue
            
            # Check quantity requirements
            if quote.min_quantity > quantity_needed:
                continue
            
            # Calculate adjusted score
            base_score = supplier.overall_score
            
            # Adjust for cost (normalize around 80 being average)
            cost_factor = 80 / (quote.total_landed_cost * 10) if quote.total_landed_cost > 0 else 1
            cost_adjustment = min(20, max(-20, (cost_factor - 1) * 20))
            
            # Adjust for lead time
            lead_time_factor = 1 - (supplier.standard_lead_time_days - 14) / 28
            lead_time_adjustment = lead_time_factor * 10
            
            # Adjust for risk
            risk_penalty = 0
            if supplier.financial_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                risk_penalty = 15
            elif supplier.financial_risk == RiskLevel.MEDIUM:
                risk_penalty = 5
            
            if supplier.geopolitical_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                risk_penalty += 10
            
            final_score = base_score + cost_adjustment + lead_time_adjustment - risk_penalty
            final_score = max(0, min(100, final_score))
            
            reason = f"Score: {supplier.overall_score:.0f}, Cost: ${quote.total_landed_cost:.2f}, Lead: {supplier.standard_lead_time_days}d"
            
            rankings.append((supplier, final_score, reason))
        
        # Sort by score descending
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    @staticmethod
    def assess_supplier_risk(supplier: Supplier) -> RiskAssessment:
        """Assess overall supplier risk."""
        
        # Calculate risk score
        risk_factors = []
        
        # Financial risk
        if supplier.financial_risk == RiskLevel.CRITICAL:
            risk_factors.append(("Financial instability", 0.9, 9))
        elif supplier.financial_risk == RiskLevel.HIGH:
            risk_factors.append(("Financial concerns", 0.7, 7))
        
        # Geopolitical risk
        if supplier.geopolitical_risk == RiskLevel.CRITICAL:
            risk_factors.append(("Geopolitical instability", 0.8, 8))
        elif supplier.geopolitical_risk == RiskLevel.HIGH:
            risk_factors.append(("Regional tensions", 0.6, 6))
        
        # Concentration risk
        if supplier.concentration_risk > 0.5:
            risk_factors.append(("High spend concentration", 0.5, 7))
        
        # Performance risk
        if supplier.delivery_score < 70:
            risk_factors.append(("Poor delivery performance", 0.6, 5))
        
        if supplier.quality_score < 70:
            risk_factors.append(("Quality concerns", 0.5, 6))
        
        # Calculate overall risk
        if risk_factors:
            max_probability = max(r[1] for r in risk_factors)
            max_severity = max(r[2] for r in risk_factors)
            risk_score = max_probability * max_severity
            
            if risk_score > 6:
                risk_level = RiskLevel.HIGH
            elif risk_score > 3:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
        else:
            risk_level = RiskLevel.MINIMAL
            max_probability = 0.1
            max_severity = 2
            risk_score = 0.2
        
        return RiskAssessment(
            assessment_id=f"RA-{supplier.supplier_id}",
            entity_type="supplier",
            entity_id=supplier.supplier_id,
            risk_category="supply",
            risk_level=risk_level,
            probability=max_probability,
            impact_severity=max_severity,
            risk_score=risk_score,
            description="; ".join(r[0] for r in risk_factors) if risk_factors else "No significant risks identified",
            potential_impact="Supply disruption, quality issues, cost increases",
            mitigation_strategy="Maintain alternative suppliers, increase safety stock, regular performance reviews",
            identified_date=datetime.now().strftime("%Y-%m-%d"),
        )


# =============================================================================
# Logistics Optimization Tools
# =============================================================================

class LogisticsOptimizer:
    """Logistics and route optimization tools."""
    
    @staticmethod
    def calculate_shipping_cost(
        weight_kg: float,
        volume_m3: float,
        distance_km: float,
        mode: ShipmentMode,
        rate: ShippingRate = None,
    ) -> float:
        """Calculate shipping cost."""
        
        # Default rates by mode (per kg-km)
        default_rates = {
            ShipmentMode.GROUND: 0.0005,
            ShipmentMode.AIR: 0.003,
            ShipmentMode.OCEAN: 0.0001,
            ShipmentMode.RAIL: 0.0003,
            ShipmentMode.EXPRESS: 0.005,
        }
        
        if rate:
            return rate.calculate_cost(weight_kg, volume_m3)
        
        rate_per_kg_km = default_rates.get(mode, 0.001)
        return weight_kg * distance_km * rate_per_kg_km
    
    @staticmethod
    def calculate_carbon_footprint(
        weight_kg: float,
        distance_km: float,
        mode: ShipmentMode,
    ) -> float:
        """
        Calculate carbon footprint in kg CO2.
        
        Based on average emission factors by transport mode.
        """
        # Emission factors (kg CO2 per tonne-km)
        emission_factors = {
            ShipmentMode.GROUND: 0.062,  # Truck
            ShipmentMode.AIR: 0.602,
            ShipmentMode.OCEAN: 0.016,
            ShipmentMode.RAIL: 0.022,
            ShipmentMode.EXPRESS: 0.700,  # Air express
        }
        
        factor = emission_factors.get(mode, 0.1)
        tonnes = weight_kg / 1000
        
        return tonnes * distance_km * factor
    
    @staticmethod
    def select_best_carrier(
        carriers: list[Carrier],
        rates: list[ShippingRate],
        weight_kg: float,
        volume_m3: float,
        required_transit_days: int = None,
        mode_preference: ShipmentMode = None,
    ) -> list[tuple[Carrier, ShippingRate, float]]:
        """
        Select best carrier for shipment.
        
        Returns: [(carrier, rate, total_cost)]
        """
        options = []
        
        for carrier in carriers:
            # Find applicable rates
            carrier_rates = [r for r in rates if r.carrier_id == carrier.carrier_id]
            
            for rate in carrier_rates:
                # Check mode preference
                if mode_preference and rate.mode != mode_preference:
                    continue
                
                # Check transit time requirement
                if required_transit_days and rate.transit_days > required_transit_days:
                    continue
                
                cost = rate.calculate_cost(weight_kg, volume_m3)
                options.append((carrier, rate, cost))
        
        # Sort by cost
        options.sort(key=lambda x: x[2])
        
        return options
    
    @staticmethod
    def optimize_route(
        origin: tuple[float, float],  # (lat, lon)
        destinations: list[tuple[str, float, float]],  # [(id, lat, lon)]
        max_stops: int = 10,
    ) -> list[str]:
        """
        Optimize delivery route using nearest neighbor heuristic.
        
        Returns: ordered list of destination IDs
        """
        if not destinations:
            return []
        
        def distance(p1: tuple, p2: tuple) -> float:
            """Haversine distance approximation."""
            lat1, lon1 = p1
            lat2, lon2 = p2
            return math.sqrt((lat2 - lat1)**2 + (lon2 - lon1)**2) * 111  # km approx
        
        route = []
        current = origin
        remaining = list(destinations)
        
        while remaining and len(route) < max_stops:
            # Find nearest
            nearest = min(remaining, key=lambda d: distance(current, (d[1], d[2])))
            route.append(nearest[0])
            current = (nearest[1], nearest[2])
            remaining.remove(nearest)
        
        return route
    
    @staticmethod
    def consolidate_shipments(
        shipments: list[dict],  # [{sku, quantity, weight, volume, destination}]
        max_weight_kg: float = 20000,
        max_volume_m3: float = 80,
    ) -> list[list[dict]]:
        """
        Consolidate shipments for efficiency.
        
        Returns: list of consolidated shipment groups
        """
        # Group by destination
        by_destination = {}
        for s in shipments:
            dest = s.get("destination", "DEFAULT")
            if dest not in by_destination:
                by_destination[dest] = []
            by_destination[dest].append(s)
        
        consolidated = []
        
        for dest, dest_shipments in by_destination.items():
            current_group = []
            current_weight = 0
            current_volume = 0
            
            for shipment in dest_shipments:
                weight = shipment.get("weight", 0)
                volume = shipment.get("volume", 0)
                
                if current_weight + weight <= max_weight_kg and current_volume + volume <= max_volume_m3:
                    current_group.append(shipment)
                    current_weight += weight
                    current_volume += volume
                else:
                    if current_group:
                        consolidated.append(current_group)
                    current_group = [shipment]
                    current_weight = weight
                    current_volume = volume
            
            if current_group:
                consolidated.append(current_group)
        
        return consolidated


# =============================================================================
# Scenario Analysis Tools
# =============================================================================

class ScenarioAnalyzer:
    """What-if scenario analysis tools."""
    
    @staticmethod
    def create_demand_shock_scenario(
        name: str,
        demand_change_pct: float,
        affected_products: list[str] = None,
    ) -> Scenario:
        """Create a demand shock scenario."""
        return Scenario(
            scenario_id=f"SCN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=name,
            description=f"Demand {'increase' if demand_change_pct > 0 else 'decrease'} of {abs(demand_change_pct)}%",
            parameters={
                "demand_change_pct": demand_change_pct,
                "affected_products": affected_products or "all",
            },
            created_at=datetime.now().isoformat(),
        )
    
    @staticmethod
    def create_supply_disruption_scenario(
        name: str,
        supplier_id: str,
        disruption_duration_days: int,
        capacity_reduction_pct: float = 100,
    ) -> Scenario:
        """Create a supply disruption scenario."""
        return Scenario(
            scenario_id=f"SCN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            name=name,
            description=f"Supplier disruption for {disruption_duration_days} days",
            parameters={
                "supplier_id": supplier_id,
                "duration_days": disruption_duration_days,
                "capacity_reduction_pct": capacity_reduction_pct,
            },
            created_at=datetime.now().isoformat(),
        )
    
    @staticmethod
    def analyze_scenario(
        scenario: Scenario,
        products: list[Product],
        inventory: dict[str, InventoryLevel],
        forecast: dict[str, int],
    ) -> dict:
        """Analyze impact of a scenario."""
        
        results = {
            "stockout_risk_products": [],
            "additional_safety_stock_needed": 0,
            "additional_cost": 0,
            "service_level_impact": 0,
        }
        
        demand_change = scenario.parameters.get("demand_change_pct", 0) / 100
        
        for product in products:
            sku = product.sku
            inv = inventory.get(sku, InventoryLevel(sku=sku))
            base_forecast = forecast.get(sku, 0)
            
            # Adjust forecast
            adjusted_forecast = int(base_forecast * (1 + demand_change))
            
            # Check stockout risk
            if inv.projected < adjusted_forecast:
                results["stockout_risk_products"].append({
                    "sku": sku,
                    "shortfall": adjusted_forecast - inv.projected,
                })
        
        results["products_at_risk"] = len(results["stockout_risk_products"])
        
        return results


# =============================================================================
# Report Generation Tools
# =============================================================================

class ReportGenerator:
    """Generate supply chain reports."""
    
    @staticmethod
    def generate_inventory_report(
        inventory_levels: list[InventoryLevel],
        products: dict[str, Product],
    ) -> str:
        """Generate inventory status report."""
        
        report = """# Inventory Status Report
        
## Summary

| Status | Count |
|--------|-------|
"""
        # Count by status
        status_counts = {}
        for inv in inventory_levels:
            status = inv.stock_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in sorted(status_counts.items()):
            report += f"| {status} | {count} |\n"
        
        report += "\n## Critical Items\n\n"
        report += "| SKU | Product | On Hand | Safety Stock | Status |\n"
        report += "|-----|---------|---------|--------------|--------|\n"
        
        critical_items = [inv for inv in inventory_levels if inv.stock_status in ["STOCKOUT", "CRITICAL"]]
        for inv in critical_items[:20]:
            product = products.get(inv.sku, Product(sku=inv.sku, name="Unknown"))
            report += f"| {inv.sku} | {product.name[:25]} | {inv.on_hand} | {inv.safety_stock} | {inv.stock_status} |\n"
        
        return report
    
    @staticmethod
    def generate_supplier_scorecard(suppliers: list[Supplier]) -> str:
        """Generate supplier scorecard report."""
        
        report = """# Supplier Scorecard

## Overall Rankings

| Rank | Supplier | Score | Quality | Delivery | Cost | Risk |
|------|----------|-------|---------|----------|------|------|
"""
        # Sort by overall score
        sorted_suppliers = sorted(suppliers, key=lambda s: s.overall_score, reverse=True)
        
        for i, supplier in enumerate(sorted_suppliers[:20], 1):
            report += f"| {i} | {supplier.name[:20]} | {supplier.overall_score:.0f} | {supplier.quality_score:.0f} | {supplier.delivery_score:.0f} | {supplier.cost_score:.0f} | {supplier.risk_indicator} |\n"
        
        # Tier distribution
        report += "\n## Tier Distribution\n\n"
        tier_counts = {}
        for s in suppliers:
            tier = s.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        for tier, count in sorted(tier_counts.items()):
            report += f"- **{tier}:** {count} suppliers\n"
        
        return report
    
    @staticmethod
    def generate_optimization_report(result: OptimizationResult) -> str:
        """Generate full optimization report."""
        return result.to_report()
