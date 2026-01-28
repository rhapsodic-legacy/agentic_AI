"""
Supply Chain Optimizer - Data Models

Comprehensive models for:
- Products and SKUs
- Suppliers and procurement
- Inventory management
- Demand forecasting
- Logistics and shipping
- Risk assessment
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import uuid
import json
import math


# =============================================================================
# Core Enums
# =============================================================================

class ProductCategory(Enum):
    """Product categories."""
    RAW_MATERIALS = "Raw Materials"
    COMPONENTS = "Components"
    FINISHED_GOODS = "Finished Goods"
    PACKAGING = "Packaging"
    MRO = "MRO"  # Maintenance, Repair, Operations


class UnitOfMeasure(Enum):
    """Units of measurement."""
    EACH = "Each"
    CASE = "Case"
    PALLET = "Pallet"
    KG = "Kilogram"
    LB = "Pound"
    LITER = "Liter"
    CUBIC_METER = "Cubic Meter"


class SupplierTier(Enum):
    """Supplier classification tiers."""
    STRATEGIC = "Strategic"
    PREFERRED = "Preferred"
    APPROVED = "Approved"
    CONDITIONAL = "Conditional"
    BLACKLISTED = "Blacklisted"


class RiskLevel(Enum):
    """Risk assessment levels."""
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    MINIMAL = "Minimal"


class OrderStatus(Enum):
    """Purchase order status."""
    DRAFT = "Draft"
    PENDING_APPROVAL = "Pending Approval"
    APPROVED = "Approved"
    SENT = "Sent"
    ACKNOWLEDGED = "Acknowledged"
    IN_PRODUCTION = "In Production"
    SHIPPED = "Shipped"
    IN_TRANSIT = "In Transit"
    DELIVERED = "Delivered"
    CANCELLED = "Cancelled"


class ShipmentMode(Enum):
    """Transportation modes."""
    GROUND = "Ground"
    AIR = "Air"
    OCEAN = "Ocean"
    RAIL = "Rail"
    INTERMODAL = "Intermodal"
    EXPRESS = "Express"


class ForecastMethod(Enum):
    """Forecasting methodologies."""
    MOVING_AVERAGE = "Moving Average"
    EXPONENTIAL_SMOOTHING = "Exponential Smoothing"
    ARIMA = "ARIMA"
    SEASONAL_DECOMPOSITION = "Seasonal Decomposition"
    ML_REGRESSION = "ML Regression"
    ENSEMBLE = "Ensemble"


# =============================================================================
# Product Models
# =============================================================================

@dataclass
class Product:
    """A product or SKU in the supply chain."""
    sku: str
    name: str
    description: str = ""
    
    # Classification
    category: ProductCategory = ProductCategory.FINISHED_GOODS
    unit_of_measure: UnitOfMeasure = UnitOfMeasure.EACH
    
    # Physical attributes
    weight_kg: float = 0.0
    volume_m3: float = 0.0
    
    # Financial
    unit_cost: float = 0.0
    selling_price: float = 0.0
    
    # Inventory parameters
    lead_time_days: int = 14
    min_order_quantity: int = 1
    order_multiple: int = 1
    
    # Safety stock parameters
    service_level_target: float = 0.95  # 95% service level
    demand_variability: float = 0.0  # Standard deviation
    
    # ABC/XYZ classification
    abc_class: str = "B"  # A, B, C
    xyz_class: str = "Y"  # X, Y, Z (demand variability)
    
    # Status
    is_active: bool = True
    is_critical: bool = False
    
    def __post_init__(self):
        if not self.sku:
            self.sku = f"SKU-{uuid.uuid4().hex[:8].upper()}"
    
    @property
    def margin(self) -> float:
        """Calculate profit margin."""
        if self.selling_price == 0:
            return 0
        return (self.selling_price - self.unit_cost) / self.selling_price
    
    def to_dict(self) -> dict:
        return {
            "sku": self.sku,
            "name": self.name,
            "category": self.category.value,
            "unit_cost": self.unit_cost,
            "lead_time_days": self.lead_time_days,
            "abc_class": self.abc_class,
        }


# =============================================================================
# Inventory Models
# =============================================================================

@dataclass
class InventoryLevel:
    """Current inventory status for a product."""
    sku: str
    
    # Quantities
    on_hand: int = 0
    on_order: int = 0
    reserved: int = 0
    in_transit: int = 0
    
    # Calculated levels
    safety_stock: int = 0
    reorder_point: int = 0
    economic_order_quantity: int = 0
    max_stock: int = 0
    
    # Location
    warehouse_id: str = "WH-MAIN"
    
    # Timestamps
    last_receipt_date: Optional[str] = None
    last_issue_date: Optional[str] = None
    
    @property
    def available(self) -> int:
        """Available to promise quantity."""
        return self.on_hand - self.reserved
    
    @property
    def projected(self) -> int:
        """Projected inventory (on-hand + on-order + in-transit - reserved)."""
        return self.on_hand + self.on_order + self.in_transit - self.reserved
    
    @property
    def days_of_supply(self) -> float:
        """Estimate days of supply (requires average daily demand)."""
        return 0  # Calculated externally
    
    @property
    def needs_reorder(self) -> bool:
        """Check if reorder is needed."""
        return self.projected <= self.reorder_point
    
    @property
    def stock_status(self) -> str:
        """Get stock status."""
        if self.available <= 0:
            return "STOCKOUT"
        elif self.available <= self.safety_stock:
            return "CRITICAL"
        elif self.needs_reorder:
            return "REORDER"
        elif self.available > self.max_stock:
            return "OVERSTOCK"
        else:
            return "HEALTHY"


@dataclass
class InventoryTransaction:
    """An inventory movement transaction."""
    transaction_id: str
    sku: str
    transaction_type: str  # RECEIPT, ISSUE, ADJUSTMENT, TRANSFER
    quantity: int
    timestamp: str
    
    # Reference
    reference_type: str = ""  # PO, SO, ADJ, TRANSFER
    reference_id: str = ""
    
    # Location
    from_location: str = ""
    to_location: str = ""
    
    # Financial
    unit_cost: float = 0.0
    
    notes: str = ""


# =============================================================================
# Supplier Models
# =============================================================================

@dataclass
class Supplier:
    """A supplier/vendor in the supply chain."""
    supplier_id: str
    name: str
    
    # Contact
    contact_name: str = ""
    email: str = ""
    phone: str = ""
    
    # Location
    country: str = ""
    city: str = ""
    address: str = ""
    
    # Classification
    tier: SupplierTier = SupplierTier.APPROVED
    
    # Performance metrics
    quality_score: float = 0.0  # 0-100
    delivery_score: float = 0.0  # 0-100
    cost_score: float = 0.0  # 0-100
    responsiveness_score: float = 0.0  # 0-100
    
    # Terms
    payment_terms: str = "Net 30"
    currency: str = "USD"
    incoterms: str = "FOB"
    
    # Lead times
    standard_lead_time_days: int = 14
    express_lead_time_days: int = 7
    
    # Risk factors
    financial_risk: RiskLevel = RiskLevel.LOW
    geopolitical_risk: RiskLevel = RiskLevel.LOW
    concentration_risk: float = 0.0  # % of category spend
    
    # Status
    is_active: bool = True
    certified: bool = False
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        weights = {
            "quality": 0.30,
            "delivery": 0.30,
            "cost": 0.25,
            "responsiveness": 0.15,
        }
        return (
            self.quality_score * weights["quality"] +
            self.delivery_score * weights["delivery"] +
            self.cost_score * weights["cost"] +
            self.responsiveness_score * weights["responsiveness"]
        )
    
    @property
    def risk_indicator(self) -> str:
        """Get risk indicator emoji."""
        risks = [self.financial_risk, self.geopolitical_risk]
        if RiskLevel.CRITICAL in risks or RiskLevel.HIGH in risks:
            return "🔴"
        elif RiskLevel.MEDIUM in risks:
            return "🟡"
        else:
            return "🟢"
    
    def to_dict(self) -> dict:
        return {
            "supplier_id": self.supplier_id,
            "name": self.name,
            "tier": self.tier.value,
            "overall_score": round(self.overall_score, 1),
            "lead_time": self.standard_lead_time_days,
            "risk": self.risk_indicator,
        }


@dataclass
class SupplierQuote:
    """A price quote from a supplier."""
    quote_id: str
    supplier_id: str
    sku: str
    
    # Pricing
    unit_price: float
    currency: str = "USD"
    
    # Quantity breaks
    min_quantity: int = 1
    max_quantity: Optional[int] = None
    
    # Terms
    lead_time_days: int = 14
    valid_from: str = ""
    valid_to: str = ""
    
    # Additional costs
    shipping_cost: float = 0.0
    handling_cost: float = 0.0
    
    @property
    def total_landed_cost(self) -> float:
        """Calculate total landed cost per unit."""
        return self.unit_price + self.shipping_cost + self.handling_cost


# =============================================================================
# Demand and Forecasting Models
# =============================================================================

@dataclass
class DemandHistory:
    """Historical demand data for a product."""
    sku: str
    period: str  # YYYY-MM or YYYY-WW
    quantity: int
    
    # Context
    revenue: float = 0.0
    num_orders: int = 0
    
    # Adjustments
    promotions: bool = False
    stockout_days: int = 0  # Days product was unavailable


@dataclass
class DemandForecast:
    """Demand forecast for a product."""
    forecast_id: str
    sku: str
    
    # Forecast details
    period: str  # YYYY-MM or YYYY-WW
    forecast_quantity: int
    
    # Confidence
    lower_bound: int = 0
    upper_bound: int = 0
    confidence_level: float = 0.95
    
    # Methodology
    method: ForecastMethod = ForecastMethod.EXPONENTIAL_SMOOTHING
    
    # Accuracy (filled after actual demand known)
    actual_quantity: Optional[int] = None
    
    # Metadata
    created_at: str = ""
    created_by: str = ""
    
    @property
    def forecast_error(self) -> Optional[float]:
        """Calculate forecast error if actual is known."""
        if self.actual_quantity is None:
            return None
        if self.forecast_quantity == 0:
            return None
        return abs(self.actual_quantity - self.forecast_quantity) / self.forecast_quantity
    
    @property
    def mape(self) -> Optional[float]:
        """Mean Absolute Percentage Error."""
        return self.forecast_error


@dataclass
class SeasonalityPattern:
    """Seasonality pattern for demand planning."""
    sku: str
    period_type: str  # "monthly", "weekly"
    
    # Seasonal indices (1.0 = average)
    indices: dict = field(default_factory=dict)  # {period: index}
    
    # Pattern characteristics
    peak_period: str = ""
    trough_period: str = ""
    variability: float = 0.0


# =============================================================================
# Order and Procurement Models
# =============================================================================

@dataclass
class PurchaseOrderLine:
    """A line item in a purchase order."""
    line_number: int
    sku: str
    product_name: str
    quantity: int
    unit_price: float
    
    # Dates
    requested_date: str = ""
    promised_date: str = ""
    
    # Status
    quantity_received: int = 0
    
    @property
    def line_total(self) -> float:
        return self.quantity * self.unit_price
    
    @property
    def is_complete(self) -> bool:
        return self.quantity_received >= self.quantity


@dataclass
class PurchaseOrder:
    """A purchase order to a supplier."""
    po_number: str
    supplier_id: str
    supplier_name: str
    
    # Lines
    lines: list[PurchaseOrderLine] = field(default_factory=list)
    
    # Dates
    order_date: str = ""
    required_date: str = ""
    expected_delivery: str = ""
    
    # Status
    status: OrderStatus = OrderStatus.DRAFT
    
    # Terms
    payment_terms: str = "Net 30"
    incoterms: str = "FOB"
    currency: str = "USD"
    
    # Shipping
    ship_to_address: str = ""
    shipping_method: ShipmentMode = ShipmentMode.GROUND
    
    # Costs
    shipping_cost: float = 0.0
    tax: float = 0.0
    
    # Notes
    notes: str = ""
    
    @property
    def subtotal(self) -> float:
        return sum(line.line_total for line in self.lines)
    
    @property
    def total(self) -> float:
        return self.subtotal + self.shipping_cost + self.tax
    
    @property
    def line_count(self) -> int:
        return len(self.lines)
    
    def add_line(self, sku: str, name: str, qty: int, price: float):
        line = PurchaseOrderLine(
            line_number=len(self.lines) + 1,
            sku=sku,
            product_name=name,
            quantity=qty,
            unit_price=price,
        )
        self.lines.append(line)
    
    def to_dict(self) -> dict:
        return {
            "po_number": self.po_number,
            "supplier": self.supplier_name,
            "status": self.status.value,
            "lines": self.line_count,
            "total": f"${self.total:,.2f}",
            "expected_delivery": self.expected_delivery,
        }


# =============================================================================
# Logistics Models
# =============================================================================

@dataclass
class Warehouse:
    """A warehouse or distribution center."""
    warehouse_id: str
    name: str
    
    # Location
    country: str = ""
    city: str = ""
    address: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    
    # Capacity
    total_capacity_m3: float = 0.0
    used_capacity_m3: float = 0.0
    
    # Operations
    receiving_docks: int = 0
    shipping_docks: int = 0
    
    # Costs
    storage_cost_per_m3: float = 0.0
    handling_cost_per_unit: float = 0.0
    
    @property
    def utilization(self) -> float:
        if self.total_capacity_m3 == 0:
            return 0
        return self.used_capacity_m3 / self.total_capacity_m3


@dataclass
class Carrier:
    """A shipping carrier."""
    carrier_id: str
    name: str
    
    # Service details
    modes: list[ShipmentMode] = field(default_factory=list)
    
    # Performance
    on_time_delivery_rate: float = 0.0  # 0-100%
    damage_rate: float = 0.0  # 0-100%
    
    # Coverage
    domestic: bool = True
    international: bool = False
    countries_served: list[str] = field(default_factory=list)
    
    # Contact
    contact_email: str = ""
    api_enabled: bool = False


@dataclass
class ShippingRate:
    """A shipping rate from a carrier."""
    rate_id: str
    carrier_id: str
    carrier_name: str
    
    # Route
    origin: str = ""
    destination: str = ""
    
    # Service
    mode: ShipmentMode = ShipmentMode.GROUND
    service_level: str = "Standard"  # Standard, Express, Economy
    
    # Pricing
    base_rate: float = 0.0
    rate_per_kg: float = 0.0
    rate_per_m3: float = 0.0
    fuel_surcharge_pct: float = 0.0
    
    # Transit
    transit_days: int = 0
    
    # Validity
    valid_from: str = ""
    valid_to: str = ""
    
    def calculate_cost(self, weight_kg: float, volume_m3: float) -> float:
        """Calculate shipping cost."""
        weight_cost = weight_kg * self.rate_per_kg
        volume_cost = volume_m3 * self.rate_per_m3
        dimensional_cost = max(weight_cost, volume_cost)
        subtotal = self.base_rate + dimensional_cost
        return subtotal * (1 + self.fuel_surcharge_pct / 100)


@dataclass
class Route:
    """An optimized shipping route."""
    route_id: str
    
    # Endpoints
    origin_warehouse: str
    destination: str  # Warehouse or customer
    
    # Stops
    stops: list[str] = field(default_factory=list)
    
    # Metrics
    total_distance_km: float = 0.0
    total_time_hours: float = 0.0
    total_cost: float = 0.0
    
    # Shipments
    shipment_count: int = 0
    total_weight_kg: float = 0.0
    total_volume_m3: float = 0.0
    
    # Mode
    mode: ShipmentMode = ShipmentMode.GROUND
    carrier_id: str = ""
    
    # Carbon
    carbon_footprint_kg: float = 0.0


# =============================================================================
# Risk Models
# =============================================================================

@dataclass
class RiskAssessment:
    """A supply chain risk assessment."""
    assessment_id: str
    entity_type: str  # "supplier", "product", "route", "region"
    entity_id: str
    
    # Risk categorization
    risk_category: str  # "supply", "demand", "logistics", "financial", "geopolitical"
    risk_level: RiskLevel = RiskLevel.MEDIUM
    
    # Impact
    probability: float = 0.0  # 0-1
    impact_severity: float = 0.0  # 0-10
    risk_score: float = 0.0  # probability * severity
    
    # Description
    description: str = ""
    potential_impact: str = ""
    
    # Mitigation
    mitigation_strategy: str = ""
    contingency_plan: str = ""
    
    # Status
    status: str = "Active"  # Active, Mitigated, Closed
    owner: str = ""
    
    # Dates
    identified_date: str = ""
    review_date: str = ""


# =============================================================================
# Scenario Analysis Models
# =============================================================================

@dataclass
class Scenario:
    """A what-if scenario for analysis."""
    scenario_id: str
    name: str
    description: str = ""
    
    # Parameters to change
    parameters: dict = field(default_factory=dict)
    # e.g., {"demand_change_pct": 20, "lead_time_change_days": 7}
    
    # Results
    results: dict = field(default_factory=dict)
    
    # Comparison
    baseline_metrics: dict = field(default_factory=dict)
    scenario_metrics: dict = field(default_factory=dict)
    
    # Status
    created_at: str = ""
    analyzed: bool = False


# =============================================================================
# KPI and Metrics Models
# =============================================================================

@dataclass
class SupplyChainKPIs:
    """Key Performance Indicators for supply chain."""
    period: str  # YYYY-MM
    
    # Inventory KPIs
    inventory_turnover: float = 0.0
    days_of_inventory: float = 0.0
    inventory_accuracy: float = 0.0
    carrying_cost_pct: float = 0.0
    
    # Service KPIs
    fill_rate: float = 0.0  # Order fill rate
    otif_rate: float = 0.0  # On-time in-full
    stockout_rate: float = 0.0
    backorder_rate: float = 0.0
    
    # Procurement KPIs
    supplier_otd: float = 0.0  # On-time delivery
    supplier_quality: float = 0.0
    cost_savings: float = 0.0
    maverick_spend_pct: float = 0.0
    
    # Logistics KPIs
    freight_cost_per_unit: float = 0.0
    warehouse_utilization: float = 0.0
    order_cycle_time_days: float = 0.0
    perfect_order_rate: float = 0.0
    
    # Financial KPIs
    total_supply_chain_cost: float = 0.0
    cost_as_pct_revenue: float = 0.0
    working_capital: float = 0.0
    cash_to_cash_days: float = 0.0
    
    # Sustainability
    carbon_footprint_tons: float = 0.0
    packaging_waste_kg: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "period": self.period,
            "inventory_turnover": self.inventory_turnover,
            "fill_rate": f"{self.fill_rate:.1%}",
            "otif_rate": f"{self.otif_rate:.1%}",
            "supplier_otd": f"{self.supplier_otd:.1%}",
            "freight_cost_per_unit": f"${self.freight_cost_per_unit:.2f}",
        }


# =============================================================================
# Optimization Output Models
# =============================================================================

@dataclass
class ReplenishmentRecommendation:
    """A replenishment recommendation for a product."""
    sku: str
    product_name: str
    
    # Current state
    current_stock: int
    safety_stock: int
    reorder_point: int
    
    # Forecast
    forecast_demand: int
    forecast_period: str
    
    # Recommendation
    recommended_order_qty: int
    recommended_supplier: str
    estimated_cost: float
    
    # Timing
    order_by_date: str
    expected_receipt_date: str
    
    # Priority
    priority: str = "Normal"  # Critical, High, Normal, Low
    
    # Reasoning
    reason: str = ""


@dataclass
class OptimizationResult:
    """Results from supply chain optimization."""
    optimization_id: str
    run_date: str
    
    # Scope
    products_analyzed: int
    suppliers_analyzed: int
    routes_analyzed: int
    
    # Inventory recommendations
    replenishment_recommendations: list[ReplenishmentRecommendation] = field(default_factory=list)
    
    # Summary metrics
    current_inventory_value: float = 0.0
    recommended_inventory_value: float = 0.0
    inventory_reduction_pct: float = 0.0
    working_capital_freed: float = 0.0
    
    # Risk improvements
    current_stockout_risk: float = 0.0
    recommended_stockout_risk: float = 0.0
    
    # Logistics improvements
    routes_before: int = 0
    routes_after: int = 0
    shipping_cost_savings: float = 0.0
    carbon_reduction_pct: float = 0.0
    
    # Supplier recommendations
    supplier_changes: list[dict] = field(default_factory=list)
    
    # Alerts
    alerts: list[str] = field(default_factory=list)
    
    def to_report(self) -> str:
        """Generate markdown report."""
        report = f"""# Supply Chain Optimization Report
**Generated:** {self.run_date}
**Optimization ID:** {self.optimization_id}

## Executive Summary

| Metric | Current | Recommended | Change |
|--------|---------|-------------|--------|
| Inventory Value | ${self.current_inventory_value:,.0f} | ${self.recommended_inventory_value:,.0f} | {self.inventory_reduction_pct:+.1f}% |
| Stockout Risk | {self.current_stockout_risk:.1%} | {self.recommended_stockout_risk:.1%} | {(self.recommended_stockout_risk - self.current_stockout_risk)*100:+.1f}pp |
| Shipping Routes | {self.routes_before} | {self.routes_after} | {((self.routes_after - self.routes_before)/self.routes_before)*100:+.1f}% |
| Carbon Footprint | - | - | {self.carbon_reduction_pct:+.1f}% |

**Working Capital Freed:** ${self.working_capital_freed:,.0f}
**Monthly Shipping Savings:** ${self.shipping_cost_savings:,.0f}

## Replenishment Recommendations

| SKU | Product | Current | Forecast | Order Qty | Supplier | Cost |
|-----|---------|---------|----------|-----------|----------|------|
"""
        for rec in self.replenishment_recommendations[:10]:
            report += f"| {rec.sku} | {rec.product_name[:20]} | {rec.current_stock} | {rec.forecast_demand} | {rec.recommended_order_qty} | {rec.recommended_supplier[:15]} | ${rec.estimated_cost:,.0f} |\n"
        
        if self.alerts:
            report += "\n## ⚠️ Alerts\n"
            for alert in self.alerts:
                report += f"- {alert}\n"
        
        return report
