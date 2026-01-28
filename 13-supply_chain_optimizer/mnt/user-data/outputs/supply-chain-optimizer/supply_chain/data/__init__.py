"""
Supply Chain Optimizer - Sample Data

Realistic sample data for:
- Products and SKUs
- Suppliers with performance metrics
- Inventory levels
- Demand history
- Carriers and rates
"""

from datetime import datetime, timedelta
import random

from ..models import (
    Product, ProductCategory, UnitOfMeasure,
    Supplier, SupplierTier, SupplierQuote, RiskLevel,
    InventoryLevel, DemandHistory,
    Carrier, ShippingRate, ShipmentMode, Warehouse,
)


# =============================================================================
# Products
# =============================================================================

PRODUCTS = {
    "SKU-001": Product(
        sku="SKU-001",
        name="Industrial Motor Assembly",
        description="High-torque electric motor for manufacturing equipment",
        category=ProductCategory.COMPONENTS,
        unit_of_measure=UnitOfMeasure.EACH,
        weight_kg=25.0,
        volume_m3=0.08,
        unit_cost=450.00,
        selling_price=720.00,
        lead_time_days=21,
        min_order_quantity=10,
        order_multiple=5,
        service_level_target=0.98,
        abc_class="A",
        is_critical=True,
    ),
    "SKU-002": Product(
        sku="SKU-002",
        name="Precision Bearing Set",
        description="High-precision ball bearing kit for rotating machinery",
        category=ProductCategory.COMPONENTS,
        unit_of_measure=UnitOfMeasure.EACH,
        weight_kg=2.5,
        volume_m3=0.005,
        unit_cost=85.00,
        selling_price=145.00,
        lead_time_days=14,
        min_order_quantity=50,
        order_multiple=25,
        service_level_target=0.95,
        abc_class="A",
    ),
    "SKU-003": Product(
        sku="SKU-003",
        name="Control Circuit Board",
        description="Programmable logic controller board",
        category=ProductCategory.COMPONENTS,
        unit_of_measure=UnitOfMeasure.EACH,
        weight_kg=0.5,
        volume_m3=0.002,
        unit_cost=125.00,
        selling_price=215.00,
        lead_time_days=28,
        min_order_quantity=25,
        order_multiple=25,
        service_level_target=0.95,
        abc_class="A",
        is_critical=True,
    ),
    "SKU-004": Product(
        sku="SKU-004",
        name="Hydraulic Cylinder",
        description="Double-acting hydraulic cylinder for heavy machinery",
        category=ProductCategory.COMPONENTS,
        unit_of_measure=UnitOfMeasure.EACH,
        weight_kg=35.0,
        volume_m3=0.12,
        unit_cost=280.00,
        selling_price=450.00,
        lead_time_days=18,
        min_order_quantity=5,
        order_multiple=1,
        service_level_target=0.95,
        abc_class="B",
    ),
    "SKU-005": Product(
        sku="SKU-005",
        name="Stainless Steel Tubing",
        description="304 stainless steel tubing, 2-inch diameter",
        category=ProductCategory.RAW_MATERIALS,
        unit_of_measure=UnitOfMeasure.KG,
        weight_kg=1.0,
        volume_m3=0.001,
        unit_cost=12.50,
        selling_price=18.00,
        lead_time_days=10,
        min_order_quantity=500,
        order_multiple=100,
        service_level_target=0.90,
        abc_class="B",
    ),
    "SKU-006": Product(
        sku="SKU-006",
        name="Industrial Sensor Package",
        description="Multi-sensor package with temperature, pressure, and vibration",
        category=ProductCategory.COMPONENTS,
        unit_of_measure=UnitOfMeasure.EACH,
        weight_kg=0.8,
        volume_m3=0.003,
        unit_cost=165.00,
        selling_price=275.00,
        lead_time_days=21,
        min_order_quantity=20,
        order_multiple=10,
        service_level_target=0.95,
        abc_class="A",
    ),
    "SKU-007": Product(
        sku="SKU-007",
        name="Pneumatic Valve Assembly",
        description="5-port pneumatic directional control valve",
        category=ProductCategory.COMPONENTS,
        unit_of_measure=UnitOfMeasure.EACH,
        weight_kg=3.2,
        volume_m3=0.008,
        unit_cost=95.00,
        selling_price=155.00,
        lead_time_days=14,
        min_order_quantity=25,
        order_multiple=5,
        service_level_target=0.93,
        abc_class="B",
    ),
    "SKU-008": Product(
        sku="SKU-008",
        name="Safety Enclosure Panel",
        description="Industrial machine guard panel with interlock",
        category=ProductCategory.FINISHED_GOODS,
        unit_of_measure=UnitOfMeasure.EACH,
        weight_kg=18.0,
        volume_m3=0.25,
        unit_cost=210.00,
        selling_price=340.00,
        lead_time_days=12,
        min_order_quantity=10,
        order_multiple=5,
        service_level_target=0.90,
        abc_class="B",
    ),
    "SKU-009": Product(
        sku="SKU-009",
        name="Electrical Cable Bundle",
        description="Industrial-grade power and signal cable bundle",
        category=ProductCategory.RAW_MATERIALS,
        unit_of_measure=UnitOfMeasure.KG,
        weight_kg=1.0,
        volume_m3=0.002,
        unit_cost=8.50,
        selling_price=12.00,
        lead_time_days=7,
        min_order_quantity=100,
        order_multiple=50,
        service_level_target=0.85,
        abc_class="C",
    ),
    "SKU-010": Product(
        sku="SKU-010",
        name="Fastener Kit - Industrial",
        description="Mixed fastener kit with bolts, nuts, and washers",
        category=ProductCategory.MRO,
        unit_of_measure=UnitOfMeasure.EACH,
        weight_kg=5.0,
        volume_m3=0.01,
        unit_cost=45.00,
        selling_price=75.00,
        lead_time_days=5,
        min_order_quantity=50,
        order_multiple=10,
        service_level_target=0.90,
        abc_class="C",
    ),
    "SKU-011": Product(
        sku="SKU-011",
        name="Drive Belt Assembly",
        description="Reinforced rubber drive belt for conveyors",
        category=ProductCategory.COMPONENTS,
        unit_of_measure=UnitOfMeasure.EACH,
        weight_kg=2.8,
        volume_m3=0.015,
        unit_cost=55.00,
        selling_price=90.00,
        lead_time_days=10,
        min_order_quantity=20,
        order_multiple=10,
        service_level_target=0.92,
        abc_class="B",
    ),
    "SKU-012": Product(
        sku="SKU-012",
        name="Lubricant - Industrial Grade",
        description="High-temperature synthetic lubricant",
        category=ProductCategory.MRO,
        unit_of_measure=UnitOfMeasure.LITER,
        weight_kg=0.9,
        volume_m3=0.001,
        unit_cost=18.00,
        selling_price=28.00,
        lead_time_days=5,
        min_order_quantity=100,
        order_multiple=20,
        service_level_target=0.85,
        abc_class="C",
    ),
}


# =============================================================================
# Suppliers
# =============================================================================

SUPPLIERS = {
    "SUP-001": Supplier(
        supplier_id="SUP-001",
        name="TechParts International",
        contact_name="John Chen",
        email="jchen@techparts.com",
        country="Taiwan",
        city="Taipei",
        tier=SupplierTier.STRATEGIC,
        quality_score=94,
        delivery_score=91,
        cost_score=85,
        responsiveness_score=88,
        payment_terms="Net 45",
        standard_lead_time_days=21,
        express_lead_time_days=10,
        financial_risk=RiskLevel.LOW,
        geopolitical_risk=RiskLevel.MEDIUM,
        concentration_risk=0.35,
        certified=True,
    ),
    "SUP-002": Supplier(
        supplier_id="SUP-002",
        name="Midwest Manufacturing Co.",
        contact_name="Sarah Johnson",
        email="sjohnson@midwestmfg.com",
        country="USA",
        city="Chicago",
        tier=SupplierTier.PREFERRED,
        quality_score=88,
        delivery_score=95,
        cost_score=75,
        responsiveness_score=92,
        payment_terms="Net 30",
        standard_lead_time_days=10,
        express_lead_time_days=5,
        financial_risk=RiskLevel.LOW,
        geopolitical_risk=RiskLevel.MINIMAL,
        concentration_risk=0.20,
        certified=True,
    ),
    "SUP-003": Supplier(
        supplier_id="SUP-003",
        name="EuroPrecision GmbH",
        contact_name="Hans Mueller",
        email="hmueller@europrecision.de",
        country="Germany",
        city="Stuttgart",
        tier=SupplierTier.STRATEGIC,
        quality_score=97,
        delivery_score=89,
        cost_score=70,
        responsiveness_score=85,
        payment_terms="Net 60",
        standard_lead_time_days=28,
        express_lead_time_days=14,
        financial_risk=RiskLevel.MINIMAL,
        geopolitical_risk=RiskLevel.MINIMAL,
        concentration_risk=0.15,
        certified=True,
    ),
    "SUP-004": Supplier(
        supplier_id="SUP-004",
        name="Dragon Industries Ltd.",
        contact_name="Wei Zhang",
        email="wzhang@dragonind.cn",
        country="China",
        city="Shenzhen",
        tier=SupplierTier.APPROVED,
        quality_score=78,
        delivery_score=82,
        cost_score=95,
        responsiveness_score=80,
        payment_terms="Net 30",
        standard_lead_time_days=35,
        express_lead_time_days=21,
        financial_risk=RiskLevel.MEDIUM,
        geopolitical_risk=RiskLevel.HIGH,
        concentration_risk=0.25,
        certified=False,
    ),
    "SUP-005": Supplier(
        supplier_id="SUP-005",
        name="Pacific Components Inc.",
        contact_name="Mike Torres",
        email="mtorres@pacificcomp.com",
        country="USA",
        city="Los Angeles",
        tier=SupplierTier.PREFERRED,
        quality_score=85,
        delivery_score=90,
        cost_score=82,
        responsiveness_score=88,
        payment_terms="Net 30",
        standard_lead_time_days=14,
        express_lead_time_days=7,
        financial_risk=RiskLevel.LOW,
        geopolitical_risk=RiskLevel.MINIMAL,
        concentration_risk=0.10,
        certified=True,
    ),
    "SUP-006": Supplier(
        supplier_id="SUP-006",
        name="IndoSteel Corp.",
        contact_name="Raj Patel",
        email="rpatel@indosteel.in",
        country="India",
        city="Mumbai",
        tier=SupplierTier.APPROVED,
        quality_score=80,
        delivery_score=75,
        cost_score=92,
        responsiveness_score=78,
        payment_terms="Net 30",
        standard_lead_time_days=28,
        express_lead_time_days=18,
        financial_risk=RiskLevel.MEDIUM,
        geopolitical_risk=RiskLevel.LOW,
        concentration_risk=0.08,
        certified=False,
    ),
    "SUP-007": Supplier(
        supplier_id="SUP-007",
        name="Northern Components AB",
        contact_name="Erik Lindqvist",
        email="elindqvist@northerncomp.se",
        country="Sweden",
        city="Stockholm",
        tier=SupplierTier.APPROVED,
        quality_score=92,
        delivery_score=88,
        cost_score=72,
        responsiveness_score=90,
        payment_terms="Net 45",
        standard_lead_time_days=21,
        express_lead_time_days=12,
        financial_risk=RiskLevel.MINIMAL,
        geopolitical_risk=RiskLevel.MINIMAL,
        concentration_risk=0.05,
        certified=True,
    ),
    "SUP-008": Supplier(
        supplier_id="SUP-008",
        name="FastTech Vietnam",
        contact_name="Nguyen Thi Lan",
        email="ntlan@fasttech.vn",
        country="Vietnam",
        city="Ho Chi Minh City",
        tier=SupplierTier.CONDITIONAL,
        quality_score=72,
        delivery_score=78,
        cost_score=96,
        responsiveness_score=75,
        payment_terms="Net 15",
        standard_lead_time_days=30,
        express_lead_time_days=20,
        financial_risk=RiskLevel.HIGH,
        geopolitical_risk=RiskLevel.MEDIUM,
        concentration_risk=0.03,
        certified=False,
    ),
}


# =============================================================================
# Supplier Quotes
# =============================================================================

SUPPLIER_QUOTES = [
    # SKU-001 quotes
    SupplierQuote(quote_id="Q-001-001", supplier_id="SUP-001", sku="SKU-001", unit_price=440.00, lead_time_days=21, min_quantity=10),
    SupplierQuote(quote_id="Q-001-002", supplier_id="SUP-002", sku="SKU-001", unit_price=475.00, lead_time_days=12, min_quantity=5),
    SupplierQuote(quote_id="Q-001-003", supplier_id="SUP-003", sku="SKU-001", unit_price=485.00, lead_time_days=28, min_quantity=10),
    
    # SKU-002 quotes
    SupplierQuote(quote_id="Q-002-001", supplier_id="SUP-003", sku="SKU-002", unit_price=82.00, lead_time_days=18, min_quantity=50),
    SupplierQuote(quote_id="Q-002-002", supplier_id="SUP-005", sku="SKU-002", unit_price=88.00, lead_time_days=10, min_quantity=25),
    SupplierQuote(quote_id="Q-002-003", supplier_id="SUP-007", sku="SKU-002", unit_price=84.00, lead_time_days=21, min_quantity=50),
    
    # SKU-003 quotes
    SupplierQuote(quote_id="Q-003-001", supplier_id="SUP-001", sku="SKU-003", unit_price=118.00, lead_time_days=28, min_quantity=25),
    SupplierQuote(quote_id="Q-003-002", supplier_id="SUP-004", sku="SKU-003", unit_price=95.00, lead_time_days=35, min_quantity=50),
    
    # SKU-004 quotes
    SupplierQuote(quote_id="Q-004-001", supplier_id="SUP-002", sku="SKU-004", unit_price=290.00, lead_time_days=14, min_quantity=5),
    SupplierQuote(quote_id="Q-004-002", supplier_id="SUP-005", sku="SKU-004", unit_price=275.00, lead_time_days=18, min_quantity=10),
    
    # SKU-005 quotes
    SupplierQuote(quote_id="Q-005-001", supplier_id="SUP-006", sku="SKU-005", unit_price=10.50, lead_time_days=25, min_quantity=1000),
    SupplierQuote(quote_id="Q-005-002", supplier_id="SUP-002", sku="SKU-005", unit_price=13.00, lead_time_days=8, min_quantity=500),
    
    # SKU-006 quotes
    SupplierQuote(quote_id="Q-006-001", supplier_id="SUP-001", sku="SKU-006", unit_price=158.00, lead_time_days=21, min_quantity=20),
    SupplierQuote(quote_id="Q-006-002", supplier_id="SUP-003", sku="SKU-006", unit_price=175.00, lead_time_days=25, min_quantity=10),
]


# =============================================================================
# Inventory Levels
# =============================================================================

def generate_inventory_levels() -> dict:
    """Generate current inventory levels."""
    levels = {}
    
    inventory_data = {
        "SKU-001": {"on_hand": 45, "on_order": 30, "safety_stock": 20, "reorder_point": 35},
        "SKU-002": {"on_hand": 280, "on_order": 0, "safety_stock": 100, "reorder_point": 150},
        "SKU-003": {"on_hand": 15, "on_order": 50, "safety_stock": 25, "reorder_point": 40},
        "SKU-004": {"on_hand": 35, "on_order": 20, "safety_stock": 15, "reorder_point": 25},
        "SKU-005": {"on_hand": 2500, "on_order": 0, "safety_stock": 800, "reorder_point": 1200},
        "SKU-006": {"on_hand": 85, "on_order": 0, "safety_stock": 40, "reorder_point": 60},
        "SKU-007": {"on_hand": 150, "on_order": 0, "safety_stock": 50, "reorder_point": 75},
        "SKU-008": {"on_hand": 25, "on_order": 30, "safety_stock": 15, "reorder_point": 20},
        "SKU-009": {"on_hand": 450, "on_order": 200, "safety_stock": 150, "reorder_point": 250},
        "SKU-010": {"on_hand": 180, "on_order": 0, "safety_stock": 75, "reorder_point": 100},
        "SKU-011": {"on_hand": 95, "on_order": 0, "safety_stock": 40, "reorder_point": 60},
        "SKU-012": {"on_hand": 320, "on_order": 0, "safety_stock": 100, "reorder_point": 150},
    }
    
    for sku, data in inventory_data.items():
        levels[sku] = InventoryLevel(
            sku=sku,
            on_hand=data["on_hand"],
            on_order=data["on_order"],
            safety_stock=data["safety_stock"],
            reorder_point=data["reorder_point"],
            economic_order_quantity=data.get("eoq", data["reorder_point"] * 2),
        )
    
    return levels


# =============================================================================
# Demand History
# =============================================================================

def generate_demand_history() -> dict:
    """Generate 12 months of demand history."""
    history = {}
    
    # Base monthly demand with seasonality
    base_demand = {
        "SKU-001": [40, 35, 45, 50, 55, 60, 55, 50, 45, 40, 35, 45],
        "SKU-002": [200, 180, 220, 250, 280, 300, 290, 270, 240, 210, 190, 220],
        "SKU-003": [30, 25, 35, 40, 45, 50, 48, 42, 38, 32, 28, 35],
        "SKU-004": [20, 18, 22, 25, 28, 30, 28, 26, 24, 21, 19, 22],
        "SKU-005": [1500, 1400, 1600, 1800, 2000, 2200, 2100, 1900, 1700, 1550, 1450, 1600],
        "SKU-006": [60, 55, 65, 75, 85, 90, 88, 80, 70, 62, 58, 68],
        "SKU-007": [100, 90, 110, 125, 140, 150, 145, 135, 120, 105, 95, 115],
        "SKU-008": [35, 30, 40, 45, 50, 55, 52, 48, 42, 36, 32, 40],
        "SKU-009": [300, 280, 320, 350, 380, 400, 390, 360, 330, 305, 285, 325],
        "SKU-010": [120, 110, 130, 145, 160, 170, 165, 155, 140, 125, 115, 135],
        "SKU-011": [75, 70, 80, 90, 100, 105, 102, 95, 85, 78, 72, 82],
        "SKU-012": [200, 190, 210, 230, 250, 260, 255, 240, 220, 205, 195, 215],
    }
    
    for sku, monthly_demand in base_demand.items():
        history[sku] = []
        
        for month_idx, demand in enumerate(monthly_demand):
            # Add some random variation
            actual_demand = int(demand * (0.9 + random.random() * 0.2))
            
            period = f"2024-{month_idx + 1:02d}"
            history[sku].append(DemandHistory(
                sku=sku,
                period=period,
                quantity=actual_demand,
                revenue=actual_demand * PRODUCTS[sku].selling_price if sku in PRODUCTS else 0,
                num_orders=random.randint(5, 25),
            ))
    
    return history


# =============================================================================
# Carriers
# =============================================================================

CARRIERS = {
    "CAR-001": Carrier(
        carrier_id="CAR-001",
        name="FedEx Freight",
        modes=[ShipmentMode.GROUND, ShipmentMode.AIR, ShipmentMode.EXPRESS],
        on_time_delivery_rate=94.5,
        damage_rate=0.8,
        domestic=True,
        international=True,
        api_enabled=True,
    ),
    "CAR-002": Carrier(
        carrier_id="CAR-002",
        name="XPO Logistics",
        modes=[ShipmentMode.GROUND, ShipmentMode.RAIL],
        on_time_delivery_rate=92.0,
        damage_rate=1.2,
        domestic=True,
        international=False,
        api_enabled=True,
    ),
    "CAR-003": Carrier(
        carrier_id="CAR-003",
        name="Maersk",
        modes=[ShipmentMode.OCEAN, ShipmentMode.INTERMODAL],
        on_time_delivery_rate=88.5,
        damage_rate=0.5,
        domestic=False,
        international=True,
        api_enabled=True,
    ),
    "CAR-004": Carrier(
        carrier_id="CAR-004",
        name="UPS Freight",
        modes=[ShipmentMode.GROUND, ShipmentMode.AIR],
        on_time_delivery_rate=93.0,
        damage_rate=1.0,
        domestic=True,
        international=True,
        api_enabled=True,
    ),
}


SHIPPING_RATES = [
    ShippingRate(rate_id="R-001", carrier_id="CAR-001", carrier_name="FedEx Freight",
                 mode=ShipmentMode.GROUND, service_level="Standard",
                 base_rate=25.00, rate_per_kg=0.45, rate_per_m3=35.00,
                 fuel_surcharge_pct=8.5, transit_days=5),
    ShippingRate(rate_id="R-002", carrier_id="CAR-001", carrier_name="FedEx Freight",
                 mode=ShipmentMode.EXPRESS, service_level="Express",
                 base_rate=75.00, rate_per_kg=1.85, rate_per_m3=120.00,
                 fuel_surcharge_pct=12.0, transit_days=2),
    ShippingRate(rate_id="R-003", carrier_id="CAR-002", carrier_name="XPO Logistics",
                 mode=ShipmentMode.GROUND, service_level="Economy",
                 base_rate=15.00, rate_per_kg=0.32, rate_per_m3=28.00,
                 fuel_surcharge_pct=6.5, transit_days=7),
    ShippingRate(rate_id="R-004", carrier_id="CAR-003", carrier_name="Maersk",
                 mode=ShipmentMode.OCEAN, service_level="Standard",
                 base_rate=200.00, rate_per_kg=0.08, rate_per_m3=45.00,
                 fuel_surcharge_pct=15.0, transit_days=28),
    ShippingRate(rate_id="R-005", carrier_id="CAR-004", carrier_name="UPS Freight",
                 mode=ShipmentMode.GROUND, service_level="Standard",
                 base_rate=22.00, rate_per_kg=0.42, rate_per_m3=32.00,
                 fuel_surcharge_pct=9.0, transit_days=5),
]


# =============================================================================
# Warehouses
# =============================================================================

WAREHOUSES = {
    "WH-MAIN": Warehouse(
        warehouse_id="WH-MAIN",
        name="Main Distribution Center",
        country="USA",
        city="Chicago",
        latitude=41.8781,
        longitude=-87.6298,
        total_capacity_m3=50000,
        used_capacity_m3=32500,
        receiving_docks=8,
        shipping_docks=12,
        storage_cost_per_m3=2.50,
        handling_cost_per_unit=0.75,
    ),
    "WH-WEST": Warehouse(
        warehouse_id="WH-WEST",
        name="West Coast Hub",
        country="USA",
        city="Los Angeles",
        latitude=34.0522,
        longitude=-118.2437,
        total_capacity_m3=35000,
        used_capacity_m3=21000,
        receiving_docks=6,
        shipping_docks=8,
        storage_cost_per_m3=3.00,
        handling_cost_per_unit=0.85,
    ),
    "WH-EAST": Warehouse(
        warehouse_id="WH-EAST",
        name="East Coast Facility",
        country="USA",
        city="Newark",
        latitude=40.7357,
        longitude=-74.1724,
        total_capacity_m3=40000,
        used_capacity_m3=28000,
        receiving_docks=6,
        shipping_docks=10,
        storage_cost_per_m3=3.50,
        handling_cost_per_unit=0.90,
    ),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_all_data() -> dict:
    """Get all sample data as a dictionary."""
    return {
        "products": PRODUCTS,
        "suppliers": SUPPLIERS,
        "supplier_quotes": SUPPLIER_QUOTES,
        "inventory": generate_inventory_levels(),
        "demand_history": generate_demand_history(),
        "carriers": CARRIERS,
        "shipping_rates": SHIPPING_RATES,
        "warehouses": WAREHOUSES,
    }
