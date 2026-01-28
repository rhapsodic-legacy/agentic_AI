# 📦 Supply Chain Optimizer

An **end-to-end supply chain optimization system** using a hierarchical multi-agent architecture. Handles demand forecasting, inventory management, supplier selection, and logistics planning.

![CrewAI](https://img.shields.io/badge/Framework-CrewAI-blue)
![Architecture](https://img.shields.io/badge/Architecture-Hierarchical-purple)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐⭐⭐-yellow)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Demand Forecasting** | ML-based forecasting with seasonality detection |
| 📦 **Inventory Optimization** | EOQ, safety stock, ABC classification |
| 🏭 **Supplier Management** | Scoring, selection, risk assessment |
| 🚛 **Logistics Planning** | Route optimization, carrier selection |
| 🔮 **Scenario Analysis** | What-if demand shocks and supply disruptions |
| 📈 **KPI Dashboard** | Real-time metrics and reporting |

## 🏗️ Hierarchical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SUPPLY CHAIN DIRECTOR                        │
│              (Strategic decisions, KPI monitoring)              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    DEMAND       │ │   INVENTORY     │ │   LOGISTICS     │
│    PLANNING     │ │   MANAGEMENT    │ │   PLANNING      │
│    MANAGER      │ │   MANAGER       │ │   MANAGER       │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
    ┌────┴────┐         ┌────┴────┐         ┌────┴────┐
    ▼         ▼         ▼         ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│Forecast│ │Market │ │Replen-│ │Safety │ │Route  │ │Carrier│
│Agent   │ │Analyst│ │ishment│ │Stock  │ │Optim- │ │Select-│
│        │ │       │ │Agent  │ │Agent  │ │izer   │ │or     │
└───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
```

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Optional: Set API key for enhanced AI capabilities
export GOOGLE_API_KEY="your-key"
```

### Run Optimization

```bash
# Full optimization with report
python main.py optimize --save

# Analyze current state
python main.py analyze

# Run demand surge scenario
python main.py scenario --demand-change 30

# Run supplier disruption scenario
python main.py scenario --supplier-disruption SUP-001 --disruption-days 45
```

### Web Dashboard

```bash
python main.py serve
# Open http://localhost:8000
```

### Python API

```python
from supply_chain import SupplyChainOptimizer, OptimizationConfig

# Configure optimization
config = OptimizationConfig(
    target_service_level_a=0.98,
    target_service_level_b=0.95,
    target_service_level_c=0.90,
    forecast_periods=3,
)

# Run optimization
optimizer = SupplyChainOptimizer(config)
result = optimizer.optimize()

# Generate report
report = optimizer.generate_report(result)
print(report)
```

## 📁 Project Structure

```
supply-chain-optimizer/
├── supply_chain/
│   ├── __init__.py           # Main optimizer engine
│   ├── agents/               # CrewAI agent definitions
│   │   └── __init__.py       # Strategic, Tactical, Operational agents
│   ├── models/               # Data models
│   │   └── __init__.py       # Products, Suppliers, Inventory, etc.
│   ├── tools/                # Optimization tools
│   │   └── __init__.py       # Forecasting, EOQ, Safety Stock, etc.
│   └── data/                 # Sample data
│       └── __init__.py       # Products, Suppliers, Carriers
├── frontend/
│   └── index.html            # React dashboard
├── main.py                   # Rich CLI
├── api.py                    # FastAPI backend
├── requirements.txt
└── README.md
```

## 🤖 Agent Roles

### Strategic Level

| Agent | Role |
|-------|------|
| **Supply Chain Director** | Overall strategy, KPI monitoring, resource allocation |

### Tactical Level

| Agent | Role |
|-------|------|
| **Demand Planning Manager** | Coordinate forecasting and market analysis |
| **Inventory Manager** | Optimize stock levels and replenishment |
| **Logistics Manager** | Plan routes and select carriers |

### Operational Level

| Agent | Role |
|-------|------|
| **Forecast Agent** | Statistical demand forecasting |
| **Market Analyst** | External factor analysis |
| **Replenishment Agent** | Order quantity optimization |
| **Safety Stock Agent** | Service level optimization |
| **Route Optimizer** | Delivery route planning |
| **Carrier Selector** | Rate comparison and selection |

## 📊 Optimization Pipeline

### Phase 1: Demand Planning
- Analyze 12 months of historical demand
- Detect trends and seasonality
- Generate quarterly forecasts
- Apply market adjustments

### Phase 2: Inventory Management
- Calculate safety stock levels
- Determine reorder points
- Generate replenishment recommendations
- Select optimal suppliers

### Phase 3: Logistics Planning
- Consolidate shipments
- Optimize delivery routes
- Select carriers by cost/service
- Calculate carbon footprint

### Phase 4: Strategic Review
- Assess supply chain risks
- Generate executive report
- Identify improvement opportunities
- Prioritize actions

## 📈 Sample Output

```markdown
# Supply Chain Optimization Report

## Executive Summary

| Metric | Current | Optimized | Change |
|--------|---------|-----------|--------|
| Inventory Value | $2.4M | $2.1M | -12% |
| Stockout Risk | 15% | 3% | -12pp |
| Shipping Routes | 12 | 8 | -33% |
| Carbon Footprint | Baseline | - | -22% |

**Working Capital Freed:** $300,000
**Monthly Shipping Savings:** $45,000

## Demand Forecast (Next Quarter)

| Product | Current Stock | Forecast | Recommended Order |
|---------|--------------|----------|-------------------|
| SKU-001 | 500 | 2,400 | 2,100 |
| SKU-002 | 1,200 | 3,100 | 2,000 |
| SKU-003 | 300 | 800 | 600 |

## Supplier Analysis

| Supplier | Score | Lead Time | Cost | Risk |
|----------|-------|-----------|------|------|
| TechParts International | 92 | 21 days | Low | 🟢 |
| Midwest Manufacturing | 88 | 10 days | Medium | 🟢 |
| Dragon Industries | 78 | 35 days | Low | 🔴 |

## Logistics Optimization

- **Routes Optimized:** 12 → 8 (33% reduction)
- **Shipping Cost Savings:** $45K/month
- **Carbon Footprint Reduction:** 22%
```

## 🔧 Key Algorithms

### Demand Forecasting
- **Moving Average**: Simple trend following
- **Exponential Smoothing**: Recent data emphasis
- **Seasonal Decomposition**: Pattern detection

### Inventory Optimization
- **EOQ (Economic Order Quantity)**: √(2DS/H)
- **Safety Stock**: Z × √(LT × σd² + d² × σLT²)
- **ABC Classification**: Pareto analysis (80/15/5)

### Supplier Scoring
- Quality (30%) + Delivery (30%) + Cost (25%) + Responsiveness (15%)

### Route Optimization
- Nearest neighbor heuristic
- Shipment consolidation
- Multi-modal comparison

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System status |
| `/api/products` | GET | All products |
| `/api/suppliers` | GET | Supplier scorecard |
| `/api/inventory` | GET | Inventory levels |
| `/api/kpis` | GET | Key performance indicators |
| `/api/optimize` | POST | Start optimization |
| `/api/optimize/{id}` | GET | Get optimization status |
| `/api/scenario` | POST | Run what-if scenario |
| `/api/recommendations` | GET | Replenishment recommendations |
| `/api/hierarchy` | GET | Agent hierarchy |

## 🔮 Scenario Analysis

### Demand Shock
```python
from supply_chain import create_demand_shock_scenario

# +30% demand surge
scenario = create_demand_shock_scenario(30)
results = optimizer.run_scenario(scenario)
```

### Supply Disruption
```python
from supply_chain import create_supply_disruption_scenario

# 45-day supplier outage
scenario = create_supply_disruption_scenario("SUP-001", 45)
results = optimizer.run_scenario(scenario)
```

## 📦 Sample Data

The system includes realistic sample data:

- **12 Products**: Industrial components (motors, bearings, sensors)
- **8 Suppliers**: Global suppliers with varying performance
- **4 Carriers**: FedEx, XPO, Maersk, UPS
- **3 Warehouses**: Chicago, LA, Newark

## ⚙️ Configuration

```python
from supply_chain import OptimizationConfig

config = OptimizationConfig(
    # Forecasting
    forecast_periods=3,
    forecast_method="exponential_smoothing",
    
    # Service levels by ABC class
    target_service_level_a=0.98,  # Class A: 98%
    target_service_level_b=0.95,  # Class B: 95%
    target_service_level_c=0.90,  # Class C: 90%
    
    # Costs
    ordering_cost=50.0,           # Per order
    holding_cost_pct=0.25,        # 25% of item cost
    
    # Procurement
    min_supplier_score=70.0,
    max_supplier_risk="MEDIUM",
    
    # Logistics
    max_transit_days=10,
    consolidation_min_savings_pct=0.10,
    
    # Output
    output_dir="output",
    verbose=True,
)
```

## 📊 KPIs Tracked

### Inventory
- Inventory Turnover
- Days of Supply
- Carrying Cost %
- Fill Rate

### Suppliers
- On-Time Delivery
- Quality Score
- Cost Position
- Risk Level

### Logistics
- Freight Cost/Unit
- Route Efficiency
- Carbon Footprint
- Perfect Order Rate

## 🔒 Risk Monitoring

The system monitors:
- **Financial Risk**: Supplier financial stability
- **Geopolitical Risk**: Regional instability
- **Concentration Risk**: Over-reliance on single supplier
- **Performance Risk**: Quality and delivery trends

## 📝 License

MIT License

---

*Optimize your supply chain with AI-powered intelligence!* 📦🚀
