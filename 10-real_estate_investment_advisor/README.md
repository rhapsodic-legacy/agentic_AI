# 🏠 Real Estate Investment Advisor

An **AI advisor** that analyzes real estate markets, evaluates properties, calculates ROI, and provides investment recommendations using a **Supervisor Pattern** architecture.

![FastAPI](https://img.shields.io/badge/Framework-FastAPI+LangChain-green)
![Architecture](https://img.shields.io/badge/Architecture-Supervisor_Pattern-blue)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐⭐-yellow)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Market Analysis** | Demographics, trends, supply/demand |
| 🏠 **Property Valuation** | Comparable sales, condition assessment |
| 💰 **Financial Modeling** | Cash flow, ROI/IRR, cap rate, GRM |
| ⚠️ **Risk Assessment** | Vacancy, appreciation, market risks |
| ⚖️ **Legal Checking** | Zoning, permits, title verification |
| 📈 **5-Year Projections** | Year-by-year cash flow forecasts |
| 🎯 **Recommendations** | Buy/Hold/Avoid with reasoning |

## 🏗️ Supervisor Pattern Architecture

```
                    ┌─────────────────────┐
                    │    SUPERVISOR       │
                    │  (Query Router)     │
                    └──────────┬──────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   MARKET    │         │  PROPERTY   │         │  FINANCIAL  │
│   ANALYST   │         │  EVALUATOR  │         │   MODELER   │
│             │         │             │         │             │
│ Demographics│         │ Comparables │         │ Cash flow   │
│ Trends      │         │ Condition   │         │ ROI/IRR     │
│ Supply/demand│        │ Valuation   │         │ Cap rate    │
└─────────────┘         └─────────────┘         └─────────────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
             ┌─────────────┐       ┌─────────────┐
             │    RISK     │       │    LEGAL    │
             │   ASSESSOR  │       │   CHECKER   │
             └─────────────┘       └─────────────┘
```

## 📍 Available Markets

| City | State | Investment Score | Market Type |
|------|-------|------------------|-------------|
| Austin | TX | 82/100 | Seller's Market |
| Tampa | FL | 85/100 | Seller's Market |
| Nashville | TN | 80/100 | Seller's Market |
| Phoenix | AZ | 78/100 | Balanced |
| Atlanta | GA | 77/100 | Balanced |
| Dallas | TX | 75/100 | Balanced |
| Denver | CO | 72/100 | Balanced |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key (optional, for enhanced analysis)
export GOOGLE_API_KEY="your-key"
```

### CLI Usage

```bash
# Search for properties
python main.py search Austin --max-price 500000

# Analyze a property
python main.py analyze prop-abc123
python main.py analyze prop-abc123 --down-payment 20 --rate 6.5 -o report.md

# Market analysis
python main.py market Austin

# Interactive mode
python main.py interactive

# Web server
python main.py serve
```

### Python API

```python
from real_estate import RealEstateAdvisor, analyze_address

# Create advisor
advisor = RealEstateAdvisor()

# Search for properties
properties = advisor.search_properties(city="Austin", max_price=500000)

# Analyze a property
analysis = advisor.analyze_property(properties[0])
print(analysis.to_markdown())

# Quick analysis by address
analysis = analyze_address(
    address="123 Main St",
    city="Austin",
    state="TX",
    price=450000,
    bedrooms=3,
    bathrooms=2,
    sqft=1850,
)
print(f"Recommendation: {analysis.recommendation.recommendation.value}")
```

## 📁 Project Structure

```
real-estate-advisor/
├── real_estate/
│   ├── __init__.py           # Package exports
│   ├── advisor.py            # Main RealEstateAdvisor class
│   ├── agents/
│   │   └── __init__.py       # 5 specialist agents + Supervisor
│   ├── tools/
│   │   └── __init__.py       # Analysis tools (valuation, financial, risk)
│   ├── models/
│   │   └── __init__.py       # Property, Market, Financial models
│   └── data/
│       └── __init__.py       # Mock data sources (Zillow, Census, etc.)
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React dashboard
├── main.py                    # Rich CLI
├── requirements.txt
└── README.md
```

## 📊 Sample Analysis Output

```markdown
# Investment Analysis: 123 Main St, Austin, TX

## Property Overview
- **List Price:** $450,000
- **Type:** Single Family, 3BR/2BA
- **Size:** 1,850 sqft | Lot: 6,500 sqft
- **Year Built:** 2015
- **Fair Value Estimate:** $435,000 - $465,000 ✓

## Financial Projections
| Metric | Value | Market Avg |
|--------|-------|------------|
| Monthly Rent | $2,800 | $2,650 |
| Cap Rate | 5.2% | 4.8% |
| Cash-on-Cash (25% down) | 7.1% | 5.5% |
| GRM | 13.4 | 14.2 |

## 5-Year Pro Forma
| Year | NOI | Appreciation | Total Return |
|------|-----|--------------|--------------|
| 1 | $24,500 | $18,000 | $42,500 |
| 2 | $25,200 | $18,500 | $43,700 |
| ... | ... | ... | ... |

## Risk Assessment
- 🟢 Market Growth: Strong (+4.2% YoY population)
- 🟢 Employment: Tech hub, diverse economy
- 🟡 Vacancy Risk: 5.2% historical
- 🟡 Appreciation: Slowing from 15% to 5%
- 🟢 Rent Growth: +6% projected

## Recommendation
**BUY** at list price or below
- Strong rental market supports cash flow
- Good long-term appreciation potential
- Below-market cap rate available
```

## 🔢 Key Metrics Calculated

| Metric | Formula | Target |
|--------|---------|--------|
| **Cap Rate** | NOI / Purchase Price | >5% |
| **Cash-on-Cash** | Annual Cash Flow / Total Investment | >7% |
| **GRM** | Purchase Price / Annual Gross Rent | <15 |
| **DSCR** | NOI / Annual Debt Service | >1.25 |
| **IRR** | Internal Rate of Return (5-year) | >12% |

## 🌐 Web API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/markets` | GET | List available markets |
| `/api/market/{city}` | GET | Market analysis |
| `/api/properties/{city}` | GET | Properties in city |
| `/api/property/{id}` | GET | Property details |
| `/api/analyze` | POST | Analyze property (async) |
| `/api/analysis/{job_id}` | GET | Get analysis status |
| `/api/custom-property` | POST | Create custom property |

## ⚙️ Configuration

```python
from real_estate import RealEstateAdvisor, AdvisorConfig

config = AdvisorConfig(
    llm_provider="gemini",        # "gemini", "anthropic", "openai"
    default_down_payment=0.25,    # 25% down
    default_interest_rate=0.07,   # 7% rate
    verbose=True,
)

advisor = RealEstateAdvisor(config)
```

## 📈 Data Sources (Mock)

| Category | Sources |
|----------|---------|
| **Listings** | Zillow, Redfin, Realtor.com |
| **Rentals** | Zillow Rentals, Apartments.com, Rentometer |
| **Demographics** | Census Bureau, BLS |
| **Economics** | FRED, Local employment data |
| **Permits** | Local building department APIs |

## 📝 License

MIT License

*Disclaimer: This is for educational purposes only and not investment advice.*
