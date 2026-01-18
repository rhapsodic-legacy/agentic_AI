# 🛍️ E-commerce Personal Shopper

A **CrewAI-powered** personal shopping assistant that understands preferences, searches across multiple retailers, compares prices, finds deals, and makes personalized recommendations.

![CrewAI](https://img.shields.io/badge/Framework-CrewAI-blue)
![Architecture](https://img.shields.io/badge/Architecture-Parallel-green)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐-yellow)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎨 **Style Matching** | Matches products to user preferences |
| 🔍 **Multi-Retailer Search** | Amazon, REI, Nordstrom, Target, etc. |
| 💰 **Price Comparison** | Finds best prices across retailers |
| 🏷️ **Deal Finding** | Coupons, sales, price drops |
| ⭐ **Review Analysis** | Summarizes pros, cons, fit info |
| ♻️ **Sustainability** | Filters for eco-friendly products |
| 📊 **Parallel Execution** | Multiple agents work simultaneously |

## 🏗️ Architecture (Parallel)

```
                    ┌─────────────────────┐
                    │   CONCIERGE         │
                    │   (Main Interface)  │
                    └──────────┬──────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   STYLE     │         │   SEARCH    │         │   DEALS     │
│   ADVISOR   │         │   AGENT     │         │   FINDER    │
└──────┬──────┘         └──────┬──────┘         └──────┬──────┘
       │                       │                       │
       │         ┌─────────────┴─────────────┐         │
       │         │                           │         │
       │         ▼                           ▼         │
       │   ┌─────────────┐           ┌─────────────┐   │
       │   │   PRICE     │           │   REVIEW    │   │
       │   │   COMPARE   │           │   ANALYZER  │   │
       │   └──────┬──────┘           └──────┬──────┘   │
       │          │                         │          │
       └──────────┴────────────┬────────────┴──────────┘
                               │
                               ▼
                        ┌─────────────┐
                        │ RECOMMENDER │
                        └─────────────┘
```

## 🤖 Shopping Team

| Agent | Role | Tools |
|-------|------|-------|
| 🎩 **Concierge** | Main interface, coordinates team | Search, Recommend |
| 👔 **Style Advisor** | Analyzes preferences, trends | Style Match |
| 🔍 **Search Agent** | Multi-retailer product search | Search, Details |
| 💰 **Deals Finder** | Finds coupons and sales | Deals, Coupons |
| 📊 **Price Compare** | Compares prices across retailers | Price Compare |
| ⭐ **Review Analyzer** | Summarizes customer reviews | Reviews, Sentiment |
| 🎁 **Recommender** | Final personalized picks | Recommend, Details |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key
export GOOGLE_API_KEY="your-key"
```

### CLI Usage

```bash
# Full shopping assistance
python main.py shop "winter jacket, sustainable, budget $200"

# Quick search
python main.py search "running shoes"

# Show deals
python main.py deals

# Interactive mode
python main.py interactive

# Web server
python main.py serve
```

### Python API

```python
from shopper import PersonalShopperCrew, shop

# Quick shopping
result = shop("winter jacket, sustainable, budget $200")

for rec in result.recommendations:
    print(f"{rec.rank}. {rec.product.name}")
    print(f"   ${rec.final_price:.2f}")
    print(f"   {', '.join(rec.match_reasons)}")

# Full crew control
shopper = PersonalShopperCrew()
result = shopper.shop(
    query="running shoes",
    budget=150,
    sustainable_only=True,
)

# Access deals
for deal in result.deals:
    print(f"🏷️ {deal.description} - code: {deal.code}")
```

## 📁 Project Structure

```
personal-shopper/
├── shopper/
│   ├── __init__.py           # Package exports
│   ├── crew.py               # Main PersonalShopperCrew
│   ├── agents/
│   │   └── __init__.py       # 7 specialized agents
│   ├── tools/
│   │   └── __init__.py       # 10 shopping tools
│   ├── models/
│   │   └── __init__.py       # Product, Deal, Recommendation
│   └── data/
│       └── __init__.py       # Mock product database
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React shopping UI
├── main.py                    # CLI
├── requirements.txt
└── README.md
```

## 💬 Sample Interaction

```
User: "I need a new winter jacket, budget around $200, prefer sustainable brands"

🎩 Concierge: "I'll have my team search for sustainable winter jackets..."

[Parallel execution]
👔 Style Advisor: Analyzes trends, sustainable materials
🔍 Search Agent: Queries Patagonia, REI, North Face, etc.
💰 Deals Finder: Checks current sales, coupon codes
📊 Price Compare: Finds best prices
⭐ Review Analyzer: Summarizes customer feedback

🎁 Recommender: "Here are my top picks:

1. 🥇 Patagonia Nano Puff - $199 (♻️ Recycled materials)
   ⭐ 4.8/5 (2,340 reviews) | "Warm, packable, great for layering"
   🏷️ 20% off with code WINTER20 → $159.20

2. 🥈 REI Co-op Stormhenge - $179 (♻️ bluesign certified)
   ⭐ 4.6/5 (892 reviews) | "Excellent waterproofing"
   🏷️ Member price: $161.10

3. 🥉 Cotopaxi Fuego Down - $185 (♻️ Responsible Down)
   ⭐ 4.7/5 (567 reviews) | "Unique colorways, very warm"

Would you like more details on any of these?"
```

## 🔧 Tools Reference

### Search Tools
```python
search_products(query, max_price, sustainable_only)
search_by_category(category, max_price)
get_product_details(product_id)
```

### Deal Tools
```python
find_deals(retailer, category)
get_deals_for_product(product_id)
compare_prices(product_name, brand)
```

### Review Tools
```python
get_reviews(product_id)
analyze_review_sentiment(product_id)
```

### Style Tools
```python
match_style(preferences, category)
generate_recommendation(product_id)
```

## 🛒 Mock Retailers

| Retailer | Categories |
|----------|------------|
| Amazon | All |
| REI | Outdoor, Sports |
| Patagonia | Outdoor |
| Nordstrom | Clothing, Accessories |
| Nike | Footwear, Sports |
| Target | Home, Clothing |
| Best Buy | Electronics |

## ⚙️ Configuration

```python
from shopper import PersonalShopperCrew, ShopperConfig

config = ShopperConfig(
    llm_provider="gemini",     # "gemini", "anthropic", "openai"
    llm_model=None,            # Uses provider default
    verbose=True,
    parallel_execution=True,
)

shopper = PersonalShopperCrew(config)
```

## 🌐 Web API

```bash
python main.py serve
```

Endpoints:
- `POST /api/shop` - Full shopping assistance (async)
- `GET /api/shop/{job_id}` - Get shopping results
- `POST /api/search` - Quick product search
- `GET /api/product/{id}` - Product details
- `GET /api/deals` - Current deals

## 📊 Output Format

```python
ShoppingResult(
    query="winter jacket",
    recommendations=[
        Recommendation(
            product=Product(...),
            final_price=159.20,
            match_reasons=["Sustainable", "Highly rated"],
            applicable_deals=[Deal(code="WINTER20", ...)],
            review_summary=ReviewSummary(...),
            rank=1,
        ),
        ...
    ],
    deals=[Deal(...), ...],
    price_range=(110.0, 220.0),
    best_value=Recommendation(...),
    total_products_searched=50,
    retailers_searched=["rei", "amazon", "nordstrom"],
)
```

## 📝 License

MIT License
