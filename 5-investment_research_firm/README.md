# 💰 Investment Research Firm

A full investment research operation using **CrewAI** with specialized analysts covering different domains, quantitative researchers, and a portfolio manager who synthesizes recommendations.

![CrewAI](https://img.shields.io/badge/Framework-CrewAI-blue)
![Architecture](https://img.shields.io/badge/Architecture-Hub%20and%20Spoke-green)
![Complexity](https://img.shields.io/badge/Complexity-⭐⭐⭐⭐⭐-yellow)

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Real-time Market Data** | Yahoo Finance integration |
| 📈 **Technical Analysis** | Moving averages, RSI, MACD, Bollinger Bands |
| 💹 **Fundamental Analysis** | Valuation, profitability, growth metrics |
| 📉 **DCF Valuation** | Discounted cash flow with scenarios |
| 📰 **Sentiment Analysis** | News-based sentiment scoring |
| ⚠️ **Risk Assessment** | Comprehensive risk factor analysis |
| 📝 **Investment Memos** | Professional research reports |
| 🤖 **8 Specialized Agents** | Full research team simulation |

## 🏗️ Architecture

```
                    ┌─────────────────────┐
                    │  RESEARCH DIRECTOR  │
                    │  (Assigns & Reviews)│
                    └──────────┬──────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
┌─────────┐              ┌─────────┐              ┌─────────┐
│ MACRO   │              │ EQUITY  │              │ QUANT   │
│ ANALYST │              │ ANALYST │              │RESEARCHER│
└─────────┘              └────┬────┘              └─────────┘
                              │
                    ┌─────────┴─────────┐
                    │   SECTOR ANALYSTS │
                    │  Tech │ Finance │ │
                    │     Healthcare   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  PORTFOLIO MANAGER  │
                    │  (Final Decisions)  │
                    └─────────────────────┘
```

## 🤖 Research Team

| Agent | Role | Tools |
|-------|------|-------|
| 📊 **Research Director** | Coordinates research, assigns tasks | All tools |
| 🌍 **Macro Analyst** | Economic context, sector analysis | News, Sentiment |
| 📈 **Equity Analyst** | Fundamental stock analysis | Financials, DCF |
| 💻 **Tech Sector Analyst** | Technology specialist | Sector comparison |
| 🏦 **Finance Sector Analyst** | Financial services expert | Sector comparison |
| 🏥 **Healthcare Analyst** | Healthcare & biotech | Sector comparison |
| 🔢 **Quant Researcher** | Technical & quantitative | Technical, Risk |
| 💼 **Portfolio Manager** | Final recommendations | Risk assessment |

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# Set API key for LLM provider
export GOOGLE_API_KEY="your-key"  # For Gemini (recommended)
```

### CLI Usage

```bash
# Full research report
python main.py research NVDA

# Screen multiple stocks
python main.py screen AAPL,MSFT,GOOGL,AMZN,NVDA

# Quick analysis
python main.py analyze NVDA --type technical
python main.py analyze NVDA --type fundamental
python main.py analyze NVDA --type valuation

# Interactive mode
python main.py interactive

# Web server
python main.py serve
```

### Python API

```python
from investment_firm import InvestmentResearchCrew, ResearchConfig

# Create research crew
crew = InvestmentResearchCrew()

# Full research (runs all agents)
result = crew.research_stock("NVDA")

# Get the investment memo
print(result.memo.to_markdown())

# Save as HTML report
from investment_firm.reports import ReportGenerator
generator = ReportGenerator("./output")
generator.save(result.memo, format="html")

# Quick screening (faster, no full crew)
results = crew.screen_stocks(["AAPL", "MSFT", "GOOGL"])
for r in results:
    print(f"{r['symbol']}: {r['upside']}")
```

## 📁 Project Structure

```
investment-research/
├── investment_firm/
│   ├── __init__.py           # Package exports
│   ├── crew.py               # Main InvestmentResearchCrew
│   ├── agents/
│   │   └── __init__.py       # 8 specialized agents
│   ├── tools/
│   │   └── __init__.py       # 10 CrewAI tools
│   ├── models/
│   │   └── __init__.py       # Stock, Memo, Analysis models
│   ├── data_sources/
│   │   └── __init__.py       # Yahoo Finance, mock data
│   ├── analysis/
│   │   └── __init__.py       # Technical, Fundamental, Sentiment
│   └── reports/
│       └── __init__.py       # HTML/Markdown generation
├── api.py                     # FastAPI backend
├── frontend/
│   └── index.html            # React dashboard
├── main.py                    # CLI
├── requirements.txt
└── README.md
```

## 🔧 Analysis Tools

### Market Data
```python
get_stock_info(symbol)      # Comprehensive stock data
get_price_history(symbol)   # Historical prices
get_financials(symbol)      # Financial statements
get_news(symbol)            # Recent news
```

### Analysis
```python
technical_analysis(symbol)      # Technical indicators
fundamental_analysis(symbol)    # Fundamental metrics
dcf_valuation(symbol)          # DCF fair value
sentiment_analysis(symbol)     # News sentiment
risk_assessment(symbol)        # Risk factors
compare_stocks(symbols)        # Peer comparison
```

## 📊 Sample Output: Investment Memo

```markdown
# Investment Memo: NVDA (NVIDIA Corporation)

## Recommendation: BUY

**Target Price:** $950 | **Current:** $875 | **Upside:** 8.6%

## Executive Summary
NVIDIA presents an attractive investment opportunity with significant 
upside based on DCF analysis. The company scores 78/100 on our 
fundamental quality assessment...

## Key Metrics
| Metric | Value | Industry Avg |
|--------|-------|--------------|
| P/E Ratio | 65.2x | 25.0x |
| Forward P/E | 32.5x | 20.0x |
| Revenue Growth | 122% | 15% |
| Gross Margin | 74% | 45% |
| ROE | 91% | 15% |

## Price Target Scenarios
| Scenario | Price | Upside |
|----------|-------|--------|
| Bull Case | $1,100 | +25% |
| Base Case | $950 | +8.6% |
| Bear Case | $700 | -20% |

## Catalyst Timeline
- **Next Quarter**: Earnings release
- **Next 6 Months**: New GPU architecture

## Risk Factors
1. **High Valuation** - P/E of 65x is elevated
2. **Semiconductor Cycle** - Industry cyclicality
3. **Competition** - AMD/Intel threats
```

## 📈 Data Sources

```python
DATA_SOURCES = {
    "market_data": ["Yahoo Finance"],
    "fundamentals": ["Yahoo Finance (financials)"],
    "news": ["Yahoo Finance News"],
    "mock": ["Built-in mock data for demo"],
}
```

## ⚙️ Configuration

```python
from investment_firm import InvestmentResearchCrew, ResearchConfig

config = ResearchConfig(
    llm_provider="gemini",     # "gemini", "anthropic", "openai"
    llm_model=None,            # Uses provider default
    use_live_data=True,        # Use Yahoo Finance
    verbose=True,              # Show agent activity
    max_iterations=10,         # Max agent iterations
)

crew = InvestmentResearchCrew(config)
```

## 🌐 Web API

```bash
python main.py serve
```

Endpoints:
- `GET /api/stock/{symbol}` - Stock information
- `GET /api/stock/{symbol}/prices` - Price history
- `GET /api/stock/{symbol}/technical` - Technical analysis
- `GET /api/stock/{symbol}/fundamental` - Fundamental analysis
- `GET /api/stock/{symbol}/valuation` - DCF valuation
- `POST /api/screen` - Screen multiple stocks
- `POST /api/research` - Start full research (async)
- `GET /api/research/{symbol}/status` - Research status

## 📋 Investment Process

1. **Research Director** assigns the research task
2. **Macro Analyst** provides economic context
3. **Equity Analyst** performs fundamental analysis
4. **Sector Analyst** provides industry perspective
5. **Quant Researcher** does technical & risk analysis
6. **Portfolio Manager** synthesizes and recommends

## 🧪 Testing Without API Keys

The system includes mock data for testing:

```python
from investment_firm import MarketDataManager

# Use mock data
dm = MarketDataManager()
stock = dm.get_stock("NVDA", use_live=False)  # Uses mock
```

Mock data includes: NVDA, AAPL, MSFT, JPM, JNJ

## 📝 License

MIT License

---

*Disclaimer: This is for educational purposes only and not investment advice.*
