"""
Investment Research Firm - Data Models

Models for stocks, financial data, analysis results, and investment memos.
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
import json


class Recommendation(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class Sector(Enum):
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    CONSUMER = "consumer"
    ENERGY = "energy"
    INDUSTRIALS = "industrials"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMUNICATIONS = "communications"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class StockPrice:
    """Stock price data point."""
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


@dataclass
class Stock:
    """Stock information."""
    symbol: str
    name: str
    sector: Optional[Sector] = None
    industry: Optional[str] = None
    
    # Current data
    current_price: float = 0.0
    market_cap: float = 0.0
    
    # Price history
    price_history: list[StockPrice] = field(default_factory=list)
    
    # Fundamentals
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    
    # Growth metrics
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    
    # Margins
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    
    # Financial health
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    
    # Dividends
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    
    # Returns
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    
    # Analyst data
    analyst_target_price: Optional[float] = None
    analyst_rating: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "sector": self.sector.value if self.sector else None,
            "current_price": self.current_price,
            "market_cap": self.market_cap,
            "pe_ratio": self.pe_ratio,
            "revenue_growth": self.revenue_growth,
        }


@dataclass
class FinancialStatement:
    """Financial statement data."""
    period: str  # "2024-Q1", "2023-FY"
    type: str  # "income", "balance", "cashflow"
    
    # Income statement
    revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_expenses: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    eps_diluted: Optional[float] = None
    
    # Balance sheet
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    cash: Optional[float] = None
    total_debt: Optional[float] = None
    
    # Cash flow
    operating_cashflow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    free_cashflow: Optional[float] = None


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators."""
    symbol: str
    date: str
    
    # Moving averages
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None
    
    # Momentum
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    
    # Volatility
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    atr_14: Optional[float] = None
    
    # Volume
    obv: Optional[float] = None
    volume_sma_20: Optional[float] = None
    
    # Trend
    adx: Optional[float] = None
    
    # Support/Resistance
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    
    # Signals
    trend: str = "neutral"  # bullish, bearish, neutral
    signals: list[str] = field(default_factory=list)


@dataclass
class SentimentAnalysis:
    """Sentiment analysis results."""
    symbol: str
    date: str
    
    # Overall sentiment
    overall_score: float = 0.0  # -1 to 1
    overall_label: str = "neutral"  # positive, negative, neutral
    
    # News sentiment
    news_score: float = 0.0
    news_articles_analyzed: int = 0
    
    # Social sentiment
    social_score: float = 0.0
    social_posts_analyzed: int = 0
    
    # Key topics
    bullish_topics: list[str] = field(default_factory=list)
    bearish_topics: list[str] = field(default_factory=list)
    
    # Notable mentions
    notable_news: list[dict] = field(default_factory=list)


@dataclass
class MacroAnalysis:
    """Macroeconomic analysis."""
    date: str
    
    # Economic indicators
    gdp_growth: Optional[float] = None
    inflation_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None
    interest_rate: Optional[float] = None
    
    # Market indicators
    sp500_level: Optional[float] = None
    vix: Optional[float] = None
    yield_10y: Optional[float] = None
    yield_2y: Optional[float] = None
    
    # Outlook
    economic_outlook: str = "neutral"
    sector_rotation: list[str] = field(default_factory=list)
    risk_factors: list[str] = field(default_factory=list)


@dataclass
class ValuationModel:
    """Valuation model results."""
    symbol: str
    model_type: str  # "dcf", "comparables", "dividend_discount"
    
    # Inputs
    assumptions: dict = field(default_factory=dict)
    
    # Results
    fair_value: float = 0.0
    upside_potential: float = 0.0
    
    # Scenarios
    bull_case: float = 0.0
    base_case: float = 0.0
    bear_case: float = 0.0
    
    # Sensitivity
    sensitivity_analysis: dict = field(default_factory=dict)


@dataclass
class Catalyst:
    """Investment catalyst."""
    title: str
    expected_date: str
    description: str
    impact: str  # positive, negative, unknown
    probability: str  # high, medium, low


@dataclass
class RiskFactor:
    """Investment risk factor."""
    category: str  # market, company, regulatory, competitive, etc.
    title: str
    description: str
    severity: str  # high, medium, low
    mitigation: Optional[str] = None


@dataclass
class InvestmentThesis:
    """Investment thesis."""
    summary: str
    key_points: list[str]
    competitive_advantage: str
    growth_drivers: list[str]
    concerns: list[str]


@dataclass
class InvestmentMemo:
    """Complete investment memo."""
    # Header
    symbol: str
    company_name: str
    date: str
    analyst: str
    
    # Recommendation
    recommendation: Recommendation
    target_price: float
    current_price: float
    upside: float
    
    # Time horizon
    time_horizon: str = "12 months"
    
    # Thesis
    thesis: Optional[InvestmentThesis] = None
    
    # Analysis
    fundamental_analysis: Optional[dict] = None
    technical_analysis: Optional[TechnicalIndicators] = None
    sentiment_analysis: Optional[SentimentAnalysis] = None
    valuation: Optional[ValuationModel] = None
    
    # Catalysts & Risks
    catalysts: list[Catalyst] = field(default_factory=list)
    risks: list[RiskFactor] = field(default_factory=list)
    
    # Key metrics comparison
    key_metrics: dict = field(default_factory=dict)
    peer_comparison: dict = field(default_factory=dict)
    
    # Price targets
    bull_case_price: float = 0.0
    base_case_price: float = 0.0
    bear_case_price: float = 0.0
    
    # Portfolio fit
    risk_level: RiskLevel = RiskLevel.MEDIUM
    position_size_recommendation: str = ""
    
    # Summary
    executive_summary: str = ""
    
    def to_markdown(self) -> str:
        """Convert to markdown format."""
        upside_pct = self.upside * 100
        
        md = f"""# Investment Memo: {self.symbol} ({self.company_name})

## Recommendation: {self.recommendation.value.upper().replace('_', ' ')}

**Target Price:** ${self.target_price:.2f} | **Current:** ${self.current_price:.2f} | **Upside:** {upside_pct:.1f}%

**Analyst:** {self.analyst} | **Date:** {self.date} | **Time Horizon:** {self.time_horizon}

---

## Executive Summary

{self.executive_summary}

---

## Investment Thesis

{self.thesis.summary if self.thesis else "N/A"}

### Key Points
"""
        if self.thesis:
            for point in self.thesis.key_points:
                md += f"- {point}\n"
            
            md += f"""
### Competitive Advantage
{self.thesis.competitive_advantage}

### Growth Drivers
"""
            for driver in self.thesis.growth_drivers:
                md += f"- {driver}\n"
        
        # Key Metrics
        md += """
---

## Key Metrics

| Metric | Value | Industry Avg |
|--------|-------|--------------|
"""
        for metric, data in self.key_metrics.items():
            value = data.get("value", "N/A")
            industry = data.get("industry_avg", "N/A")
            md += f"| {metric} | {value} | {industry} |\n"
        
        # Catalysts
        md += """
---

## Catalyst Timeline

"""
        for cat in self.catalysts:
            md += f"- **{cat.expected_date}**: {cat.title}\n  - {cat.description}\n"
        
        # Risks
        md += """
---

## Risk Factors

"""
        for i, risk in enumerate(self.risks, 1):
            md += f"{i}. **{risk.title}** ({risk.severity} severity)\n   - {risk.description}\n"
        
        # Price Targets
        md += f"""
---

## Price Target Scenarios

| Scenario | Price | Upside/Downside |
|----------|-------|-----------------|
| Bull Case | ${self.bull_case_price:.2f} | {((self.bull_case_price/self.current_price)-1)*100:.1f}% |
| Base Case | ${self.base_case_price:.2f} | {((self.base_case_price/self.current_price)-1)*100:.1f}% |
| Bear Case | ${self.bear_case_price:.2f} | {((self.bear_case_price/self.current_price)-1)*100:.1f}% |

---

## Portfolio Considerations

- **Risk Level:** {self.risk_level.value.title()}
- **Position Size:** {self.position_size_recommendation}

---

*Report generated by AI Investment Research Firm*
"""
        return md
    
    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "date": self.date,
            "recommendation": self.recommendation.value,
            "target_price": self.target_price,
            "current_price": self.current_price,
            "upside": self.upside,
            "risk_level": self.risk_level.value,
        }


@dataclass 
class PortfolioRecommendation:
    """Portfolio-level recommendation."""
    date: str
    
    # Overall allocation
    equity_allocation: float = 0.6
    fixed_income_allocation: float = 0.3
    cash_allocation: float = 0.1
    
    # Sector weights
    sector_weights: dict = field(default_factory=dict)
    
    # Top picks
    top_picks: list[InvestmentMemo] = field(default_factory=list)
    
    # Sells
    sell_recommendations: list[str] = field(default_factory=list)
    
    # Market outlook
    market_outlook: str = ""
    risk_assessment: str = ""
    
    # Performance attribution
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
