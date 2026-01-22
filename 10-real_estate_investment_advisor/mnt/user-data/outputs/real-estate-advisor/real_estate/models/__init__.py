"""
Real Estate Investment Advisor - Data Models

Models for:
- Properties and listings
- Market data and demographics
- Financial projections
- Risk assessment
- Investment recommendations
"""

from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid


class PropertyType(Enum):
    """Types of real estate properties."""
    SINGLE_FAMILY = "Single Family"
    MULTI_FAMILY = "Multi-Family"
    CONDO = "Condo"
    TOWNHOUSE = "Townhouse"
    DUPLEX = "Duplex"
    TRIPLEX = "Triplex"
    FOURPLEX = "Fourplex"
    APARTMENT = "Apartment Building"
    COMMERCIAL = "Commercial"
    MIXED_USE = "Mixed Use"


class PropertyCondition(Enum):
    """Property condition ratings."""
    EXCELLENT = "Excellent"
    GOOD = "Good"
    FAIR = "Fair"
    POOR = "Poor"
    NEEDS_RENOVATION = "Needs Renovation"


class RiskLevel(Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Recommendation(Enum):
    """Investment recommendations."""
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    AVOID = "Avoid"
    STRONG_AVOID = "Strong Avoid"


@dataclass
class Address:
    """Property address."""
    street: str
    city: str
    state: str
    zip_code: str
    county: str = ""
    
    def __str__(self) -> str:
        return f"{self.street}, {self.city}, {self.state} {self.zip_code}"


@dataclass
class PropertyFeatures:
    """Property physical features."""
    bedrooms: int
    bathrooms: float
    sqft: int
    lot_sqft: int = 0
    year_built: int = 0
    stories: int = 1
    garage_spaces: int = 0
    pool: bool = False
    hoa: bool = False
    hoa_fee: float = 0.0


@dataclass
class Property:
    """A real estate property."""
    property_id: str
    address: Address
    property_type: PropertyType
    features: PropertyFeatures
    
    # Pricing
    list_price: float
    price_per_sqft: float = 0.0
    
    # Condition
    condition: PropertyCondition = PropertyCondition.GOOD
    
    # Listing info
    days_on_market: int = 0
    listing_date: str = ""
    
    # Images
    image_urls: list[str] = field(default_factory=list)
    
    # Description
    description: str = ""
    
    def __post_init__(self):
        if self.price_per_sqft == 0 and self.features.sqft > 0:
            self.price_per_sqft = self.list_price / self.features.sqft
    
    def to_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "address": str(self.address),
            "type": self.property_type.value,
            "price": self.list_price,
            "bedrooms": self.features.bedrooms,
            "bathrooms": self.features.bathrooms,
            "sqft": self.features.sqft,
            "year_built": self.features.year_built,
            "price_per_sqft": self.price_per_sqft,
        }


@dataclass
class ComparableSale:
    """A comparable property sale (comp)."""
    address: str
    sale_price: float
    sale_date: str
    sqft: int
    bedrooms: int
    bathrooms: float
    price_per_sqft: float = 0.0
    distance_miles: float = 0.0
    similarity_score: float = 0.0  # 0-100
    
    def __post_init__(self):
        if self.price_per_sqft == 0 and self.sqft > 0:
            self.price_per_sqft = self.sale_price / self.sqft


@dataclass
class Valuation:
    """Property valuation estimate."""
    estimated_value: float
    low_estimate: float
    high_estimate: float
    confidence: float  # 0-100
    
    # Methodology
    comps_value: float = 0.0
    income_value: float = 0.0
    cost_value: float = 0.0
    
    # Supporting data
    comparable_sales: list[ComparableSale] = field(default_factory=list)
    
    @property
    def is_good_deal(self) -> bool:
        """Check if list price is below estimated value."""
        return True  # Determined by comparison to list price


@dataclass
class MarketMetrics:
    """Real estate market metrics."""
    # Price metrics
    median_price: float
    median_price_sqft: float
    price_change_yoy: float  # Year over year %
    
    # Inventory
    active_listings: int
    months_supply: float
    days_on_market_avg: int
    
    # Sales volume
    sales_last_month: int
    sales_yoy_change: float
    
    # Rental metrics
    median_rent: float
    rent_change_yoy: float
    vacancy_rate: float


@dataclass
class Demographics:
    """Area demographics and economics."""
    population: int
    population_growth: float  # Annual %
    median_household_income: float
    income_growth: float
    unemployment_rate: float
    
    # Age distribution
    median_age: float
    
    # Education
    college_educated_pct: float
    
    # Housing
    owner_occupied_pct: float
    renter_occupied_pct: float


@dataclass
class MarketData:
    """Complete market data for an area."""
    city: str
    state: str
    zip_code: str
    
    # Metrics
    metrics: MarketMetrics = None
    demographics: Demographics = None
    
    # Trends
    price_trend: str = ""  # "rising", "falling", "stable"
    rent_trend: str = ""
    market_type: str = ""  # "seller's", "buyer's", "balanced"
    
    # Scores (0-100)
    investment_score: int = 0
    growth_score: int = 0
    affordability_score: int = 0
    
    def to_dict(self) -> dict:
        return {
            "location": f"{self.city}, {self.state}",
            "median_price": self.metrics.median_price if self.metrics else 0,
            "price_change_yoy": self.metrics.price_change_yoy if self.metrics else 0,
            "median_rent": self.metrics.median_rent if self.metrics else 0,
            "investment_score": self.investment_score,
        }


@dataclass
class RentalEstimate:
    """Rental income estimate."""
    monthly_rent: float
    low_estimate: float
    high_estimate: float
    
    # Market comparison
    market_average: float
    percentile: int  # Where this property falls in market
    
    # Confidence
    confidence: float
    comparable_rentals: int


@dataclass
class Expense:
    """An operating expense."""
    name: str
    annual_amount: float
    category: str  # "fixed", "variable", "reserve"
    notes: str = ""


@dataclass
class CashFlowProjection:
    """Annual cash flow projection."""
    year: int
    
    # Income
    gross_rental_income: float
    vacancy_loss: float
    effective_gross_income: float
    
    # Expenses
    operating_expenses: float
    
    # NOI
    net_operating_income: float
    
    # Debt service
    mortgage_payment: float = 0.0
    
    # Cash flow
    cash_flow_before_tax: float = 0.0
    
    # Appreciation
    property_value: float = 0.0
    appreciation: float = 0.0
    
    # Equity
    principal_paydown: float = 0.0
    total_equity_buildup: float = 0.0
    
    # Returns
    total_return: float = 0.0
    cash_on_cash: float = 0.0


@dataclass
class FinancialAnalysis:
    """Complete financial analysis."""
    property_id: str
    
    # Purchase
    purchase_price: float
    down_payment: float
    down_payment_pct: float
    loan_amount: float
    
    # Loan terms
    interest_rate: float
    loan_term_years: int
    monthly_mortgage: float
    
    # Closing costs
    closing_costs: float
    total_investment: float
    
    # Income
    rental_estimate: RentalEstimate = None
    gross_annual_income: float = 0.0
    
    # Expenses
    expenses: list[Expense] = field(default_factory=list)
    total_annual_expenses: float = 0.0
    expense_ratio: float = 0.0
    
    # Key metrics
    net_operating_income: float = 0.0
    cap_rate: float = 0.0
    cash_on_cash_return: float = 0.0
    gross_rent_multiplier: float = 0.0
    debt_service_coverage: float = 0.0
    
    # Projections
    projections: list[CashFlowProjection] = field(default_factory=list)
    
    # IRR (5 year)
    irr_5_year: float = 0.0
    total_return_5_year: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "purchase_price": self.purchase_price,
            "down_payment": self.down_payment,
            "monthly_rent": self.rental_estimate.monthly_rent if self.rental_estimate else 0,
            "cap_rate": self.cap_rate,
            "cash_on_cash": self.cash_on_cash_return,
            "noi": self.net_operating_income,
            "grm": self.gross_rent_multiplier,
        }


@dataclass
class RiskFactor:
    """A risk factor in the analysis."""
    name: str
    level: RiskLevel
    score: int  # 0-100 (higher = more risk)
    description: str
    mitigation: str = ""
    
    @property
    def level_emoji(self) -> str:
        return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(self.level.value, "⚪")


@dataclass
class RiskAssessment:
    """Complete risk assessment."""
    factors: list[RiskFactor] = field(default_factory=list)
    
    # Overall
    overall_risk_level: RiskLevel = RiskLevel.MEDIUM
    overall_risk_score: int = 50
    
    def calculate_overall(self):
        if not self.factors:
            return
        
        self.overall_risk_score = sum(f.score for f in self.factors) // len(self.factors)
        
        if self.overall_risk_score < 33:
            self.overall_risk_level = RiskLevel.LOW
        elif self.overall_risk_score < 66:
            self.overall_risk_level = RiskLevel.MEDIUM
        else:
            self.overall_risk_level = RiskLevel.HIGH


@dataclass
class LegalCheck:
    """Legal and regulatory check result."""
    item: str
    status: str  # "clear", "warning", "issue"
    details: str
    
    @property
    def status_emoji(self) -> str:
        return {"clear": "✅", "warning": "⚠️", "issue": "❌"}.get(self.status, "❓")


@dataclass
class LegalAnalysis:
    """Complete legal analysis."""
    checks: list[LegalCheck] = field(default_factory=list)
    
    # Zoning
    zoning_code: str = ""
    zoning_description: str = ""
    zoning_compliant: bool = True
    
    # Permits
    open_permits: int = 0
    permit_issues: list[str] = field(default_factory=list)
    
    # Title
    title_clear: bool = True
    liens: list[str] = field(default_factory=list)
    
    # HOA
    hoa_restrictions: list[str] = field(default_factory=list)


@dataclass
class InvestmentRecommendation:
    """Final investment recommendation."""
    recommendation: Recommendation
    confidence: float  # 0-100
    
    # Summary
    summary: str
    
    # Key points
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    
    # Suggested actions
    suggested_offer: float = 0.0
    max_price: float = 0.0
    
    # Conditions
    conditions: list[str] = field(default_factory=list)
    
    @property
    def recommendation_emoji(self) -> str:
        emojis = {
            Recommendation.STRONG_BUY: "🟢🟢",
            Recommendation.BUY: "🟢",
            Recommendation.HOLD: "🟡",
            Recommendation.AVOID: "🔴",
            Recommendation.STRONG_AVOID: "🔴🔴",
        }
        return emojis.get(self.recommendation, "⚪")


@dataclass
class InvestmentAnalysis:
    """Complete investment analysis report."""
    analysis_id: str
    created_at: str
    
    # Subject property
    property: Property
    
    # Analysis components
    valuation: Valuation = None
    market_data: MarketData = None
    financial_analysis: FinancialAnalysis = None
    risk_assessment: RiskAssessment = None
    legal_analysis: LegalAnalysis = None
    
    # Final recommendation
    recommendation: InvestmentRecommendation = None
    
    def to_dict(self) -> dict:
        return {
            "analysis_id": self.analysis_id,
            "property": self.property.to_dict(),
            "valuation": {
                "estimated_value": self.valuation.estimated_value if self.valuation else 0,
                "confidence": self.valuation.confidence if self.valuation else 0,
            },
            "financials": self.financial_analysis.to_dict() if self.financial_analysis else {},
            "recommendation": self.recommendation.recommendation.value if self.recommendation else "Pending",
        }
    
    def to_markdown(self) -> str:
        """Generate markdown report."""
        prop = self.property
        
        md = f"""# Investment Analysis: {prop.address}

**Analysis ID:** {self.analysis_id}
**Generated:** {self.created_at}

---

## Property Overview

- **List Price:** ${prop.list_price:,.0f}
- **Type:** {prop.property_type.value}, {prop.features.bedrooms}BR/{prop.features.bathrooms}BA
- **Size:** {prop.features.sqft:,} sqft | Lot: {prop.features.lot_sqft:,} sqft
- **Year Built:** {prop.features.year_built}
- **Price/sqft:** ${prop.price_per_sqft:.0f}
"""
        
        if self.valuation:
            md += f"- **Fair Value Estimate:** ${self.valuation.low_estimate:,.0f} - ${self.valuation.high_estimate:,.0f}"
            if prop.list_price <= self.valuation.estimated_value:
                md += " ✓ Good Price\n"
            else:
                md += " ⚠️ Above Estimate\n"
        
        # Financial Projections
        if self.financial_analysis:
            fa = self.financial_analysis
            md += f"""
## Financial Projections

| Metric | Value | Market Avg |
|--------|-------|------------|
| Monthly Rent | ${fa.rental_estimate.monthly_rent:,.0f} | ${fa.rental_estimate.market_average:,.0f} |
| Cap Rate | {fa.cap_rate:.1f}% | {fa.cap_rate - 0.4:.1f}% |
| Cash-on-Cash ({fa.down_payment_pct:.0f}% down) | {fa.cash_on_cash_return:.1f}% | {fa.cash_on_cash_return - 1.5:.1f}% |
| GRM | {fa.gross_rent_multiplier:.1f} | {fa.gross_rent_multiplier + 0.8:.1f} |
| NOI | ${fa.net_operating_income:,.0f} | - |
| DSCR | {fa.debt_service_coverage:.2f} | 1.25+ |

### 5-Year Pro Forma
| Year | NOI | Appreciation | Total Return |
|------|-----|--------------|--------------|
"""
            for proj in fa.projections[:5]:
                md += f"| {proj.year} | ${proj.net_operating_income:,.0f} | ${proj.appreciation:,.0f} | ${proj.total_return:,.0f} |\n"
            
            if fa.irr_5_year:
                md += f"\n**5-Year IRR:** {fa.irr_5_year:.1f}%\n"
        
        # Risk Assessment
        if self.risk_assessment:
            ra = self.risk_assessment
            md += "\n## Risk Assessment\n\n"
            
            for factor in ra.factors:
                md += f"- {factor.level_emoji} **{factor.name}:** {factor.description}\n"
            
            md += f"\n**Overall Risk:** {ra.overall_risk_level.value.upper()} (Score: {ra.overall_risk_score}/100)\n"
        
        # Market Data
        if self.market_data:
            mkt = self.market_data
            md += f"""
## Market Analysis: {mkt.city}, {mkt.state}

- **Investment Score:** {mkt.investment_score}/100
- **Market Type:** {mkt.market_type}
- **Price Trend:** {mkt.price_trend}
"""
            if mkt.metrics:
                md += f"""- **Median Price:** ${mkt.metrics.median_price:,.0f} ({mkt.metrics.price_change_yoy:+.1f}% YoY)
- **Days on Market:** {mkt.metrics.days_on_market_avg} days
- **Vacancy Rate:** {mkt.metrics.vacancy_rate:.1f}%
"""
        
        # Legal
        if self.legal_analysis:
            la = self.legal_analysis
            md += "\n## Legal & Zoning\n\n"
            for check in la.checks:
                md += f"- {check.status_emoji} **{check.item}:** {check.details}\n"
        
        # Recommendation
        if self.recommendation:
            rec = self.recommendation
            md += f"""
## Recommendation

### {rec.recommendation_emoji} {rec.recommendation.value.upper()}

{rec.summary}

**Confidence:** {rec.confidence:.0f}%
"""
            if rec.suggested_offer:
                md += f"\n**Suggested Offer:** ${rec.suggested_offer:,.0f}"
                md += f"\n**Max Price:** ${rec.max_price:,.0f}\n"
            
            if rec.pros:
                md += "\n**Pros:**\n"
                for pro in rec.pros:
                    md += f"- ✅ {pro}\n"
            
            if rec.cons:
                md += "\n**Cons:**\n"
                for con in rec.cons:
                    md += f"- ⚠️ {con}\n"
            
            if rec.conditions:
                md += "\n**Conditions:**\n"
                for cond in rec.conditions:
                    md += f"- {cond}\n"
        
        return md
