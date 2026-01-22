"""
Real Estate Investment Advisor - Specialist Agents

Supervisor Pattern with specialized agents:
- Market Analyst: Demographics, trends, supply/demand
- Property Evaluator: Comparables, condition, valuation
- Financial Modeler: Cash flow, ROI/IRR, cap rate
- Risk Assessor: Vacancy, appreciation, market risks
- Legal Checker: Zoning, permits, title
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import os

from ..models import (
    Property, MarketData, Valuation, FinancialAnalysis,
    RiskAssessment, LegalAnalysis, InvestmentRecommendation
)
from ..tools import (
    get_property_valuation, get_comparable_sales, get_market_analysis,
    calculate_financial_analysis, assess_risks, check_legal,
    generate_recommendation
)


@dataclass
class AgentConfig:
    """Configuration for agents."""
    llm_provider: str = "gemini"
    verbose: bool = True


def get_llm(provider: str = "gemini"):
    """Get LLM for agent reasoning."""
    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
            )
        except ImportError:
            pass
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model="claude-sonnet-4-20250514",
                temperature=0.3,
            )
        except ImportError:
            pass
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.3,
            )
        except ImportError:
            pass
    
    return None


# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent:
    """Base class for specialist agents."""
    
    def __init__(self, name: str, description: str, config: AgentConfig = None):
        self.name = name
        self.description = description
        self.config = config or AgentConfig()
        self.llm = get_llm(self.config.llm_provider)
    
    def log(self, message: str):
        """Log agent activity."""
        if self.config.verbose:
            print(f"  [{self.name}] {message}")
    
    def analyze(self, property: Property, context: dict) -> dict:
        """Analyze property - to be implemented by subclasses."""
        raise NotImplementedError


# =============================================================================
# Market Analyst Agent
# =============================================================================

class MarketAnalyst(BaseAgent):
    """
    Market Analyst Agent
    
    Responsibilities:
    - Analyze local market conditions
    - Track demographic trends
    - Assess supply/demand dynamics
    - Evaluate economic indicators
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Market Analyst",
            description="Analyzes real estate market conditions and trends",
            config=config,
        )
    
    def analyze(self, property: Property, context: dict) -> dict:
        """Analyze market conditions for property location."""
        self.log(f"Analyzing market for {property.address.city}, {property.address.state}")
        
        # Get market data
        market_data = get_market_analysis(property.address.city)
        
        # Build analysis
        analysis = {
            "market_data": market_data,
            "summary": self._generate_summary(market_data),
            "outlook": self._assess_outlook(market_data),
        }
        
        self.log(f"✓ Market analysis complete. Investment score: {market_data.investment_score}/100")
        
        return analysis
    
    def _generate_summary(self, market: MarketData) -> str:
        """Generate market summary."""
        if not market or not market.metrics:
            return "Limited market data available."
        
        m = market.metrics
        
        return f"""
{market.city}, {market.state} Real Estate Market:

- Median home price: ${m.median_price:,.0f} ({m.price_change_yoy:+.1f}% YoY)
- Median rent: ${m.median_rent:,.0f}/month ({m.rent_change_yoy:+.1f}% YoY)
- Days on market: {m.days_on_market_avg} days
- Vacancy rate: {m.vacancy_rate:.1f}%
- Market type: {market.market_type}
- Price trend: {market.price_trend}

Investment Score: {market.investment_score}/100
"""
    
    def _assess_outlook(self, market: MarketData) -> str:
        """Assess market outlook."""
        if market.investment_score >= 80:
            return "Strong investment potential. High growth market with strong fundamentals."
        elif market.investment_score >= 65:
            return "Good investment potential. Stable market with reasonable returns."
        elif market.investment_score >= 50:
            return "Moderate investment potential. Consider carefully."
        else:
            return "Challenging market. Proceed with caution."


# =============================================================================
# Property Evaluator Agent
# =============================================================================

class PropertyEvaluator(BaseAgent):
    """
    Property Evaluator Agent
    
    Responsibilities:
    - Evaluate property condition
    - Analyze comparable sales
    - Estimate fair market value
    - Identify property-specific factors
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Property Evaluator",
            description="Evaluates properties and estimates value",
            config=config,
        )
    
    def analyze(self, property: Property, context: dict) -> dict:
        """Evaluate property and estimate value."""
        self.log(f"Evaluating property at {property.address}")
        
        # Get valuation
        valuation = get_property_valuation(property)
        
        # Get comparables
        comps = get_comparable_sales(property)
        
        # Build analysis
        analysis = {
            "valuation": valuation,
            "comparables": comps,
            "summary": self._generate_summary(property, valuation),
            "price_assessment": self._assess_price(property, valuation),
        }
        
        self.log(f"✓ Valuation complete. Estimated value: ${valuation.estimated_value:,.0f}")
        
        return analysis
    
    def _generate_summary(self, property: Property, valuation: Valuation) -> str:
        """Generate property evaluation summary."""
        return f"""
Property: {property.address}

Type: {property.property_type.value}
Size: {property.features.sqft:,} sqft on {property.features.lot_sqft:,} sqft lot
Built: {property.features.year_built}
Condition: {property.condition.value}

Valuation:
- Estimated Value: ${valuation.estimated_value:,.0f}
- Value Range: ${valuation.low_estimate:,.0f} - ${valuation.high_estimate:,.0f}
- Confidence: {valuation.confidence:.0f}%

Methodology:
- Comparable Sales Approach: ${valuation.comps_value:,.0f}
- Income Approach: ${valuation.income_value:,.0f}
- Cost Approach: ${valuation.cost_value:,.0f}
"""
    
    def _assess_price(self, property: Property, valuation: Valuation) -> str:
        """Assess listing price vs value."""
        diff = property.list_price - valuation.estimated_value
        pct = (diff / valuation.estimated_value) * 100
        
        if pct < -5:
            return f"UNDERPRICED: Listed {abs(pct):.1f}% below estimated value. Good deal potential."
        elif pct > 5:
            return f"OVERPRICED: Listed {pct:.1f}% above estimated value. Negotiate down."
        else:
            return f"FAIR PRICE: Listed within {abs(pct):.1f}% of estimated value."


# =============================================================================
# Financial Modeler Agent
# =============================================================================

class FinancialModeler(BaseAgent):
    """
    Financial Modeler Agent
    
    Responsibilities:
    - Model cash flows
    - Calculate returns (ROI, IRR, cap rate)
    - Project long-term performance
    - Analyze financing scenarios
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Financial Modeler",
            description="Models financial performance and returns",
            config=config,
        )
    
    def analyze(
        self,
        property: Property,
        context: dict,
        down_payment_pct: float = 0.25,
        interest_rate: float = 0.07,
    ) -> dict:
        """Analyze financial performance."""
        self.log(f"Modeling financials with {down_payment_pct*100:.0f}% down, {interest_rate*100:.1f}% rate")
        
        # Calculate financials
        financials = calculate_financial_analysis(
            property,
            down_payment_pct=down_payment_pct,
            interest_rate=interest_rate,
        )
        
        # Build analysis
        analysis = {
            "financials": financials,
            "summary": self._generate_summary(financials),
            "cash_flow_assessment": self._assess_cash_flow(financials),
        }
        
        self.log(f"✓ Financial analysis complete. Cap rate: {financials.cap_rate:.1f}%, CoC: {financials.cash_on_cash_return:.1f}%")
        
        return analysis
    
    def _generate_summary(self, fa: FinancialAnalysis) -> str:
        """Generate financial summary."""
        return f"""
Financial Analysis:

Purchase:
- Price: ${fa.purchase_price:,.0f}
- Down Payment: ${fa.down_payment:,.0f} ({fa.down_payment_pct:.0f}%)
- Loan Amount: ${fa.loan_amount:,.0f}
- Monthly Mortgage: ${fa.monthly_mortgage:,.0f}

Income:
- Monthly Rent: ${fa.rental_estimate.monthly_rent:,.0f}
- Annual Gross: ${fa.gross_annual_income:,.0f}

Expenses:
- Annual Expenses: ${fa.total_annual_expenses:,.0f}
- Expense Ratio: {fa.expense_ratio:.1f}%

Performance:
- NOI: ${fa.net_operating_income:,.0f}
- Cap Rate: {fa.cap_rate:.2f}%
- Cash-on-Cash: {fa.cash_on_cash_return:.2f}%
- GRM: {fa.gross_rent_multiplier:.1f}
- DSCR: {fa.debt_service_coverage:.2f}
- 5-Year IRR: {fa.irr_5_year:.1f}%
"""
    
    def _assess_cash_flow(self, fa: FinancialAnalysis) -> str:
        """Assess cash flow quality."""
        if fa.cash_on_cash_return >= 8:
            return "EXCELLENT cash flow. Strong investment returns."
        elif fa.cash_on_cash_return >= 5:
            return "GOOD cash flow. Meets typical investor targets."
        elif fa.cash_on_cash_return >= 2:
            return "MODERATE cash flow. Consider appreciation potential."
        else:
            return "WEAK cash flow. Negative or minimal returns."


# =============================================================================
# Risk Assessor Agent
# =============================================================================

class RiskAssessor(BaseAgent):
    """
    Risk Assessor Agent
    
    Responsibilities:
    - Identify investment risks
    - Assess market volatility
    - Evaluate vacancy risk
    - Analyze property-specific risks
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Risk Assessor",
            description="Assesses investment risks",
            config=config,
        )
    
    def analyze(self, property: Property, context: dict) -> dict:
        """Assess investment risks."""
        self.log("Assessing investment risks...")
        
        market_data = context.get("market_data")
        financials = context.get("financials")
        
        # Assess risks
        risk_assessment = assess_risks(property, market_data, financials)
        
        # Build analysis
        analysis = {
            "risk_assessment": risk_assessment,
            "summary": self._generate_summary(risk_assessment),
            "mitigation_plan": self._create_mitigation_plan(risk_assessment),
        }
        
        self.log(f"✓ Risk assessment complete. Overall: {risk_assessment.overall_risk_level.value.upper()}")
        
        return analysis
    
    def _generate_summary(self, ra: RiskAssessment) -> str:
        """Generate risk summary."""
        lines = [f"Overall Risk Level: {ra.overall_risk_level.value.upper()} (Score: {ra.overall_risk_score}/100)\n"]
        
        for factor in ra.factors:
            lines.append(f"- {factor.level_emoji} {factor.name}: {factor.description}")
        
        return "\n".join(lines)
    
    def _create_mitigation_plan(self, ra: RiskAssessment) -> list[str]:
        """Create risk mitigation plan."""
        mitigations = []
        
        for factor in ra.factors:
            if factor.mitigation:
                mitigations.append(f"{factor.name}: {factor.mitigation}")
        
        return mitigations


# =============================================================================
# Legal Checker Agent
# =============================================================================

class LegalChecker(BaseAgent):
    """
    Legal Checker Agent
    
    Responsibilities:
    - Check zoning compliance
    - Verify building permits
    - Review title status
    - Identify legal restrictions
    """
    
    def __init__(self, config: AgentConfig = None):
        super().__init__(
            name="Legal Checker",
            description="Checks legal and regulatory compliance",
            config=config,
        )
    
    def analyze(self, property: Property, context: dict) -> dict:
        """Check legal status."""
        self.log("Checking legal and regulatory status...")
        
        # Run legal checks
        legal_analysis = check_legal(property)
        
        # Build analysis
        analysis = {
            "legal_analysis": legal_analysis,
            "summary": self._generate_summary(legal_analysis),
            "issues": self._identify_issues(legal_analysis),
        }
        
        issues = [c for c in legal_analysis.checks if c.status != "clear"]
        self.log(f"✓ Legal check complete. {len(issues)} issue(s) found.")
        
        return analysis
    
    def _generate_summary(self, la: LegalAnalysis) -> str:
        """Generate legal summary."""
        lines = [f"Zoning: {la.zoning_code} - {la.zoning_description}\n"]
        
        for check in la.checks:
            lines.append(f"{check.status_emoji} {check.item}: {check.details}")
        
        return "\n".join(lines)
    
    def _identify_issues(self, la: LegalAnalysis) -> list[str]:
        """Identify legal issues."""
        issues = []
        
        if not la.title_clear:
            issues.append("Title is not clear - liens must be resolved")
        
        if la.open_permits > 0:
            issues.append(f"{la.open_permits} open permit(s) require verification")
        
        if not la.zoning_compliant:
            issues.append("Property may not be zoning compliant")
        
        for check in la.checks:
            if check.status == "issue":
                issues.append(f"{check.item}: {check.details}")
        
        return issues


# =============================================================================
# Supervisor Agent
# =============================================================================

class SupervisorAgent:
    """
    Supervisor Agent - Coordinates specialist agents.
    
    Routes queries to appropriate specialists and synthesizes results.
    """
    
    def __init__(self, config: AgentConfig = None):
        self.config = config or AgentConfig()
        self.name = "Supervisor"
        
        # Initialize specialist agents
        self.market_analyst = MarketAnalyst(config)
        self.property_evaluator = PropertyEvaluator(config)
        self.financial_modeler = FinancialModeler(config)
        self.risk_assessor = RiskAssessor(config)
        self.legal_checker = LegalChecker(config)
    
    def log(self, message: str):
        """Log supervisor activity."""
        if self.config.verbose:
            print(f"[{self.name}] {message}")
    
    def analyze_property(
        self,
        property: Property,
        down_payment_pct: float = 0.25,
        interest_rate: float = 0.07,
    ) -> dict:
        """
        Run full property analysis with all specialists.
        
        Returns complete analysis with all agent outputs.
        """
        self.log(f"Starting analysis of {property.address}")
        self.log("="*50)
        
        context = {"property": property}
        results = {}
        
        # 1. Market Analysis
        self.log("📊 Running Market Analysis...")
        market_result = self.market_analyst.analyze(property, context)
        results["market"] = market_result
        context["market_data"] = market_result["market_data"]
        
        # 2. Property Evaluation
        self.log("🏠 Running Property Evaluation...")
        property_result = self.property_evaluator.analyze(property, context)
        results["property"] = property_result
        context["valuation"] = property_result["valuation"]
        
        # 3. Financial Modeling
        self.log("💰 Running Financial Modeling...")
        financial_result = self.financial_modeler.analyze(
            property, context,
            down_payment_pct=down_payment_pct,
            interest_rate=interest_rate,
        )
        results["financial"] = financial_result
        context["financials"] = financial_result["financials"]
        
        # 4. Risk Assessment
        self.log("⚠️ Running Risk Assessment...")
        risk_result = self.risk_assessor.analyze(property, context)
        results["risk"] = risk_result
        context["risk_assessment"] = risk_result["risk_assessment"]
        
        # 5. Legal Check
        self.log("⚖️ Running Legal Check...")
        legal_result = self.legal_checker.analyze(property, context)
        results["legal"] = legal_result
        context["legal_analysis"] = legal_result["legal_analysis"]
        
        # 6. Generate Recommendation
        self.log("📋 Generating Recommendation...")
        recommendation = generate_recommendation(
            property=property,
            valuation=context["valuation"],
            market=context["market_data"],
            financials=context["financials"],
            risks=context["risk_assessment"],
            legal=context["legal_analysis"],
        )
        results["recommendation"] = recommendation
        
        self.log("="*50)
        self.log(f"✅ Analysis complete: {recommendation.recommendation.value}")
        
        return {
            "property": property,
            "market_data": context["market_data"],
            "valuation": context["valuation"],
            "financials": context["financials"],
            "risk_assessment": context["risk_assessment"],
            "legal_analysis": context["legal_analysis"],
            "recommendation": recommendation,
            "agent_outputs": results,
        }
