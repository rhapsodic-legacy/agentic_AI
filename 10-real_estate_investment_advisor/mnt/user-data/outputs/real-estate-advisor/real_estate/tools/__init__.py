"""
Real Estate Investment Advisor - Analysis Tools

Tools for:
- Property valuation
- Market analysis
- Financial modeling
- Risk assessment
- Legal checking
"""

from typing import Optional
from datetime import datetime
import math

from ..models import (
    Property, PropertyType, PropertyCondition,
    Valuation, ComparableSale, MarketData,
    RentalEstimate, FinancialAnalysis, Expense, CashFlowProjection,
    RiskFactor, RiskLevel, RiskAssessment,
    LegalAnalysis, LegalCheck,
    InvestmentRecommendation, Recommendation
)
from ..data import property_source, market_source, rental_source, legal_source


# =============================================================================
# Valuation Tools
# =============================================================================

def get_property_valuation(property: Property) -> Valuation:
    """
    Estimate property value using multiple methods.
    
    Methods:
    1. Comparable Sales Approach
    2. Income Approach (for rentals)
    3. Cost Approach
    """
    # Get comparable sales
    comps = property_source.get_comparable_sales(property)
    
    # Comparable Sales Approach
    if comps:
        comp_prices = [c.sale_price for c in comps]
        comps_value = sum(comp_prices) / len(comp_prices)
        
        # Adjust for property differences
        avg_comp_sqft = sum(c.sqft for c in comps) / len(comps)
        sqft_diff = property.features.sqft - avg_comp_sqft
        comps_value += sqft_diff * (comps_value / avg_comp_sqft) * 0.8
    else:
        comps_value = property.list_price
    
    # Income Approach
    rental_est = rental_source.estimate_rent(property)
    annual_rent = rental_est.monthly_rent * 12
    
    # Get market cap rate
    market_data = market_source.get_market_data(property.address.city)
    cap_rate = 0.055  # Default 5.5%
    
    if market_data and market_data.metrics:
        # Estimate cap rate from market data
        cap_rate = max(0.04, min(0.08, annual_rent / (market_data.metrics.median_price * 0.9)))
    
    # Assuming 40% expense ratio
    noi = annual_rent * 0.6
    income_value = noi / cap_rate
    
    # Cost Approach (rough estimate)
    cost_per_sqft = 150  # Construction cost estimate
    land_value = property.features.lot_sqft * 30 if property.features.lot_sqft else property.list_price * 0.2
    depreciation = (datetime.now().year - property.features.year_built) * 0.01 if property.features.year_built else 0.15
    
    replacement_cost = property.features.sqft * cost_per_sqft
    depreciated_value = replacement_cost * (1 - min(0.5, depreciation))
    cost_value = land_value + depreciated_value
    
    # Weighted average (comps weighted highest for residential)
    estimated_value = (comps_value * 0.5 + income_value * 0.3 + cost_value * 0.2)
    
    # Calculate confidence based on comp quality
    confidence = 75
    if comps:
        avg_similarity = sum(c.similarity_score for c in comps) / len(comps)
        confidence = min(95, avg_similarity)
    
    # Calculate range
    variance = 0.05 + (100 - confidence) / 500
    
    return Valuation(
        estimated_value=estimated_value,
        low_estimate=estimated_value * (1 - variance),
        high_estimate=estimated_value * (1 + variance),
        confidence=confidence,
        comps_value=comps_value,
        income_value=income_value,
        cost_value=cost_value,
        comparable_sales=comps[:5],
    )


def get_comparable_sales(property: Property, radius_miles: float = 1.0) -> list[ComparableSale]:
    """Get comparable sales for a property."""
    return property_source.get_comparable_sales(property, radius_miles)


# =============================================================================
# Market Analysis Tools
# =============================================================================

def get_market_analysis(city: str) -> MarketData:
    """Get comprehensive market analysis for a city."""
    return market_source.get_market_data(city)


def calculate_market_score(market: MarketData) -> dict:
    """Calculate detailed market scores."""
    scores = {
        "investment": market.investment_score,
        "growth": market.growth_score,
        "affordability": market.affordability_score,
        "rental_demand": 0,
        "price_stability": 0,
    }
    
    if market.metrics:
        # Rental demand (based on vacancy rate)
        scores["rental_demand"] = max(0, 100 - int(market.metrics.vacancy_rate * 10))
        
        # Price stability
        abs_change = abs(market.metrics.price_change_yoy)
        if abs_change < 3:
            scores["price_stability"] = 90
        elif abs_change < 6:
            scores["price_stability"] = 75
        elif abs_change < 10:
            scores["price_stability"] = 60
        else:
            scores["price_stability"] = 40
    
    scores["overall"] = sum(scores.values()) // len(scores)
    
    return scores


# =============================================================================
# Financial Modeling Tools
# =============================================================================

def calculate_financial_analysis(
    property: Property,
    down_payment_pct: float = 0.25,
    interest_rate: float = 0.07,
    loan_term_years: int = 30,
    closing_cost_pct: float = 0.03,
    appreciation_rate: float = 0.03,
    rent_growth_rate: float = 0.03,
    expense_growth_rate: float = 0.02,
    holding_years: int = 5,
) -> FinancialAnalysis:
    """
    Calculate comprehensive financial analysis.
    
    Returns ROI, cap rate, cash flow projections, and IRR.
    """
    purchase_price = property.list_price
    
    # Financing
    down_payment = purchase_price * down_payment_pct
    loan_amount = purchase_price - down_payment
    closing_costs = purchase_price * closing_cost_pct
    total_investment = down_payment + closing_costs
    
    # Monthly mortgage (P&I)
    monthly_rate = interest_rate / 12
    num_payments = loan_term_years * 12
    
    if loan_amount > 0 and monthly_rate > 0:
        monthly_mortgage = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    else:
        monthly_mortgage = 0
    
    annual_mortgage = monthly_mortgage * 12
    
    # Rental income estimate
    rental_estimate = rental_source.estimate_rent(property)
    monthly_rent = rental_estimate.monthly_rent
    gross_annual_income = monthly_rent * 12
    
    # Operating expenses
    expenses = create_expense_list(property, gross_annual_income)
    total_annual_expenses = sum(e.annual_amount for e in expenses)
    expense_ratio = total_annual_expenses / gross_annual_income if gross_annual_income > 0 else 0
    
    # NOI
    vacancy_rate = 0.05
    effective_income = gross_annual_income * (1 - vacancy_rate)
    noi = effective_income - total_annual_expenses
    
    # Key metrics
    cap_rate = (noi / purchase_price) * 100 if purchase_price > 0 else 0
    
    annual_cash_flow = noi - annual_mortgage
    cash_on_cash = (annual_cash_flow / total_investment) * 100 if total_investment > 0 else 0
    
    grm = purchase_price / gross_annual_income if gross_annual_income > 0 else 0
    
    dscr = noi / annual_mortgage if annual_mortgage > 0 else float('inf')
    
    # Generate projections
    projections = generate_cash_flow_projections(
        purchase_price=purchase_price,
        initial_noi=noi,
        annual_mortgage=annual_mortgage,
        loan_amount=loan_amount,
        interest_rate=interest_rate,
        appreciation_rate=appreciation_rate,
        rent_growth_rate=rent_growth_rate,
        expense_growth_rate=expense_growth_rate,
        expense_ratio=expense_ratio,
        vacancy_rate=vacancy_rate,
        holding_years=holding_years,
        total_investment=total_investment,
    )
    
    # Calculate IRR
    irr_5_year = calculate_irr(projections[:5], total_investment)
    total_return_5_year = sum(p.total_return for p in projections[:5])
    
    return FinancialAnalysis(
        property_id=property.property_id,
        purchase_price=purchase_price,
        down_payment=down_payment,
        down_payment_pct=down_payment_pct * 100,
        loan_amount=loan_amount,
        interest_rate=interest_rate * 100,
        loan_term_years=loan_term_years,
        monthly_mortgage=monthly_mortgage,
        closing_costs=closing_costs,
        total_investment=total_investment,
        rental_estimate=rental_estimate,
        gross_annual_income=gross_annual_income,
        expenses=expenses,
        total_annual_expenses=total_annual_expenses,
        expense_ratio=expense_ratio * 100,
        net_operating_income=noi,
        cap_rate=cap_rate,
        cash_on_cash_return=cash_on_cash,
        gross_rent_multiplier=grm,
        debt_service_coverage=dscr,
        projections=projections,
        irr_5_year=irr_5_year,
        total_return_5_year=total_return_5_year,
    )


def create_expense_list(property: Property, gross_income: float) -> list[Expense]:
    """Create list of operating expenses."""
    expenses = []
    
    # Property taxes (estimate 1.5% of value)
    expenses.append(Expense(
        name="Property Taxes",
        annual_amount=property.list_price * 0.015,
        category="fixed",
    ))
    
    # Insurance (estimate 0.5% of value)
    expenses.append(Expense(
        name="Insurance",
        annual_amount=property.list_price * 0.005,
        category="fixed",
    ))
    
    # HOA if applicable
    if property.features.hoa and property.features.hoa_fee > 0:
        expenses.append(Expense(
            name="HOA Fees",
            annual_amount=property.features.hoa_fee * 12,
            category="fixed",
        ))
    
    # Maintenance (5% of rent)
    expenses.append(Expense(
        name="Maintenance & Repairs",
        annual_amount=gross_income * 0.05,
        category="variable",
    ))
    
    # Property management (8% of rent)
    expenses.append(Expense(
        name="Property Management",
        annual_amount=gross_income * 0.08,
        category="variable",
    ))
    
    # Capital expenditure reserve (5% of rent)
    expenses.append(Expense(
        name="CapEx Reserve",
        annual_amount=gross_income * 0.05,
        category="reserve",
    ))
    
    return expenses


def generate_cash_flow_projections(
    purchase_price: float,
    initial_noi: float,
    annual_mortgage: float,
    loan_amount: float,
    interest_rate: float,
    appreciation_rate: float,
    rent_growth_rate: float,
    expense_growth_rate: float,
    expense_ratio: float,
    vacancy_rate: float,
    holding_years: int,
    total_investment: float,
) -> list[CashFlowProjection]:
    """Generate year-by-year cash flow projections."""
    projections = []
    
    property_value = purchase_price
    remaining_balance = loan_amount
    
    for year in range(1, holding_years + 1):
        # Income growth
        gross_income = (purchase_price / 13) * (1 + rent_growth_rate) ** year  # Rough estimate
        actual_gross = initial_noi / (1 - expense_ratio) * (1 + rent_growth_rate) ** (year - 1)
        
        vacancy_loss = actual_gross * vacancy_rate
        egi = actual_gross - vacancy_loss
        
        # Expense growth
        expenses = egi * expense_ratio * (1 + expense_growth_rate) ** (year - 1)
        
        noi = egi - expenses
        
        # Mortgage payment (calculate principal portion)
        annual_interest = remaining_balance * interest_rate
        principal_paydown = annual_mortgage - annual_interest if annual_mortgage > annual_interest else 0
        remaining_balance -= principal_paydown
        
        # Cash flow
        cash_flow = noi - annual_mortgage
        
        # Property appreciation
        prev_value = property_value
        property_value = purchase_price * (1 + appreciation_rate) ** year
        appreciation = property_value - prev_value
        
        # Total equity buildup
        equity_buildup = principal_paydown + appreciation
        
        # Total return
        total_return = cash_flow + equity_buildup
        
        # Cash on cash
        coc = (cash_flow / total_investment) * 100 if total_investment > 0 else 0
        
        projections.append(CashFlowProjection(
            year=year,
            gross_rental_income=actual_gross,
            vacancy_loss=vacancy_loss,
            effective_gross_income=egi,
            operating_expenses=expenses,
            net_operating_income=noi,
            mortgage_payment=annual_mortgage,
            cash_flow_before_tax=cash_flow,
            property_value=property_value,
            appreciation=appreciation,
            principal_paydown=principal_paydown,
            total_equity_buildup=equity_buildup,
            total_return=total_return,
            cash_on_cash=coc,
        ))
    
    return projections


def calculate_irr(projections: list[CashFlowProjection], initial_investment: float) -> float:
    """Calculate Internal Rate of Return."""
    if not projections:
        return 0.0
    
    # Cash flows: initial investment (negative), then annual returns
    cash_flows = [-initial_investment]
    
    for i, proj in enumerate(projections):
        if i == len(projections) - 1:
            # Last year includes sale proceeds (property value - remaining mortgage)
            sale_proceeds = proj.property_value * 0.94  # 6% selling costs
            cash_flows.append(proj.cash_flow_before_tax + sale_proceeds)
        else:
            cash_flows.append(proj.cash_flow_before_tax)
    
    # Newton's method for IRR
    irr = 0.10  # Initial guess
    
    for _ in range(100):
        npv = sum(cf / (1 + irr) ** t for t, cf in enumerate(cash_flows))
        npv_derivative = sum(-t * cf / (1 + irr) ** (t + 1) for t, cf in enumerate(cash_flows))
        
        if abs(npv_derivative) < 1e-10:
            break
        
        new_irr = irr - npv / npv_derivative
        
        if abs(new_irr - irr) < 1e-6:
            break
        
        irr = new_irr
    
    return max(0, irr * 100)


# =============================================================================
# Risk Assessment Tools
# =============================================================================

def assess_risks(property: Property, market: MarketData, financials: FinancialAnalysis) -> RiskAssessment:
    """
    Comprehensive risk assessment.
    
    Factors:
    - Market risk (price volatility, supply/demand)
    - Vacancy risk
    - Cash flow risk
    - Appreciation risk
    - Condition/maintenance risk
    - Location risk
    """
    factors = []
    
    # Market Risk
    if market and market.metrics:
        if market.metrics.price_change_yoy > 10:
            factors.append(RiskFactor(
                name="Market Volatility",
                level=RiskLevel.MEDIUM,
                score=55,
                description=f"High price growth ({market.metrics.price_change_yoy:.1f}%) may indicate overheated market",
                mitigation="Consider offering below asking price",
            ))
        elif market.metrics.price_change_yoy < 0:
            factors.append(RiskFactor(
                name="Declining Market",
                level=RiskLevel.HIGH,
                score=75,
                description=f"Negative price trend ({market.metrics.price_change_yoy:.1f}%)",
                mitigation="Ensure strong cash flow to offset potential value decline",
            ))
        else:
            factors.append(RiskFactor(
                name="Market Stability",
                level=RiskLevel.LOW,
                score=25,
                description=f"Stable market growth ({market.metrics.price_change_yoy:.1f}% YoY)",
            ))
        
        # Vacancy Risk
        vacancy = market.metrics.vacancy_rate
        if vacancy > 8:
            level = RiskLevel.HIGH
            score = 70
        elif vacancy > 5:
            level = RiskLevel.MEDIUM
            score = 45
        else:
            level = RiskLevel.LOW
            score = 20
        
        factors.append(RiskFactor(
            name="Vacancy Risk",
            level=level,
            score=score,
            description=f"{vacancy:.1f}% market vacancy rate",
            mitigation="Budget for extended vacancies" if level != RiskLevel.LOW else "",
        ))
        
        # Supply Risk (months of inventory)
        if market.metrics.months_supply > 6:
            factors.append(RiskFactor(
                name="Oversupply",
                level=RiskLevel.MEDIUM,
                score=50,
                description=f"{market.metrics.months_supply:.1f} months of inventory (buyer's market)",
            ))
    
    # Cash Flow Risk
    if financials:
        if financials.cash_on_cash_return < 3:
            factors.append(RiskFactor(
                name="Low Cash Flow",
                level=RiskLevel.HIGH,
                score=65,
                description=f"Cash-on-cash return of {financials.cash_on_cash_return:.1f}% is below 5% target",
                mitigation="Negotiate lower price or find ways to increase rent",
            ))
        elif financials.cash_on_cash_return < 5:
            factors.append(RiskFactor(
                name="Moderate Cash Flow",
                level=RiskLevel.MEDIUM,
                score=40,
                description=f"Cash-on-cash return of {financials.cash_on_cash_return:.1f}%",
            ))
        else:
            factors.append(RiskFactor(
                name="Strong Cash Flow",
                level=RiskLevel.LOW,
                score=15,
                description=f"Cash-on-cash return of {financials.cash_on_cash_return:.1f}%",
            ))
        
        # DSCR Risk
        if financials.debt_service_coverage < 1.2:
            factors.append(RiskFactor(
                name="Debt Coverage",
                level=RiskLevel.HIGH,
                score=70,
                description=f"DSCR of {financials.debt_service_coverage:.2f} is below 1.25 threshold",
                mitigation="Increase down payment or negotiate lower price",
            ))
    
    # Condition Risk
    condition_risk = {
        "Excellent": (RiskLevel.LOW, 10, "Property in excellent condition"),
        "Good": (RiskLevel.LOW, 20, "Property in good condition"),
        "Fair": (RiskLevel.MEDIUM, 40, "Property may need some updates"),
        "Poor": (RiskLevel.HIGH, 70, "Property needs significant repairs"),
        "Needs Renovation": (RiskLevel.HIGH, 80, "Major renovation required"),
    }
    
    cond = property.condition.value
    if cond in condition_risk:
        level, score, desc = condition_risk[cond]
        factors.append(RiskFactor(
            name="Property Condition",
            level=level,
            score=score,
            description=desc,
            mitigation="Get professional inspection and repair estimates" if level != RiskLevel.LOW else "",
        ))
    
    # Age Risk
    if property.features.year_built:
        age = datetime.now().year - property.features.year_built
        if age > 50:
            factors.append(RiskFactor(
                name="Property Age",
                level=RiskLevel.MEDIUM,
                score=45,
                description=f"Property is {age} years old. Major systems may need replacement.",
                mitigation="Budget for roof, HVAC, plumbing updates",
            ))
    
    assessment = RiskAssessment(factors=factors)
    assessment.calculate_overall()
    
    return assessment


# =============================================================================
# Legal Analysis Tools
# =============================================================================

def check_legal(property: Property) -> LegalAnalysis:
    """Run legal checks on property."""
    checks = legal_source.get_legal_checks(property)
    zoning = legal_source.check_zoning(property)
    permits = legal_source.check_permits(property)
    title = legal_source.check_title(property)
    
    return LegalAnalysis(
        checks=checks,
        zoning_code=zoning["zoning_code"],
        zoning_description=zoning["zoning_description"],
        zoning_compliant=zoning["residential_allowed"],
        open_permits=len(permits),
        permit_issues=[p["type"] for p in permits if p["status"] == "Open"],
        title_clear=title["title_clear"],
        liens=[f"{l['type']}: ${l['amount']:,}" for l in title.get("liens", [])],
    )


# =============================================================================
# Recommendation Tools
# =============================================================================

def generate_recommendation(
    property: Property,
    valuation: Valuation,
    market: MarketData,
    financials: FinancialAnalysis,
    risks: RiskAssessment,
    legal: LegalAnalysis,
) -> InvestmentRecommendation:
    """Generate final investment recommendation."""
    
    pros = []
    cons = []
    conditions = []
    
    # Analyze valuation
    price_vs_value = (property.list_price - valuation.estimated_value) / valuation.estimated_value
    
    if price_vs_value < -0.05:
        pros.append(f"Listed below estimated value by {abs(price_vs_value)*100:.1f}%")
    elif price_vs_value > 0.05:
        cons.append(f"Listed above estimated value by {price_vs_value*100:.1f}%")
    
    # Analyze financials
    if financials.cap_rate > 5:
        pros.append(f"Strong cap rate of {financials.cap_rate:.1f}%")
    elif financials.cap_rate < 4:
        cons.append(f"Low cap rate of {financials.cap_rate:.1f}%")
    
    if financials.cash_on_cash_return > 7:
        pros.append(f"Excellent cash-on-cash return of {financials.cash_on_cash_return:.1f}%")
    elif financials.cash_on_cash_return < 4:
        cons.append(f"Low cash-on-cash return of {financials.cash_on_cash_return:.1f}%")
    
    # Analyze market
    if market:
        if market.investment_score > 75:
            pros.append(f"Strong investment market (score: {market.investment_score})")
        elif market.investment_score < 50:
            cons.append(f"Weak investment market (score: {market.investment_score})")
        
        if market.metrics and market.metrics.population_growth > 2:
            pros.append(f"Growing population ({market.metrics.population_growth}% growth)")
    
    # Analyze risks
    high_risks = [r for r in risks.factors if r.level == RiskLevel.HIGH]
    if high_risks:
        for risk in high_risks:
            cons.append(f"{risk.name}: {risk.description}")
    
    # Legal issues
    if not legal.title_clear:
        cons.append("Title issues found")
        conditions.append("Title must be cleared before closing")
    
    if legal.open_permits > 0:
        conditions.append(f"Verify {legal.open_permits} open permit(s) are resolved")
    
    # Calculate overall score
    score = 50  # Base score
    
    # Valuation impact
    score += int(-price_vs_value * 100)
    
    # Cap rate impact
    score += int((financials.cap_rate - 4.5) * 10)
    
    # Cash flow impact
    score += int((financials.cash_on_cash_return - 5) * 5)
    
    # Market impact
    if market:
        score += int((market.investment_score - 70) * 0.3)
    
    # Risk impact
    score -= risks.overall_risk_score // 5
    
    # Legal impact
    if not legal.title_clear:
        score -= 20
    
    score = max(0, min(100, score))
    
    # Determine recommendation
    if score >= 75:
        recommendation = Recommendation.STRONG_BUY
        summary = "This property presents an excellent investment opportunity with strong fundamentals."
    elif score >= 60:
        recommendation = Recommendation.BUY
        summary = "This property is a good investment with solid returns potential."
    elif score >= 45:
        recommendation = Recommendation.HOLD
        summary = "This property has mixed indicators. Consider negotiating better terms."
    elif score >= 30:
        recommendation = Recommendation.AVOID
        summary = "This property has significant concerns that outweigh potential benefits."
    else:
        recommendation = Recommendation.STRONG_AVOID
        summary = "This property is not recommended as an investment at current terms."
    
    # Calculate suggested offer
    if score >= 45:
        suggested_offer = min(property.list_price, valuation.estimated_value * 0.97)
        max_price = valuation.estimated_value * 1.02
    else:
        suggested_offer = valuation.low_estimate * 0.95
        max_price = valuation.low_estimate
    
    return InvestmentRecommendation(
        recommendation=recommendation,
        confidence=min(95, score + 20),
        summary=summary,
        pros=pros,
        cons=cons,
        suggested_offer=suggested_offer,
        max_price=max_price,
        conditions=conditions,
    )
