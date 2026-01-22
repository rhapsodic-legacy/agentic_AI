"""
Real Estate Investment Advisor - Main Module

FastAPI + LangChain powered real estate investment analysis.

Architecture (Supervisor Pattern):
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

Usage:
    from real_estate import RealEstateAdvisor
    
    advisor = RealEstateAdvisor()
    
    # Search for properties
    properties = advisor.search_properties(city="Austin", max_price=500000)
    
    # Analyze a specific property
    analysis = advisor.analyze_property(properties[0])
    
    # View report
    print(analysis.to_markdown())
"""

from typing import Optional
from dataclasses import dataclass
from datetime import datetime
import uuid

from .models import (
    Property, PropertyType, PropertyCondition, PropertyFeatures, Address,
    MarketData, Valuation, FinancialAnalysis, RiskAssessment, LegalAnalysis,
    InvestmentRecommendation, InvestmentAnalysis
)
from .agents import SupervisorAgent, AgentConfig
from .data import property_source, market_source


@dataclass
class AdvisorConfig:
    """Configuration for the advisor."""
    llm_provider: str = "gemini"
    default_down_payment: float = 0.25
    default_interest_rate: float = 0.07
    verbose: bool = True


class RealEstateAdvisor:
    """
    Real Estate Investment Advisor
    
    An AI advisor that analyzes real estate markets, evaluates properties,
    calculates ROI, and provides investment recommendations.
    
    Example:
        advisor = RealEstateAdvisor()
        
        # Search for properties
        properties = advisor.search_properties(city="Austin", max_price=500000)
        
        # Analyze a property
        analysis = advisor.analyze_property(properties[0])
        
        # Get recommendation
        print(f"Recommendation: {analysis.recommendation.recommendation.value}")
        print(analysis.to_markdown())
    """
    
    def __init__(self, config: Optional[AdvisorConfig] = None):
        self.config = config or AdvisorConfig()
        
        # Initialize supervisor agent
        agent_config = AgentConfig(
            llm_provider=self.config.llm_provider,
            verbose=self.config.verbose,
        )
        self.supervisor = SupervisorAgent(agent_config)
    
    def search_properties(
        self,
        city: str = None,
        min_price: float = 0,
        max_price: float = float('inf'),
        min_beds: int = 0,
        property_type: PropertyType = None,
    ) -> list[Property]:
        """
        Search for properties.
        
        Args:
            city: City name
            min_price: Minimum price
            max_price: Maximum price
            min_beds: Minimum bedrooms
            property_type: Property type filter
        
        Returns:
            List of matching properties
        """
        return property_source.search_properties(
            city=city,
            min_price=min_price,
            max_price=max_price,
            min_beds=min_beds,
            property_type=property_type,
        )
    
    def get_property(self, property_id: str) -> Optional[Property]:
        """Get a property by ID."""
        return property_source.get_property(property_id)
    
    def get_market_data(self, city: str) -> MarketData:
        """Get market data for a city."""
        return market_source.get_market_data(city)
    
    def analyze_property(
        self,
        property: Property,
        down_payment_pct: float = None,
        interest_rate: float = None,
    ) -> InvestmentAnalysis:
        """
        Run complete investment analysis on a property.
        
        Args:
            property: Property to analyze
            down_payment_pct: Down payment percentage (default: 25%)
            interest_rate: Interest rate (default: 7%)
        
        Returns:
            InvestmentAnalysis with full report
        """
        down_payment = down_payment_pct or self.config.default_down_payment
        rate = interest_rate or self.config.default_interest_rate
        
        if self.config.verbose:
            print("\n" + "="*60)
            print("🏠 REAL ESTATE INVESTMENT ADVISOR")
            print("="*60 + "\n")
        
        # Run analysis with supervisor
        results = self.supervisor.analyze_property(
            property,
            down_payment_pct=down_payment,
            interest_rate=rate,
        )
        
        # Create analysis report
        analysis = InvestmentAnalysis(
            analysis_id=f"analysis-{uuid.uuid4().hex[:8]}",
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            property=property,
            valuation=results["valuation"],
            market_data=results["market_data"],
            financial_analysis=results["financials"],
            risk_assessment=results["risk_assessment"],
            legal_analysis=results["legal_analysis"],
            recommendation=results["recommendation"],
        )
        
        if self.config.verbose:
            rec = results["recommendation"]
            print(f"\n{'='*60}")
            print(f"📋 RECOMMENDATION: {rec.recommendation_emoji} {rec.recommendation.value}")
            print(f"{'='*60}\n")
        
        return analysis
    
    def quick_analysis(self, property: Property) -> dict:
        """
        Quick analysis without full agent orchestration.
        
        Returns summary metrics only.
        """
        from .tools import (
            get_property_valuation, get_market_analysis,
            calculate_financial_analysis
        )
        
        valuation = get_property_valuation(property)
        market = get_market_analysis(property.address.city)
        financials = calculate_financial_analysis(property)
        
        return {
            "property": property.to_dict(),
            "estimated_value": valuation.estimated_value,
            "market_score": market.investment_score,
            "cap_rate": financials.cap_rate,
            "cash_on_cash": financials.cash_on_cash_return,
            "monthly_rent": financials.rental_estimate.monthly_rent,
        }
    
    def compare_properties(self, properties: list[Property]) -> list[dict]:
        """
        Compare multiple properties side by side.
        
        Returns sorted list by investment potential.
        """
        comparisons = []
        
        for prop in properties:
            quick = self.quick_analysis(prop)
            
            # Calculate simple score
            score = (
                quick["cap_rate"] * 10 +
                quick["cash_on_cash"] * 5 +
                quick["market_score"] * 0.5
            )
            
            comparisons.append({
                **quick,
                "score": score,
            })
        
        return sorted(comparisons, key=lambda x: x["score"], reverse=True)
    
    def create_custom_property(
        self,
        address: str,
        city: str,
        state: str,
        zip_code: str,
        price: float,
        bedrooms: int,
        bathrooms: float,
        sqft: int,
        lot_sqft: int = 0,
        year_built: int = 2000,
        property_type: str = "Single Family",
    ) -> Property:
        """
        Create a custom property for analysis.
        
        Use this to analyze properties not in the database.
        """
        type_map = {
            "single family": PropertyType.SINGLE_FAMILY,
            "multi-family": PropertyType.MULTI_FAMILY,
            "condo": PropertyType.CONDO,
            "townhouse": PropertyType.TOWNHOUSE,
            "duplex": PropertyType.DUPLEX,
        }
        
        prop_type = type_map.get(property_type.lower(), PropertyType.SINGLE_FAMILY)
        
        prop = Property(
            property_id=f"custom-{uuid.uuid4().hex[:8]}",
            address=Address(
                street=address,
                city=city,
                state=state,
                zip_code=zip_code,
            ),
            property_type=prop_type,
            features=PropertyFeatures(
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                sqft=sqft,
                lot_sqft=lot_sqft,
                year_built=year_built,
            ),
            list_price=price,
        )
        
        # Add to database
        property_source.add_property(prop)
        
        return prop


# =============================================================================
# Convenience Functions
# =============================================================================

def create_advisor(provider: str = "gemini", verbose: bool = True) -> RealEstateAdvisor:
    """Create a real estate advisor."""
    config = AdvisorConfig(llm_provider=provider, verbose=verbose)
    return RealEstateAdvisor(config)


def analyze_address(
    address: str,
    city: str,
    state: str,
    price: float,
    bedrooms: int,
    bathrooms: float,
    sqft: int,
) -> InvestmentAnalysis:
    """
    Quick function to analyze a property by address.
    
    Example:
        analysis = analyze_address(
            address="123 Main St",
            city="Austin",
            state="TX",
            price=450000,
            bedrooms=3,
            bathrooms=2,
            sqft=1850,
        )
    """
    advisor = create_advisor()
    
    prop = advisor.create_custom_property(
        address=address,
        city=city,
        state=state,
        zip_code="00000",
        price=price,
        bedrooms=bedrooms,
        bathrooms=bathrooms,
        sqft=sqft,
    )
    
    return advisor.analyze_property(prop)
