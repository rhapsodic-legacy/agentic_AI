"""
E-commerce Personal Shopper - CrewAI Agents

Parallel agent architecture:
- Concierge (main interface)
- Style Advisor
- Search Agent
- Deals Finder
- Price Compare
- Review Analyzer
- Recommender
"""

from typing import Optional
import os

try:
    from crewai import Agent
except ImportError:
    raise ImportError("Install crewai: pip install crewai")

from ..tools import get_tools_for_role


def create_llm(provider: str = "gemini", model: str = None):
    """Create LLM instance for agents."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash",
            temperature=0.7,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-sonnet-4-20250514",
            temperature=0.7,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=0.7,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# Agent Definitions
# =============================================================================

def create_concierge(llm) -> Agent:
    """Create the Concierge agent - main shopping interface."""
    return Agent(
        role="Personal Shopping Concierge",
        goal="""Be a helpful and friendly personal shopping assistant. Understand what
        customers are looking for, coordinate with specialists, and provide excellent
        shopping recommendations.""",
        backstory="""You are an experienced luxury personal shopper who has worked with
        high-end clientele for years. You have an exceptional ability to understand what
        people want, even when they can't articulate it clearly. You're warm, helpful,
        and always go the extra mile to find the perfect products. You coordinate a team
        of specialists to deliver comprehensive shopping assistance.""",
        tools=get_tools_for_role("concierge"),
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )


def create_style_advisor(llm) -> Agent:
    """Create the Style Advisor agent."""
    return Agent(
        role="Style Advisor",
        goal="""Analyze user preferences and style to recommend products that match their
        personal aesthetic, lifestyle, and values.""",
        backstory="""You are a fashion-forward style consultant with expertise in current
        trends, sustainable fashion, and personal styling. You've helped countless clients
        discover their personal style and find pieces that make them feel confident.
        You understand that style is personal and there's no one-size-fits-all approach.""",
        tools=get_tools_for_role("style_advisor"),
        llm=llm,
        verbose=True,
    )


def create_search_agent(llm) -> Agent:
    """Create the Search Agent."""
    return Agent(
        role="Product Search Specialist",
        goal="""Search across multiple retailers to find products that match the customer's
        requirements. Cast a wide net to ensure no good options are missed.""",
        backstory="""You are a product research expert who knows how to navigate multiple
        retail platforms efficiently. You're thorough in your searches and always look
        at multiple sources to ensure customers have the best options. You understand
        product specifications and can quickly identify quality items.""",
        tools=get_tools_for_role("search_agent"),
        llm=llm,
        verbose=True,
    )


def create_deals_finder(llm) -> Agent:
    """Create the Deals Finder agent."""
    return Agent(
        role="Deals & Coupons Specialist",
        goal="""Find the best deals, coupons, and discounts for products. Help customers
        save money without compromising on quality.""",
        backstory="""You are obsessed with finding the best deals. You know all the tricks -
        coupon codes, price tracking, sale timing, and loyalty programs. You take pride
        in helping customers save money and always look for ways to stretch their budget.
        You stay up-to-date on current promotions across all major retailers.""",
        tools=get_tools_for_role("deals_finder"),
        llm=llm,
        verbose=True,
    )


def create_price_compare_agent(llm) -> Agent:
    """Create the Price Compare agent."""
    return Agent(
        role="Price Comparison Analyst",
        goal="""Compare prices across different retailers to find the best value.
        Track price history and identify the best time to buy.""",
        backstory="""You are a data-driven analyst who specializes in price comparison
        and market analysis. You track prices across retailers and understand pricing
        patterns. You help customers make informed decisions about when and where to buy
        for the best value.""",
        tools=get_tools_for_role("price_compare"),
        llm=llm,
        verbose=True,
    )


def create_review_analyzer(llm) -> Agent:
    """Create the Review Analyzer agent."""
    return Agent(
        role="Review & Sentiment Analyst",
        goal="""Analyze product reviews to extract genuine insights about quality, fit,
        durability, and customer satisfaction.""",
        backstory="""You are an expert at reading between the lines of product reviews.
        You can identify genuine feedback from fake reviews, understand common complaints,
        and summarize what real customers think. You pay special attention to fit
        information for clothing and practical usability for all products.""",
        tools=get_tools_for_role("review_analyzer"),
        llm=llm,
        verbose=True,
    )


def create_recommender(llm) -> Agent:
    """Create the Recommender agent."""
    return Agent(
        role="Product Recommendation Specialist",
        goal="""Synthesize all research to create compelling, personalized product
        recommendations with clear reasoning and actionable information.""",
        backstory="""You are the final decision-maker who synthesizes all the research
        into clear, actionable recommendations. You understand how to present options
        in a way that helps customers make confident decisions. You always explain your
        reasoning and highlight the key differentiators between options.""",
        tools=get_tools_for_role("recommender"),
        llm=llm,
        verbose=True,
    )


# =============================================================================
# Team Factory
# =============================================================================

class ShoppingTeam:
    """
    Factory for creating the personal shopping team.
    """
    
    def __init__(self, llm_provider: str = "gemini", llm_model: str = None):
        self.llm = create_llm(llm_provider, llm_model)
        self._agents = {}
    
    @property
    def concierge(self) -> Agent:
        if "concierge" not in self._agents:
            self._agents["concierge"] = create_concierge(self.llm)
        return self._agents["concierge"]
    
    @property
    def style_advisor(self) -> Agent:
        if "style_advisor" not in self._agents:
            self._agents["style_advisor"] = create_style_advisor(self.llm)
        return self._agents["style_advisor"]
    
    @property
    def search_agent(self) -> Agent:
        if "search_agent" not in self._agents:
            self._agents["search_agent"] = create_search_agent(self.llm)
        return self._agents["search_agent"]
    
    @property
    def deals_finder(self) -> Agent:
        if "deals_finder" not in self._agents:
            self._agents["deals_finder"] = create_deals_finder(self.llm)
        return self._agents["deals_finder"]
    
    @property
    def price_compare(self) -> Agent:
        if "price_compare" not in self._agents:
            self._agents["price_compare"] = create_price_compare_agent(self.llm)
        return self._agents["price_compare"]
    
    @property
    def review_analyzer(self) -> Agent:
        if "review_analyzer" not in self._agents:
            self._agents["review_analyzer"] = create_review_analyzer(self.llm)
        return self._agents["review_analyzer"]
    
    @property
    def recommender(self) -> Agent:
        if "recommender" not in self._agents:
            self._agents["recommender"] = create_recommender(self.llm)
        return self._agents["recommender"]
    
    def get_all_agents(self) -> list[Agent]:
        """Get all agents."""
        return [
            self.concierge,
            self.style_advisor,
            self.search_agent,
            self.deals_finder,
            self.price_compare,
            self.review_analyzer,
            self.recommender,
        ]
