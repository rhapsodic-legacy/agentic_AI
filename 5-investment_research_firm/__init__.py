"""
Investment Research Firm - CrewAI Agents

Hub-and-Spoke organization:
- Research Director (hub) - assigns and reviews
- Macro Analyst - economic analysis
- Equity Analyst - stock analysis
- Sector Analysts - Tech, Finance, Healthcare
- Quant Researcher - quantitative analysis
- Portfolio Manager - final decisions
"""

from typing import Optional
import os

try:
    from crewai import Agent, Crew, Task, Process
except ImportError:
    raise ImportError("Install crewai: pip install crewai")

from ..tools import get_tools_for_role


def create_llm(provider: str = "gemini", model: str = None):
    """Create LLM instance for agents."""
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash",
            temperature=0.3,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-sonnet-4-20250514",
            temperature=0.3,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=0.3,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================================
# Agent Definitions
# =============================================================================

def create_research_director(llm) -> Agent:
    """Create the Research Director agent."""
    return Agent(
        role="Research Director",
        goal="""Coordinate investment research activities, assign research tasks to specialists,
        review their findings, and ensure high-quality investment recommendations.""",
        backstory="""You are an experienced Research Director with 20+ years on Wall Street.
        You've led research teams at top investment banks and hedge funds. You're known for
        your rigorous analytical standards and ability to synthesize complex information
        into actionable investment insights. You manage a team of specialized analysts
        and ensure all research meets institutional quality standards.""",
        tools=get_tools_for_role("research_director"),
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )


def create_macro_analyst(llm) -> Agent:
    """Create the Macro Analyst agent."""
    return Agent(
        role="Macro Analyst",
        goal="""Analyze macroeconomic trends, monetary policy, and global economic conditions
        to provide context for investment decisions.""",
        backstory="""You are a seasoned macroeconomist who previously worked at the Federal Reserve
        and major investment banks. You have deep expertise in monetary policy, fiscal policy,
        and global economic dynamics. You provide the economic backdrop that informs all
        investment decisions, identifying how macro trends will impact different sectors
        and asset classes.""",
        tools=get_tools_for_role("macro_analyst"),
        llm=llm,
        verbose=True,
    )


def create_equity_analyst(llm) -> Agent:
    """Create the Equity Analyst agent."""
    return Agent(
        role="Senior Equity Analyst",
        goal="""Conduct comprehensive fundamental analysis of individual stocks,
        including financial statement analysis, competitive positioning, and valuation.""",
        backstory="""You are a CFA charterholder with 15 years of equity research experience.
        You've covered multiple sectors and have a strong track record of identifying
        undervalued stocks. You're meticulous in financial modeling and known for your
        detailed company analysis that uncovers insights others miss.""",
        tools=get_tools_for_role("equity_analyst"),
        llm=llm,
        verbose=True,
    )


def create_tech_sector_analyst(llm) -> Agent:
    """Create the Technology Sector Analyst agent."""
    return Agent(
        role="Technology Sector Analyst",
        goal="""Provide deep expertise on technology companies including software,
        semiconductors, internet, and hardware sectors.""",
        backstory="""You are a technology sector specialist with both engineering and finance
        backgrounds. You understand the technical aspects of products and can assess
        competitive moats in technology. You've covered tech stocks through multiple
        cycles and understand the unique dynamics of high-growth tech companies,
        including cloud computing, AI/ML, and semiconductors.""",
        tools=get_tools_for_role("sector_analyst"),
        llm=llm,
        verbose=True,
    )


def create_finance_sector_analyst(llm) -> Agent:
    """Create the Financial Sector Analyst agent."""
    return Agent(
        role="Financial Sector Analyst",
        goal="""Analyze financial services companies including banks, insurance,
        asset managers, and fintech companies.""",
        backstory="""You are a financial sector expert who previously worked in banking
        and now covers the sector as an analyst. You understand bank balance sheets,
        credit cycles, regulatory dynamics, and the competitive landscape of financial
        services. You can assess credit quality, capital adequacy, and the impact of
        interest rates on financial companies.""",
        tools=get_tools_for_role("sector_analyst"),
        llm=llm,
        verbose=True,
    )


def create_healthcare_sector_analyst(llm) -> Agent:
    """Create the Healthcare Sector Analyst agent."""
    return Agent(
        role="Healthcare Sector Analyst",
        goal="""Analyze healthcare companies including pharmaceuticals, biotechnology,
        medical devices, and healthcare services.""",
        backstory="""You have a PhD in biochemistry and transitioned to equity research.
        You can evaluate drug pipelines, understand clinical trial results, and assess
        the commercial potential of therapies. You're familiar with FDA processes,
        patent cliffs, and the complex dynamics of healthcare reimbursement.""",
        tools=get_tools_for_role("sector_analyst"),
        llm=llm,
        verbose=True,
    )


def create_quant_researcher(llm) -> Agent:
    """Create the Quantitative Researcher agent."""
    return Agent(
        role="Quantitative Researcher",
        goal="""Apply quantitative methods to analyze stocks including technical analysis,
        statistical models, and risk metrics.""",
        backstory="""You have a PhD in financial engineering and have built quantitative
        trading systems at top hedge funds. You combine technical analysis with
        statistical methods to identify patterns and assess risk. You're proficient
        in factor models, volatility analysis, and portfolio optimization.""",
        tools=get_tools_for_role("quant_researcher"),
        llm=llm,
        verbose=True,
    )


def create_portfolio_manager(llm) -> Agent:
    """Create the Portfolio Manager agent."""
    return Agent(
        role="Portfolio Manager",
        goal="""Synthesize research from all analysts to make final investment recommendations
        and portfolio allocation decisions.""",
        backstory="""You are a seasoned portfolio manager who has managed billions in assets.
        You excel at synthesizing diverse viewpoints into coherent investment decisions.
        You balance conviction with risk management and understand how individual positions
        fit into overall portfolio construction. You make the final call on buy/sell/hold
        recommendations and position sizing.""",
        tools=get_tools_for_role("portfolio_manager"),
        llm=llm,
        verbose=True,
    )


# =============================================================================
# Agent Factory
# =============================================================================

class InvestmentTeam:
    """
    Factory for creating the investment research team.
    """
    
    def __init__(self, llm_provider: str = "gemini", llm_model: str = None):
        self.llm = create_llm(llm_provider, llm_model)
        self._agents = {}
    
    @property
    def research_director(self) -> Agent:
        if "research_director" not in self._agents:
            self._agents["research_director"] = create_research_director(self.llm)
        return self._agents["research_director"]
    
    @property
    def macro_analyst(self) -> Agent:
        if "macro_analyst" not in self._agents:
            self._agents["macro_analyst"] = create_macro_analyst(self.llm)
        return self._agents["macro_analyst"]
    
    @property
    def equity_analyst(self) -> Agent:
        if "equity_analyst" not in self._agents:
            self._agents["equity_analyst"] = create_equity_analyst(self.llm)
        return self._agents["equity_analyst"]
    
    @property
    def tech_analyst(self) -> Agent:
        if "tech_analyst" not in self._agents:
            self._agents["tech_analyst"] = create_tech_sector_analyst(self.llm)
        return self._agents["tech_analyst"]
    
    @property
    def finance_analyst(self) -> Agent:
        if "finance_analyst" not in self._agents:
            self._agents["finance_analyst"] = create_finance_sector_analyst(self.llm)
        return self._agents["finance_analyst"]
    
    @property
    def healthcare_analyst(self) -> Agent:
        if "healthcare_analyst" not in self._agents:
            self._agents["healthcare_analyst"] = create_healthcare_sector_analyst(self.llm)
        return self._agents["healthcare_analyst"]
    
    @property
    def quant_researcher(self) -> Agent:
        if "quant_researcher" not in self._agents:
            self._agents["quant_researcher"] = create_quant_researcher(self.llm)
        return self._agents["quant_researcher"]
    
    @property
    def portfolio_manager(self) -> Agent:
        if "portfolio_manager" not in self._agents:
            self._agents["portfolio_manager"] = create_portfolio_manager(self.llm)
        return self._agents["portfolio_manager"]
    
    def get_sector_analyst(self, sector: str) -> Agent:
        """Get the appropriate sector analyst."""
        sector_map = {
            "technology": self.tech_analyst,
            "tech": self.tech_analyst,
            "finance": self.finance_analyst,
            "financial": self.finance_analyst,
            "healthcare": self.healthcare_analyst,
            "health": self.healthcare_analyst,
        }
        return sector_map.get(sector.lower(), self.equity_analyst)
    
    def get_all_agents(self) -> list[Agent]:
        """Get all agents."""
        return [
            self.research_director,
            self.macro_analyst,
            self.equity_analyst,
            self.tech_analyst,
            self.finance_analyst,
            self.healthcare_analyst,
            self.quant_researcher,
            self.portfolio_manager,
        ]
