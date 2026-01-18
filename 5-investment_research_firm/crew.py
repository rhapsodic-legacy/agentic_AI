"""
Investment Research Firm - Research Crew

Main orchestration for investment research using CrewAI.
Hub-and-Spoke architecture with Research Director coordinating specialists. 
"""

from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

try:
    from crewai import Crew, Task, Process
except ImportError:
    raise ImportError("Install crewai: pip install crewai")

from .agents import InvestmentTeam
from .data_sources import MarketDataManager
from .analysis import TechnicalAnalyzer, FundamentalAnalyzer, SentimentAnalyzer, RiskAnalyzer
from .models import (
    Stock, InvestmentMemo, InvestmentThesis, Catalyst, RiskFactor,
    Recommendation, RiskLevel, Sector
)


@dataclass
class ResearchConfig:
    """Configuration for the research crew."""
    llm_provider: str = "gemini"
    llm_model: Optional[str] = None
    use_live_data: bool = True
    verbose: bool = True
    max_iterations: int = 10


@dataclass
class ResearchResult:
    """Result from a research task."""
    success: bool
    symbol: str
    company_name: str
    
    # Outputs
    memo: Optional[InvestmentMemo] = None
    raw_research: dict = field(default_factory=dict)
    
    # Metadata
    analysts_involved: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    errors: list[str] = field(default_factory=list)


class InvestmentResearchCrew:
    """
    Investment Research Crew
    
    Hub-and-Spoke architecture for comprehensive investment research:
    
                    ┌─────────────────────┐
                    │  RESEARCH DIRECTOR  │
                    │  (Assigns & Reviews)│
                    └──────────┬──────────┘
                               │
    ┌──────────────────────────┼──────────────────────────┐
    │                          │                          │
    ▼                          ▼                          ▼
    MACRO              EQUITY/SECTOR                   QUANT
    ANALYST              ANALYSTS                   RESEARCHER
                               │
                               ▼
                    ┌─────────────────────┐
                    │  PORTFOLIO MANAGER  │
                    │  (Final Decisions)  │
                    └─────────────────────┘
    
    Usage:
        crew = InvestmentResearchCrew()
        result = crew.research_stock("NVDA")
        print(result.memo.to_markdown())
    """
    
    def __init__(self, config: Optional[ResearchConfig] = None):
        self.config = config or ResearchConfig()
        
        # Initialize team
        self.team = InvestmentTeam(
            llm_provider=self.config.llm_provider,
            llm_model=self.config.llm_model,
        )
        
        # Initialize data and analysis
        self.data_manager = MarketDataManager()
        self.technical_analyzer = TechnicalAnalyzer()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_analyzer = RiskAnalyzer()
    
    def research_stock(self, symbol: str) -> ResearchResult:
        """
        Conduct comprehensive research on a stock.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            ResearchResult with investment memo
        """
        import time
        start_time = time.time()
        
        # Get initial data
        stock = self.data_manager.get_stock(symbol, use_live=self.config.use_live_data)
        
        # Determine sector analyst
        sector_analyst = self.team.equity_analyst
        if stock.sector:
            sector_analyst = self.team.get_sector_analyst(stock.sector.value)
        
        # Create tasks
        tasks = self._create_research_tasks(symbol, stock, sector_analyst)
        
        # Create crew
        crew = Crew(
            agents=[
                self.team.research_director,
                self.team.macro_analyst,
                self.team.equity_analyst,
                sector_analyst,
                self.team.quant_researcher,
                self.team.portfolio_manager,
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=self.config.verbose,
        )
        
        # Execute
        try:
            result = crew.kickoff()
            
            # Generate investment memo from results
            memo = self._generate_memo(symbol, stock, result)
            
            return ResearchResult(
                success=True,
                symbol=symbol,
                company_name=stock.name,
                memo=memo,
                raw_research={"crew_output": str(result)},
                analysts_involved=[
                    "Research Director", "Macro Analyst", "Equity Analyst",
                    sector_analyst.role, "Quant Researcher", "Portfolio Manager"
                ],
                execution_time=time.time() - start_time,
            )
            
        except Exception as e:
            return ResearchResult(
                success=False,
                symbol=symbol,
                company_name=stock.name,
                errors=[str(e)],
                execution_time=time.time() - start_time,
            )
    
    def _create_research_tasks(self, symbol: str, stock: Stock, sector_analyst) -> list[Task]:
        """Create the research task pipeline."""
        
        # Task 1: Research Director assigns and outlines research
        task_assign = Task(
            description=f"""Initiate research on {symbol} ({stock.name}).
            
            1. Gather initial stock information
            2. Identify key research questions
            3. Outline the research approach
            4. Assign specific focus areas to the team
            
            Provide a research brief that will guide the analysis.""",
            expected_output="Research brief with key questions and approach",
            agent=self.team.research_director,
        )
        
        # Task 2: Macro analyst provides economic context
        task_macro = Task(
            description=f"""Analyze the macroeconomic context for {symbol}:
            
            1. Current economic environment and outlook
            2. Sector-specific macro factors
            3. Interest rate and inflation impact
            4. Any regulatory or policy considerations
            
            Focus on factors most relevant to {stock.sector.value if stock.sector else 'this'} sector.""",
            expected_output="Macro analysis with sector implications",
            agent=self.team.macro_analyst,
        )
        
        # Task 3: Equity analyst does fundamental analysis
        task_fundamental = Task(
            description=f"""Conduct fundamental analysis of {symbol}:
            
            1. Get comprehensive stock information
            2. Analyze financial statements
            3. Evaluate valuation (P/E, PEG, DCF)
            4. Assess competitive position
            5. Identify growth drivers and risks
            
            Use the fundamental_analysis and dcf_valuation tools.""",
            expected_output="Fundamental analysis with valuation assessment",
            agent=self.team.equity_analyst,
        )
        
        # Task 4: Sector analyst provides industry context
        task_sector = Task(
            description=f"""Provide sector-specific analysis for {symbol}:
            
            1. Industry competitive dynamics
            2. Compare to key peers
            3. Sector-specific trends and catalysts
            4. Regulatory or technology changes impacting the sector
            
            Use the compare_stocks tool to benchmark against peers.""",
            expected_output="Sector analysis with peer comparison",
            agent=sector_analyst,
        )
        
        # Task 5: Quant researcher does technical and risk analysis
        task_quant = Task(
            description=f"""Perform quantitative analysis of {symbol}:
            
            1. Technical analysis (trends, indicators, patterns)
            2. Risk assessment (volatility, drawdown risk)
            3. Statistical analysis of returns
            4. Support/resistance levels and price targets
            
            Use technical_analysis and risk_assessment tools.""",
            expected_output="Quantitative analysis with technical signals and risk metrics",
            agent=self.team.quant_researcher,
        )
        
        # Task 6: Portfolio manager synthesizes and decides
        task_portfolio = Task(
            description=f"""Synthesize all research on {symbol} and make investment recommendation:
            
            Based on:
            - Macro analysis
            - Fundamental analysis and valuation
            - Sector dynamics
            - Technical signals and risk assessment
            
            Provide:
            1. Investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
            2. Target price with bull/base/bear scenarios
            3. Key catalysts and timeline
            4. Primary risks and mitigation
            5. Position sizing recommendation
            
            Format as a clear investment recommendation.""",
            expected_output="Investment recommendation with target price, catalysts, and risks",
            agent=self.team.portfolio_manager,
        )
        
        return [task_assign, task_macro, task_fundamental, task_sector, task_quant, task_portfolio]
    
    def _generate_memo(self, symbol: str, stock: Stock, crew_result) -> InvestmentMemo:
        """Generate an investment memo from research results."""
        
        # Get additional analysis
        prices = self.data_manager.get_price_history(symbol)
        financials = self.data_manager.get_financials(symbol)
        technical = self.technical_analyzer.analyze(symbol, prices)
        fundamental = self.fundamental_analyzer.analyze(stock, financials)
        news = self.data_manager.get_news(symbol)
        sentiment = self.sentiment_analyzer.analyze(symbol, news)
        valuation = self.fundamental_analyzer.dcf_valuation(stock, financials)
        risk = self.risk_analyzer.assess_risk(stock, technical, fundamental)
        
        # Determine recommendation based on analysis
        quality_score = fundamental.get("quality_score", 50)
        upside = valuation.upside_potential
        
        if upside > 0.20 and quality_score > 70:
            recommendation = Recommendation.STRONG_BUY
        elif upside > 0.10 and quality_score > 50:
            recommendation = Recommendation.BUY
        elif upside < -0.10 or quality_score < 30:
            recommendation = Recommendation.SELL
        elif upside < -0.20 and quality_score < 40:
            recommendation = Recommendation.STRONG_SELL
        else:
            recommendation = Recommendation.HOLD
        
        # Create thesis
        thesis = InvestmentThesis(
            summary=f"{stock.name} presents {'an attractive' if upside > 0.1 else 'a challenging'} "
                   f"investment opportunity with {upside*100:.0f}% potential upside based on DCF analysis.",
            key_points=[
                f"Quality score of {quality_score:.0f}/100 indicates {'strong' if quality_score > 60 else 'moderate'} fundamentals",
                f"Technical trend is {technical.trend}",
                f"Sentiment is {sentiment.overall_label}",
            ],
            competitive_advantage=f"{stock.name} operates in the {stock.industry or 'industry'} with "
                                 f"{'strong' if stock.gross_margin and stock.gross_margin > 0.5 else 'moderate'} margins",
            growth_drivers=[
                f"Revenue growth of {stock.revenue_growth*100:.0f}%" if stock.revenue_growth else "Market expansion",
                "Continued innovation and product development",
                "Industry tailwinds",
            ],
            concerns=[
                rf["title"] for rf in risk.get("risk_factors", [])[:3]
            ] or ["Market volatility"],
        )
        
        # Create catalysts
        catalysts = [
            Catalyst(
                title="Earnings Release",
                expected_date="Next Quarter",
                description="Upcoming quarterly earnings could drive price action",
                impact="positive" if sentiment.overall_label == "positive" else "unknown",
                probability="high",
            ),
            Catalyst(
                title="Product Launch/Update",
                expected_date="Next 6 Months",
                description="Potential new product or service announcement",
                impact="positive",
                probability="medium",
            ),
        ]
        
        # Create risks
        risks = [
            RiskFactor(
                category=rf.get("category", "market"),
                title=rf.get("title", "Risk"),
                description=rf.get("description", ""),
                severity=rf.get("severity", "medium"),
            )
            for rf in risk.get("risk_factors", [])
        ]
        
        if not risks:
            risks = [
                RiskFactor(
                    category="market",
                    title="Market Risk",
                    description="General market volatility could impact the stock",
                    severity="medium",
                )
            ]
        
        # Key metrics
        key_metrics = {
            "P/E Ratio": {
                "value": f"{stock.pe_ratio:.1f}x" if stock.pe_ratio else "N/A",
                "industry_avg": "20x",
            },
            "Forward P/E": {
                "value": f"{stock.forward_pe:.1f}x" if stock.forward_pe else "N/A",
                "industry_avg": "18x",
            },
            "Revenue Growth": {
                "value": f"{stock.revenue_growth*100:.0f}%" if stock.revenue_growth else "N/A",
                "industry_avg": "10%",
            },
            "Gross Margin": {
                "value": f"{stock.gross_margin*100:.0f}%" if stock.gross_margin else "N/A",
                "industry_avg": "40%",
            },
            "ROE": {
                "value": f"{stock.return_on_equity*100:.0f}%" if stock.return_on_equity else "N/A",
                "industry_avg": "15%",
            },
        }
        
        # Executive summary
        exec_summary = f"""{stock.name} ({symbol}) is rated {recommendation.value.upper().replace('_', ' ')} 
with a target price of ${valuation.fair_value:.2f}, representing {upside*100:.0f}% upside from the current 
price of ${stock.current_price:.2f}. The company scores {quality_score:.0f}/100 on our fundamental quality 
assessment. Key drivers include {'strong' if stock.revenue_growth and stock.revenue_growth > 0.15 else 'moderate'} 
revenue growth and {'expanding' if stock.gross_margin and stock.gross_margin > 0.4 else 'stable'} margins. 
Primary risks include {risks[0].title if risks else 'market volatility'}."""
        
        return InvestmentMemo(
            symbol=symbol,
            company_name=stock.name,
            date=datetime.now().strftime("%Y-%m-%d"),
            analyst="AI Investment Research Team",
            recommendation=recommendation,
            target_price=valuation.fair_value,
            current_price=stock.current_price,
            upside=upside,
            thesis=thesis,
            fundamental_analysis=fundamental,
            technical_analysis=technical,
            sentiment_analysis=sentiment,
            valuation=valuation,
            catalysts=catalysts,
            risks=risks,
            key_metrics=key_metrics,
            bull_case_price=valuation.bull_case,
            base_case_price=valuation.base_case,
            bear_case_price=valuation.bear_case,
            risk_level=risk.get("risk_level", RiskLevel.MEDIUM),
            position_size_recommendation=self._get_position_recommendation(risk.get("risk_level", RiskLevel.MEDIUM)),
            executive_summary=exec_summary,
        )
    
    def _get_position_recommendation(self, risk_level: RiskLevel) -> str:
        """Get position size recommendation based on risk."""
        recommendations = {
            RiskLevel.LOW: "Full position (3-5% of portfolio)",
            RiskLevel.MEDIUM: "Standard position (2-3% of portfolio)",
            RiskLevel.HIGH: "Reduced position (1-2% of portfolio)",
            RiskLevel.VERY_HIGH: "Small position only (0.5-1% of portfolio)",
        }
        return recommendations.get(risk_level, "Standard position (2-3% of portfolio)")
    
    def quick_analysis(self, symbol: str) -> dict:
        """
        Quick analysis without full crew execution.
        
        Returns key metrics and signals for rapid screening.
        """
        stock = self.data_manager.get_stock(symbol, use_live=self.config.use_live_data)
        prices = self.data_manager.get_price_history(symbol, "6mo")
        financials = self.data_manager.get_financials(symbol)
        
        technical = self.technical_analyzer.analyze(symbol, prices)
        fundamental = self.fundamental_analyzer.analyze(stock, financials)
        valuation = self.fundamental_analyzer.dcf_valuation(stock, financials)
        
        return {
            "symbol": symbol,
            "name": stock.name,
            "price": stock.current_price,
            "target": valuation.fair_value,
            "upside": f"{valuation.upside_potential*100:.1f}%",
            "quality_score": fundamental.get("quality_score", 0),
            "trend": technical.trend,
            "rsi": technical.rsi_14,
            "pe_ratio": stock.pe_ratio,
            "revenue_growth": f"{stock.revenue_growth*100:.0f}%" if stock.revenue_growth else "N/A",
        }
    
    def screen_stocks(self, symbols: list[str]) -> list[dict]:
        """
        Screen multiple stocks for quick comparison.
        
        Args:
            symbols: List of stock ticker symbols
        
        Returns:
            List of quick analysis results
        """
        results = []
        for symbol in symbols:
            try:
                analysis = self.quick_analysis(symbol)
                results.append(analysis)
            except Exception as e:
                results.append({"symbol": symbol, "error": str(e)})
        
        # Sort by upside potential
        results.sort(key=lambda x: float(x.get("upside", "0%").replace("%", "")) if "upside" in x else -999, reverse=True)
        
        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def research_stock(symbol: str, provider: str = "gemini") -> InvestmentMemo:
    """
    Quick function to research a stock.
    
    Args:
        symbol: Stock ticker symbol
        provider: LLM provider
    
    Returns:
        InvestmentMemo
    """
    config = ResearchConfig(llm_provider=provider)
    crew = InvestmentResearchCrew(config)
    result = crew.research_stock(symbol)
    return result.memo


def quick_screen(symbols: list[str]) -> list[dict]:
    """
    Quick screen multiple stocks.
    
    Args:
        symbols: List of stock symbols
    
    Returns:
        List of screening results
    """
    crew = InvestmentResearchCrew()
    return crew.screen_stocks(symbols)
