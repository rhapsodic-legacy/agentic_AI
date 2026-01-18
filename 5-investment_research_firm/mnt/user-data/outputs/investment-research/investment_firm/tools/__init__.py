"""
Investment Research Firm - CrewAI Tools

Tools that agents use for research:
- Market data tools
- Analysis tools
- News tools
- Report generation tools
"""

from typing import Optional, Type
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
except ImportError:
    # Fallback for older crewai versions
    class BaseTool:
        name: str = ""
        description: str = ""
        
        def _run(self, *args, **kwargs):
            raise NotImplementedError


from ..data_sources import MarketDataManager
from ..analysis import (
    TechnicalAnalyzer, FundamentalAnalyzer, 
    SentimentAnalyzer, RiskAnalyzer
)


# Initialize shared instances
_data_manager = MarketDataManager()
_technical_analyzer = TechnicalAnalyzer()
_fundamental_analyzer = FundamentalAnalyzer()
_sentiment_analyzer = SentimentAnalyzer()
_risk_analyzer = RiskAnalyzer()


# =============================================================================
# Input Schemas
# =============================================================================

class StockInput(BaseModel):
    """Input for stock-related tools."""
    symbol: str = Field(..., description="Stock ticker symbol (e.g., AAPL, NVDA)")


class AnalysisInput(BaseModel):
    """Input for analysis tools."""
    symbol: str = Field(..., description="Stock ticker symbol")
    period: str = Field(default="1y", description="Time period for analysis")


class CompareInput(BaseModel):
    """Input for comparison tools."""
    symbols: str = Field(..., description="Comma-separated stock symbols to compare")


# =============================================================================
# Market Data Tools
# =============================================================================

class GetStockInfoTool(BaseTool):
    name: str = "get_stock_info"
    description: str = """Get comprehensive information about a stock including current price, 
    valuation metrics, margins, growth rates, and analyst ratings.
    Input: stock ticker symbol (e.g., NVDA, AAPL)"""
    args_schema: Type[BaseModel] = StockInput
    
    def _run(self, symbol: str) -> str:
        stock = _data_manager.get_stock(symbol)
        
        return f"""Stock Information: {stock.symbol} ({stock.name})

Sector: {stock.sector.value if stock.sector else 'N/A'}
Industry: {stock.industry or 'N/A'}

Current Price: ${stock.current_price:.2f}
Market Cap: ${stock.market_cap/1e9:.2f}B

Valuation:
- P/E Ratio: {stock.pe_ratio:.2f if stock.pe_ratio else 'N/A'}
- Forward P/E: {stock.forward_pe:.2f if stock.forward_pe else 'N/A'}
- PEG Ratio: {stock.peg_ratio:.2f if stock.peg_ratio else 'N/A'}
- Price/Book: {stock.price_to_book:.2f if stock.price_to_book else 'N/A'}

Growth:
- Revenue Growth: {stock.revenue_growth*100:.1f}% if stock.revenue_growth else 'N/A'
- Earnings Growth: {stock.earnings_growth*100:.1f}% if stock.earnings_growth else 'N/A'

Margins:
- Gross Margin: {stock.gross_margin*100:.1f}% if stock.gross_margin else 'N/A'
- Operating Margin: {stock.operating_margin*100:.1f}% if stock.operating_margin else 'N/A'
- Profit Margin: {stock.profit_margin*100:.1f}% if stock.profit_margin else 'N/A'

Financial Health:
- Debt/Equity: {stock.debt_to_equity:.2f if stock.debt_to_equity else 'N/A'}
- Current Ratio: {stock.current_ratio:.2f if stock.current_ratio else 'N/A'}
- ROE: {stock.return_on_equity*100:.1f}% if stock.return_on_equity else 'N/A'

Analyst:
- Target Price: ${stock.analyst_target_price:.2f if stock.analyst_target_price else 'N/A'}
- Rating: {stock.analyst_rating or 'N/A'}
"""


class GetPriceHistoryTool(BaseTool):
    name: str = "get_price_history"
    description: str = """Get historical price data for a stock.
    Input: stock ticker symbol and optional period (1mo, 3mo, 6mo, 1y, 2y)"""
    args_schema: Type[BaseModel] = AnalysisInput
    
    def _run(self, symbol: str, period: str = "1y") -> str:
        prices = _data_manager.get_price_history(symbol, period)
        
        if not prices:
            return f"No price history available for {symbol}"
        
        # Summary statistics
        closes = [p.close for p in prices]
        current = closes[-1]
        high_52w = max(closes[-252:]) if len(closes) >= 252 else max(closes)
        low_52w = min(closes[-252:]) if len(closes) >= 252 else min(closes)
        
        return_1m = (closes[-1] / closes[-21] - 1) * 100 if len(closes) > 21 else 0
        return_3m = (closes[-1] / closes[-63] - 1) * 100 if len(closes) > 63 else 0
        return_ytd = (closes[-1] / closes[0] - 1) * 100
        
        return f"""Price History for {symbol}:

Current Price: ${current:.2f}
52-Week High: ${high_52w:.2f}
52-Week Low: ${low_52w:.2f}

Returns:
- 1 Month: {return_1m:.1f}%
- 3 Month: {return_3m:.1f}%
- YTD: {return_ytd:.1f}%

Recent Prices (last 5 days):
""" + "\n".join([f"  {p.date}: ${p.close:.2f} (Vol: {p.volume:,})" for p in prices[-5:]])


class GetFinancialsTool(BaseTool):
    name: str = "get_financials"
    description: str = """Get financial statements (income, balance sheet, cash flow) for a company.
    Input: stock ticker symbol"""
    args_schema: Type[BaseModel] = StockInput
    
    def _run(self, symbol: str) -> str:
        financials = _data_manager.get_financials(symbol)
        
        if not financials:
            return f"No financial data available for {symbol}"
        
        result = f"Financial Statements for {symbol}:\n\n"
        
        # Group by type
        income = [f for f in financials if f.type == "income"]
        balance = [f for f in financials if f.type == "balance"]
        cashflow = [f for f in financials if f.type == "cashflow"]
        
        if income:
            result += "INCOME STATEMENT:\n"
            for stmt in income[:3]:
                result += f"\n{stmt.period}:\n"
                if stmt.revenue:
                    result += f"  Revenue: ${stmt.revenue/1e9:.2f}B\n"
                if stmt.gross_profit:
                    result += f"  Gross Profit: ${stmt.gross_profit/1e9:.2f}B\n"
                if stmt.operating_income:
                    result += f"  Operating Income: ${stmt.operating_income/1e9:.2f}B\n"
                if stmt.net_income:
                    result += f"  Net Income: ${stmt.net_income/1e9:.2f}B\n"
        
        if balance:
            result += "\nBALANCE SHEET:\n"
            stmt = balance[0]
            result += f"\n{stmt.period}:\n"
            if stmt.total_assets:
                result += f"  Total Assets: ${stmt.total_assets/1e9:.2f}B\n"
            if stmt.total_liabilities:
                result += f"  Total Liabilities: ${stmt.total_liabilities/1e9:.2f}B\n"
            if stmt.total_equity:
                result += f"  Total Equity: ${stmt.total_equity/1e9:.2f}B\n"
            if stmt.cash:
                result += f"  Cash: ${stmt.cash/1e9:.2f}B\n"
        
        if cashflow:
            result += "\nCASH FLOW:\n"
            stmt = cashflow[0]
            result += f"\n{stmt.period}:\n"
            if stmt.operating_cashflow:
                result += f"  Operating Cash Flow: ${stmt.operating_cashflow/1e9:.2f}B\n"
            if stmt.free_cashflow:
                result += f"  Free Cash Flow: ${stmt.free_cashflow/1e9:.2f}B\n"
        
        return result


class GetNewsTool(BaseTool):
    name: str = "get_news"
    description: str = """Get recent news articles about a company.
    Input: stock ticker symbol"""
    args_schema: Type[BaseModel] = StockInput
    
    def _run(self, symbol: str) -> str:
        news = _data_manager.get_news(symbol)
        
        if not news:
            return f"No recent news found for {symbol}"
        
        result = f"Recent News for {symbol}:\n\n"
        
        for i, article in enumerate(news[:10], 1):
            result += f"{i}. {article.get('title', 'No title')}\n"
            result += f"   Source: {article.get('publisher', 'Unknown')}\n"
            result += f"   Date: {article.get('published', 'Unknown')}\n\n"
        
        return result


# =============================================================================
# Analysis Tools
# =============================================================================

class TechnicalAnalysisTool(BaseTool):
    name: str = "technical_analysis"
    description: str = """Perform technical analysis on a stock including moving averages, 
    RSI, MACD, Bollinger Bands, and trend identification.
    Input: stock ticker symbol"""
    args_schema: Type[BaseModel] = StockInput
    
    def _run(self, symbol: str) -> str:
        prices = _data_manager.get_price_history(symbol, "1y")
        indicators = _technical_analyzer.analyze(symbol, prices)
        
        current_price = prices[-1].close if prices else 0
        
        return f"""Technical Analysis for {symbol}:

Current Price: ${current_price:.2f}
Overall Trend: {indicators.trend.upper()}

Moving Averages:
- SMA 20: ${indicators.sma_20:.2f if indicators.sma_20 else 'N/A'}
- SMA 50: ${indicators.sma_50:.2f if indicators.sma_50 else 'N/A'}
- SMA 200: ${indicators.sma_200:.2f if indicators.sma_200 else 'N/A'}

Momentum Indicators:
- RSI (14): {indicators.rsi_14:.1f if indicators.rsi_14 else 'N/A'}
- MACD: {indicators.macd:.2f if indicators.macd else 'N/A'}
- MACD Signal: {indicators.macd_signal:.2f if indicators.macd_signal else 'N/A'}

Volatility:
- Bollinger Upper: ${indicators.bollinger_upper:.2f if indicators.bollinger_upper else 'N/A'}
- Bollinger Lower: ${indicators.bollinger_lower:.2f if indicators.bollinger_lower else 'N/A'}
- ATR (14): ${indicators.atr_14:.2f if indicators.atr_14 else 'N/A'}

Support/Resistance:
- Support: ${indicators.support_level:.2f if indicators.support_level else 'N/A'}
- Resistance: ${indicators.resistance_level:.2f if indicators.resistance_level else 'N/A'}

Signals:
""" + "\n".join([f"- {s}" for s in indicators.signals]) if indicators.signals else "- No significant signals"


class FundamentalAnalysisTool(BaseTool):
    name: str = "fundamental_analysis"
    description: str = """Perform fundamental analysis including valuation, profitability, 
    growth, and financial health assessment.
    Input: stock ticker symbol"""
    args_schema: Type[BaseModel] = StockInput
    
    def _run(self, symbol: str) -> str:
        stock = _data_manager.get_stock(symbol)
        financials = _data_manager.get_financials(symbol)
        analysis = _fundamental_analyzer.analyze(stock, financials)
        
        val = analysis.get("valuation", {})
        prof = analysis.get("profitability", {})
        growth = analysis.get("growth", {})
        health = analysis.get("financial_health", {})
        
        return f"""Fundamental Analysis for {symbol}:

Quality Score: {analysis.get('quality_score', 0):.0f}/100

VALUATION ({val.get('assessment', 'N/A').upper()}):
- P/E Ratio: {val.get('pe_ratio', 'N/A')}
- Forward P/E: {val.get('forward_pe', 'N/A')}
- PEG Ratio: {val.get('peg_ratio', 'N/A')}
- Price/Book: {val.get('price_to_book', 'N/A')}

PROFITABILITY ({prof.get('assessment', 'N/A').upper()}):
- Gross Margin: {prof.get('gross_margin', 'N/A')}
- Operating Margin: {prof.get('operating_margin', 'N/A')}
- Profit Margin: {prof.get('profit_margin', 'N/A')}
- ROE: {prof.get('roe', 'N/A')}

GROWTH ({growth.get('assessment', 'N/A').upper()}):
- Revenue Growth: {growth.get('revenue_growth', 'N/A')}
- Earnings Growth: {growth.get('earnings_growth', 'N/A')}

FINANCIAL HEALTH ({health.get('assessment', 'N/A').upper()}):
- Debt/Equity: {health.get('debt_to_equity', 'N/A')}
- Current Ratio: {health.get('current_ratio', 'N/A')}
"""


class DCFValuationTool(BaseTool):
    name: str = "dcf_valuation"
    description: str = """Perform DCF (Discounted Cash Flow) valuation to estimate fair value.
    Input: stock ticker symbol"""
    args_schema: Type[BaseModel] = StockInput
    
    def _run(self, symbol: str) -> str:
        stock = _data_manager.get_stock(symbol)
        financials = _data_manager.get_financials(symbol)
        
        valuation = _fundamental_analyzer.dcf_valuation(stock, financials)
        
        upside = valuation.upside_potential * 100
        
        return f"""DCF Valuation for {symbol}:

Current Price: ${stock.current_price:.2f}
Fair Value (Base Case): ${valuation.fair_value:.2f}
Upside/Downside: {upside:+.1f}%

Scenario Analysis:
- Bull Case: ${valuation.bull_case:.2f} ({((valuation.bull_case/stock.current_price)-1)*100:+.1f}%)
- Base Case: ${valuation.base_case:.2f} ({upside:+.1f}%)
- Bear Case: ${valuation.bear_case:.2f} ({((valuation.bear_case/stock.current_price)-1)*100:+.1f}%)

Assumptions:
- Growth Rate: {valuation.assumptions.get('growth_rate', 0)*100:.0f}%
- Terminal Growth: {valuation.assumptions.get('terminal_growth', 0)*100:.0f}%
- Discount Rate: {valuation.assumptions.get('discount_rate', 0)*100:.0f}%
- Projection Years: {valuation.assumptions.get('projection_years', 5)}
"""


class SentimentAnalysisTool(BaseTool):
    name: str = "sentiment_analysis"
    description: str = """Analyze market sentiment from news and social media for a stock.
    Input: stock ticker symbol"""
    args_schema: Type[BaseModel] = StockInput
    
    def _run(self, symbol: str) -> str:
        news = _data_manager.get_news(symbol)
        sentiment = _sentiment_analyzer.analyze(symbol, news)
        
        return f"""Sentiment Analysis for {symbol}:

Overall Sentiment: {sentiment.overall_label.upper()}
Sentiment Score: {sentiment.overall_score:.2f} (-1 to +1 scale)

News Analysis:
- Articles Analyzed: {sentiment.news_articles_analyzed}
- News Sentiment Score: {sentiment.news_score:.2f}

Bullish Topics:
""" + "\n".join([f"- {t}" for t in sentiment.bullish_topics[:5]]) + """

Bearish Topics:
""" + "\n".join([f"- {t}" for t in sentiment.bearish_topics[:5]])


class RiskAssessmentTool(BaseTool):
    name: str = "risk_assessment"
    description: str = """Assess investment risks for a stock.
    Input: stock ticker symbol"""
    args_schema: Type[BaseModel] = StockInput
    
    def _run(self, symbol: str) -> str:
        stock = _data_manager.get_stock(symbol)
        prices = _data_manager.get_price_history(symbol, "1y")
        technical = _technical_analyzer.analyze(symbol, prices)
        financials = _data_manager.get_financials(symbol)
        fundamental = _fundamental_analyzer.analyze(stock, financials)
        
        risk = _risk_analyzer.assess_risk(stock, technical, fundamental)
        
        result = f"""Risk Assessment for {symbol}:

Risk Score: {risk['risk_score']}/100
Risk Level: {risk['risk_level'].value.upper()}

Risk Factors:
"""
        for rf in risk['risk_factors']:
            result += f"\n{rf['category'].upper()}: {rf['title']}\n"
            result += f"  Severity: {rf['severity']}\n"
            result += f"  {rf['description']}\n"
        
        if not risk['risk_factors']:
            result += "\nNo significant risk factors identified."
        
        return result


class CompareStocksTool(BaseTool):
    name: str = "compare_stocks"
    description: str = """Compare multiple stocks side by side on key metrics.
    Input: comma-separated stock symbols (e.g., AAPL,MSFT,GOOGL)"""
    args_schema: Type[BaseModel] = CompareInput
    
    def _run(self, symbols: str) -> str:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        
        stocks = [_data_manager.get_stock(s) for s in symbol_list]
        
        result = "Stock Comparison:\n\n"
        result += f"{'Metric':<20}"
        for stock in stocks:
            result += f"{stock.symbol:>15}"
        result += "\n" + "-" * (20 + 15 * len(stocks)) + "\n"
        
        metrics = [
            ("Price", lambda s: f"${s.current_price:.2f}"),
            ("Market Cap", lambda s: f"${s.market_cap/1e9:.1f}B"),
            ("P/E Ratio", lambda s: f"{s.pe_ratio:.1f}" if s.pe_ratio else "N/A"),
            ("Forward P/E", lambda s: f"{s.forward_pe:.1f}" if s.forward_pe else "N/A"),
            ("Revenue Growth", lambda s: f"{s.revenue_growth*100:.1f}%" if s.revenue_growth else "N/A"),
            ("Gross Margin", lambda s: f"{s.gross_margin*100:.1f}%" if s.gross_margin else "N/A"),
            ("ROE", lambda s: f"{s.return_on_equity*100:.1f}%" if s.return_on_equity else "N/A"),
            ("Debt/Equity", lambda s: f"{s.debt_to_equity:.2f}" if s.debt_to_equity else "N/A"),
        ]
        
        for metric_name, metric_func in metrics:
            result += f"{metric_name:<20}"
            for stock in stocks:
                result += f"{metric_func(stock):>15}"
            result += "\n"
        
        return result


# =============================================================================
# Tool Factory
# =============================================================================

def get_all_tools() -> list:
    """Get all available tools."""
    return [
        GetStockInfoTool(),
        GetPriceHistoryTool(),
        GetFinancialsTool(),
        GetNewsTool(),
        TechnicalAnalysisTool(),
        FundamentalAnalysisTool(),
        DCFValuationTool(),
        SentimentAnalysisTool(),
        RiskAssessmentTool(),
        CompareStocksTool(),
    ]


def get_tools_for_role(role: str) -> list:
    """Get tools appropriate for a specific role."""
    role_tools = {
        "macro_analyst": [GetStockInfoTool(), GetNewsTool(), SentimentAnalysisTool()],
        "equity_analyst": [
            GetStockInfoTool(), GetFinancialsTool(), GetPriceHistoryTool(),
            FundamentalAnalysisTool(), CompareStocksTool()
        ],
        "sector_analyst": [
            GetStockInfoTool(), GetFinancialsTool(), GetNewsTool(),
            FundamentalAnalysisTool(), CompareStocksTool()
        ],
        "quant_researcher": [
            GetPriceHistoryTool(), TechnicalAnalysisTool(), 
            DCFValuationTool(), RiskAssessmentTool()
        ],
        "portfolio_manager": [
            GetStockInfoTool(), RiskAssessmentTool(), CompareStocksTool()
        ],
        "research_director": get_all_tools(),
    }
    
    return role_tools.get(role, get_all_tools())
