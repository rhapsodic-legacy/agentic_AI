"""
Investment Research Firm - Analysis Tools

Tools for:
- Technical analysis (indicators, patterns)
- Fundamental analysis (DCF, comparables)
- Sentiment analysis
- Risk assessment
"""

from typing import Optional, Any
from dataclasses import dataclass
import math

from ..models import (
    Stock, StockPrice, FinancialStatement, TechnicalIndicators,
    SentimentAnalysis, ValuationModel, MacroAnalysis,
    Sector, RiskLevel
)


class TechnicalAnalyzer:
    """
    Technical analysis tools.
    """
    
    def analyze(self, symbol: str, prices: list[StockPrice]) -> TechnicalIndicators:
        """Perform comprehensive technical analysis."""
        if not prices or len(prices) < 20:
            return TechnicalIndicators(symbol=symbol, date="N/A")
        
        closes = [p.close for p in prices]
        highs = [p.high for p in prices]
        lows = [p.low for p in prices]
        volumes = [p.volume for p in prices]
        
        current_price = closes[-1]
        
        indicators = TechnicalIndicators(
            symbol=symbol,
            date=prices[-1].date,
        )
        
        # Moving Averages
        indicators.sma_20 = self._sma(closes, 20)
        indicators.sma_50 = self._sma(closes, 50)
        indicators.sma_200 = self._sma(closes, 200)
        indicators.ema_12 = self._ema(closes, 12)
        indicators.ema_26 = self._ema(closes, 26)
        
        # RSI
        indicators.rsi_14 = self._rsi(closes, 14)
        
        # MACD
        if indicators.ema_12 and indicators.ema_26:
            indicators.macd = indicators.ema_12 - indicators.ema_26
            signal_line = self._ema(closes[-26:], 9) if len(closes) >= 26 else None
            if signal_line:
                indicators.macd_signal = signal_line
                indicators.macd_histogram = indicators.macd - signal_line
        
        # Bollinger Bands
        if indicators.sma_20:
            std = self._std(closes[-20:])
            indicators.bollinger_upper = indicators.sma_20 + (2 * std)
            indicators.bollinger_lower = indicators.sma_20 - (2 * std)
        
        # ATR
        indicators.atr_14 = self._atr(highs, lows, closes, 14)
        
        # Support/Resistance
        indicators.support_level = min(lows[-20:])
        indicators.resistance_level = max(highs[-20:])
        
        # Trend determination
        indicators.trend = self._determine_trend(closes, indicators)
        
        # Generate signals
        indicators.signals = self._generate_signals(current_price, indicators)
        
        return indicators
    
    def _sma(self, data: list[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average."""
        if len(data) < period:
            return None
        return sum(data[-period:]) / period
    
    def _ema(self, data: list[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period  # Start with SMA
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema
    
    def _rsi(self, data: list[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index."""
        if len(data) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(data)):
            change = data[i] - data[i-1]
            gains.append(max(change, 0))
            losses.append(abs(min(change, 0)))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _atr(self, highs: list, lows: list, closes: list, period: int) -> Optional[float]:
        """Calculate Average True Range."""
        if len(closes) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            true_ranges.append(tr)
        
        return sum(true_ranges[-period:]) / period
    
    def _std(self, data: list[float]) -> float:
        """Calculate standard deviation."""
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return math.sqrt(variance)
    
    def _determine_trend(self, closes: list[float], indicators: TechnicalIndicators) -> str:
        """Determine overall trend."""
        current = closes[-1]
        
        bullish_signals = 0
        bearish_signals = 0
        
        # Price vs MAs
        if indicators.sma_50 and current > indicators.sma_50:
            bullish_signals += 1
        elif indicators.sma_50:
            bearish_signals += 1
        
        if indicators.sma_200 and current > indicators.sma_200:
            bullish_signals += 1
        elif indicators.sma_200:
            bearish_signals += 1
        
        # Golden/Death cross
        if indicators.sma_50 and indicators.sma_200:
            if indicators.sma_50 > indicators.sma_200:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # RSI
        if indicators.rsi_14:
            if indicators.rsi_14 > 50:
                bullish_signals += 1
            else:
                bearish_signals += 1
        
        # MACD
        if indicators.macd and indicators.macd > 0:
            bullish_signals += 1
        elif indicators.macd:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals + 1:
            return "bullish"
        elif bearish_signals > bullish_signals + 1:
            return "bearish"
        return "neutral"
    
    def _generate_signals(self, current_price: float, indicators: TechnicalIndicators) -> list[str]:
        """Generate trading signals."""
        signals = []
        
        # RSI signals
        if indicators.rsi_14:
            if indicators.rsi_14 < 30:
                signals.append("RSI Oversold - Potential bounce")
            elif indicators.rsi_14 > 70:
                signals.append("RSI Overbought - Potential pullback")
        
        # Moving average signals
        if indicators.sma_50 and indicators.sma_200:
            if indicators.sma_50 > indicators.sma_200:
                signals.append("Golden Cross - Bullish long-term")
            else:
                signals.append("Death Cross - Bearish long-term")
        
        # Bollinger Band signals
        if indicators.bollinger_lower and current_price < indicators.bollinger_lower:
            signals.append("Below lower Bollinger Band - Oversold")
        elif indicators.bollinger_upper and current_price > indicators.bollinger_upper:
            signals.append("Above upper Bollinger Band - Overbought")
        
        # Support/Resistance
        if indicators.support_level:
            distance_to_support = (current_price - indicators.support_level) / current_price
            if distance_to_support < 0.03:
                signals.append("Near support level")
        
        if indicators.resistance_level:
            distance_to_resistance = (indicators.resistance_level - current_price) / current_price
            if distance_to_resistance < 0.03:
                signals.append("Near resistance level")
        
        return signals


class FundamentalAnalyzer:
    """
    Fundamental analysis tools.
    """
    
    def analyze(self, stock: Stock, financials: list[FinancialStatement]) -> dict:
        """Perform comprehensive fundamental analysis."""
        analysis = {
            "symbol": stock.symbol,
            "valuation": self._analyze_valuation(stock),
            "profitability": self._analyze_profitability(stock),
            "growth": self._analyze_growth(stock, financials),
            "financial_health": self._analyze_health(stock),
            "quality_score": 0,
        }
        
        # Calculate quality score
        analysis["quality_score"] = self._calculate_quality_score(analysis)
        
        return analysis
    
    def _analyze_valuation(self, stock: Stock) -> dict:
        """Analyze valuation metrics."""
        valuation = {
            "pe_ratio": stock.pe_ratio,
            "forward_pe": stock.forward_pe,
            "peg_ratio": stock.peg_ratio,
            "price_to_book": stock.price_to_book,
            "price_to_sales": stock.price_to_sales,
            "assessment": "fair",
        }
        
        # Assess valuation
        if stock.pe_ratio:
            if stock.pe_ratio < 15:
                valuation["assessment"] = "undervalued"
            elif stock.pe_ratio > 30:
                valuation["assessment"] = "expensive"
        
        if stock.peg_ratio:
            if stock.peg_ratio < 1:
                valuation["peg_assessment"] = "attractive"
            elif stock.peg_ratio > 2:
                valuation["peg_assessment"] = "expensive"
        
        return valuation
    
    def _analyze_profitability(self, stock: Stock) -> dict:
        """Analyze profitability metrics."""
        return {
            "gross_margin": stock.gross_margin,
            "operating_margin": stock.operating_margin,
            "profit_margin": stock.profit_margin,
            "roe": stock.return_on_equity,
            "roa": stock.return_on_assets,
            "assessment": self._assess_profitability(stock),
        }
    
    def _assess_profitability(self, stock: Stock) -> str:
        """Assess profitability quality."""
        score = 0
        
        if stock.gross_margin and stock.gross_margin > 0.40:
            score += 1
        if stock.operating_margin and stock.operating_margin > 0.20:
            score += 1
        if stock.profit_margin and stock.profit_margin > 0.10:
            score += 1
        if stock.return_on_equity and stock.return_on_equity > 0.15:
            score += 1
        
        if score >= 3:
            return "excellent"
        elif score >= 2:
            return "good"
        elif score >= 1:
            return "average"
        return "poor"
    
    def _analyze_growth(self, stock: Stock, financials: list[FinancialStatement]) -> dict:
        """Analyze growth metrics."""
        growth = {
            "revenue_growth": stock.revenue_growth,
            "earnings_growth": stock.earnings_growth,
            "assessment": "moderate",
        }
        
        # Calculate historical growth from financials
        income_statements = [f for f in financials if f.type == "income"]
        if len(income_statements) >= 2:
            revenues = [f.revenue for f in income_statements if f.revenue]
            if len(revenues) >= 2:
                growth["historical_revenue_cagr"] = (revenues[0] / revenues[-1]) ** (1/len(revenues)) - 1
        
        # Assess growth
        if stock.revenue_growth:
            if stock.revenue_growth > 0.25:
                growth["assessment"] = "high growth"
            elif stock.revenue_growth > 0.10:
                growth["assessment"] = "moderate growth"
            elif stock.revenue_growth > 0:
                growth["assessment"] = "low growth"
            else:
                growth["assessment"] = "declining"
        
        return growth
    
    def _analyze_health(self, stock: Stock) -> dict:
        """Analyze financial health."""
        health = {
            "debt_to_equity": stock.debt_to_equity,
            "current_ratio": stock.current_ratio,
            "quick_ratio": stock.quick_ratio,
            "assessment": "healthy",
        }
        
        risk_flags = 0
        
        if stock.debt_to_equity and stock.debt_to_equity > 2:
            risk_flags += 1
            health["debt_concern"] = True
        
        if stock.current_ratio and stock.current_ratio < 1:
            risk_flags += 1
            health["liquidity_concern"] = True
        
        if risk_flags >= 2:
            health["assessment"] = "concerning"
        elif risk_flags == 1:
            health["assessment"] = "moderate risk"
        
        return health
    
    def _calculate_quality_score(self, analysis: dict) -> float:
        """Calculate overall quality score (0-100)."""
        score = 50  # Base score
        
        # Valuation contribution
        val = analysis.get("valuation", {})
        if val.get("assessment") == "undervalued":
            score += 15
        elif val.get("assessment") == "expensive":
            score -= 10
        
        # Profitability contribution
        prof = analysis.get("profitability", {})
        if prof.get("assessment") == "excellent":
            score += 20
        elif prof.get("assessment") == "good":
            score += 10
        elif prof.get("assessment") == "poor":
            score -= 15
        
        # Growth contribution
        growth = analysis.get("growth", {})
        if growth.get("assessment") == "high growth":
            score += 15
        elif growth.get("assessment") == "declining":
            score -= 15
        
        # Health contribution
        health = analysis.get("financial_health", {})
        if health.get("assessment") == "concerning":
            score -= 20
        elif health.get("assessment") == "moderate risk":
            score -= 10
        
        return max(0, min(100, score))
    
    def dcf_valuation(
        self,
        stock: Stock,
        financials: list[FinancialStatement],
        growth_rate: float = 0.10,
        terminal_growth: float = 0.03,
        discount_rate: float = 0.10,
        years: int = 5
    ) -> ValuationModel:
        """Perform DCF valuation."""
        # Get latest free cash flow
        cashflow_statements = [f for f in financials if f.type == "cashflow"]
        
        if cashflow_statements and cashflow_statements[0].free_cashflow:
            fcf = cashflow_statements[0].free_cashflow
        else:
            # Estimate from profit margin and revenue
            fcf = stock.market_cap * 0.05  # Rough estimate
        
        # Project cash flows
        projected_fcf = []
        current_fcf = fcf
        
        for year in range(1, years + 1):
            current_fcf *= (1 + growth_rate)
            discounted = current_fcf / ((1 + discount_rate) ** year)
            projected_fcf.append(discounted)
        
        # Terminal value
        terminal_fcf = current_fcf * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        discounted_terminal = terminal_value / ((1 + discount_rate) ** years)
        
        # Enterprise value
        enterprise_value = sum(projected_fcf) + discounted_terminal
        
        # Equity value (simplified - should subtract debt, add cash)
        equity_value = enterprise_value
        
        # Per share value (estimate shares from market cap)
        shares = stock.market_cap / stock.current_price if stock.current_price else 1
        fair_value = equity_value / shares
        
        # Scenarios
        bull_case = fair_value * 1.25
        bear_case = fair_value * 0.75
        
        return ValuationModel(
            symbol=stock.symbol,
            model_type="dcf",
            assumptions={
                "growth_rate": growth_rate,
                "terminal_growth": terminal_growth,
                "discount_rate": discount_rate,
                "projection_years": years,
                "base_fcf": fcf,
            },
            fair_value=fair_value,
            upside_potential=(fair_value - stock.current_price) / stock.current_price,
            bull_case=bull_case,
            base_case=fair_value,
            bear_case=bear_case,
        )


class SentimentAnalyzer:
    """
    Sentiment analysis from news and social media.
    """
    
    def analyze(self, symbol: str, news: list[dict]) -> SentimentAnalysis:
        """Analyze sentiment from news."""
        sentiment = SentimentAnalysis(
            symbol=symbol,
            date=datetime.now().isoformat() if 'datetime' in dir() else "today",
        )
        
        if not news:
            return sentiment
        
        # Simple keyword-based sentiment (in production, use NLP models)
        positive_keywords = [
            "beat", "exceeds", "strong", "growth", "upgrade", "buy",
            "outperform", "bullish", "record", "surge", "rally", "gain"
        ]
        negative_keywords = [
            "miss", "decline", "weak", "downgrade", "sell", "underperform",
            "bearish", "loss", "drop", "fall", "concern", "warning"
        ]
        
        positive_count = 0
        negative_count = 0
        
        for article in news:
            title = article.get("title", "").lower()
            
            for kw in positive_keywords:
                if kw in title:
                    positive_count += 1
                    sentiment.bullish_topics.append(article.get("title", "")[:50])
                    break
            
            for kw in negative_keywords:
                if kw in title:
                    negative_count += 1
                    sentiment.bearish_topics.append(article.get("title", "")[:50])
                    break
        
        sentiment.news_articles_analyzed = len(news)
        sentiment.notable_news = news[:5]
        
        # Calculate score
        total = positive_count + negative_count
        if total > 0:
            sentiment.news_score = (positive_count - negative_count) / total
        
        sentiment.overall_score = sentiment.news_score
        
        if sentiment.overall_score > 0.3:
            sentiment.overall_label = "positive"
        elif sentiment.overall_score < -0.3:
            sentiment.overall_label = "negative"
        else:
            sentiment.overall_label = "neutral"
        
        return sentiment


class RiskAnalyzer:
    """
    Risk assessment tools.
    """
    
    def assess_risk(
        self,
        stock: Stock,
        technical: TechnicalIndicators,
        fundamental: dict
    ) -> dict:
        """Comprehensive risk assessment."""
        risk_factors = []
        risk_score = 50  # Base score
        
        # Valuation risk
        if stock.pe_ratio and stock.pe_ratio > 40:
            risk_factors.append({
                "category": "valuation",
                "title": "High Valuation",
                "description": f"P/E ratio of {stock.pe_ratio:.1f}x is elevated",
                "severity": "medium",
            })
            risk_score += 10
        
        # Volatility risk
        if technical.atr_14 and stock.current_price:
            atr_percent = technical.atr_14 / stock.current_price * 100
            if atr_percent > 5:
                risk_factors.append({
                    "category": "volatility",
                    "title": "High Volatility",
                    "description": f"Daily volatility of {atr_percent:.1f}%",
                    "severity": "medium",
                })
                risk_score += 10
        
        # Debt risk
        if stock.debt_to_equity and stock.debt_to_equity > 1.5:
            risk_factors.append({
                "category": "financial",
                "title": "Elevated Debt",
                "description": f"Debt/Equity ratio of {stock.debt_to_equity:.2f}",
                "severity": "high" if stock.debt_to_equity > 2.5 else "medium",
            })
            risk_score += 15
        
        # Growth risk
        if stock.revenue_growth and stock.revenue_growth < 0:
            risk_factors.append({
                "category": "growth",
                "title": "Declining Revenue",
                "description": "Revenue is contracting year over year",
                "severity": "high",
            })
            risk_score += 20
        
        # Technical risk
        if technical.trend == "bearish":
            risk_factors.append({
                "category": "technical",
                "title": "Bearish Trend",
                "description": "Price is in a downtrend",
                "severity": "medium",
            })
            risk_score += 10
        
        # Determine risk level
        if risk_score >= 80:
            risk_level = RiskLevel.VERY_HIGH
        elif risk_score >= 65:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 50:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
        }


# Import datetime for SentimentAnalyzer
from datetime import datetime
