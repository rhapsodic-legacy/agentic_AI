#!/usr/bin/env python3
"""
Investment Research Firm - Command Line Interface

Usage:
    python main.py research NVDA
    python main.py screen AAPL,MSFT,GOOGL
    python main.py analyze NVDA --type technical
    python main.py serve
"""

import argparse
import sys
import json
from pathlib import Path

# Rich console
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_banner():
    """Print the CLI banner."""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold blue]💰 Investment Research Firm[/bold blue]\n"
            "[dim]AI-Powered Equity Research & Analysis[/dim]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*50)
        print("💰 Investment Research Firm")
        print("AI-Powered Equity Research & Analysis")
        print("="*50 + "\n")


def print_team():
    """Print the research team structure."""
    if RICH_AVAILABLE:
        table = Table(title="Research Team")
        table.add_column("Role", style="cyan")
        table.add_column("Responsibility")
        
        table.add_row("📊 Research Director", "Coordinates research, assigns tasks")
        table.add_row("🌍 Macro Analyst", "Economic analysis, sector context")
        table.add_row("📈 Equity Analyst", "Fundamental stock analysis")
        table.add_row("💻 Tech Analyst", "Technology sector specialist")
        table.add_row("🏦 Finance Analyst", "Financial sector specialist")
        table.add_row("🏥 Healthcare Analyst", "Healthcare sector specialist")
        table.add_row("🔢 Quant Researcher", "Technical & quantitative analysis")
        table.add_row("💼 Portfolio Manager", "Final recommendations")
        
        console.print(table)


def run_research(args):
    """Run full research on a stock."""
    from investment_firm import InvestmentResearchCrew, ResearchConfig
    from investment_firm.reports import ReportGenerator
    
    print_banner()
    
    symbol = args.symbol.upper()
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Initiating research on {symbol}...[/bold]\n")
        print_team()
    else:
        print(f"\nInitiating research on {symbol}...\n")
    
    config = ResearchConfig(
        llm_provider=args.provider,
        verbose=args.verbose,
    )
    
    crew = InvestmentResearchCrew(config)
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Researching {symbol}...", total=None)
            result = crew.research_stock(symbol)
    else:
        print(f"Researching {symbol}...")
        result = crew.research_stock(symbol)
    
    if result.success and result.memo:
        # Print memo
        if RICH_AVAILABLE:
            console.print(Markdown(result.memo.to_markdown()))
        else:
            print(result.memo.to_markdown())
        
        # Save report
        if args.output:
            generator = ReportGenerator(args.output)
            filepath = generator.save(result.memo, args.format)
            print(f"\n✓ Report saved to: {filepath}")
    else:
        print(f"✗ Research failed: {result.errors}")
        return 1
    
    return 0


def run_screen(args):
    """Screen multiple stocks."""
    from investment_firm import InvestmentResearchCrew
    
    print_banner()
    
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    
    print(f"\nScreening {len(symbols)} stocks: {', '.join(symbols)}\n")
    
    crew = InvestmentResearchCrew()
    results = crew.screen_stocks(symbols)
    
    if RICH_AVAILABLE:
        table = Table(title="Stock Screening Results")
        table.add_column("Symbol", style="cyan")
        table.add_column("Name")
        table.add_column("Price", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Upside", justify="right")
        table.add_column("Quality", justify="center")
        table.add_column("Trend", justify="center")
        
        for r in results:
            if "error" in r:
                table.add_row(r["symbol"], "Error", "-", "-", "-", "-", "-")
            else:
                upside = r.get("upside", "N/A")
                upside_style = "green" if upside.startswith("+") else "red" if upside.startswith("-") else "white"
                
                table.add_row(
                    r.get("symbol", ""),
                    r.get("name", "")[:25],
                    f"${r.get('price', 0):.2f}",
                    f"${r.get('target', 0):.2f}",
                    f"[{upside_style}]{upside}[/{upside_style}]",
                    f"{r.get('quality_score', 0):.0f}",
                    r.get("trend", "N/A"),
                )
        
        console.print(table)
    else:
        print(f"{'Symbol':<8} {'Name':<25} {'Price':>10} {'Target':>10} {'Upside':>10}")
        print("-" * 70)
        for r in results:
            if "error" not in r:
                print(f"{r.get('symbol', ''):<8} {r.get('name', '')[:25]:<25} "
                      f"${r.get('price', 0):>8.2f} ${r.get('target', 0):>8.2f} {r.get('upside', 'N/A'):>10}")


def run_analyze(args):
    """Run specific analysis on a stock."""
    from investment_firm import MarketDataManager, TechnicalAnalyzer, FundamentalAnalyzer
    
    print_banner()
    
    symbol = args.symbol.upper()
    data_manager = MarketDataManager()
    
    stock = data_manager.get_stock(symbol)
    
    print(f"\n{args.type.title()} Analysis for {symbol} ({stock.name})\n")
    
    if args.type == "technical":
        prices = data_manager.get_price_history(symbol)
        analyzer = TechnicalAnalyzer()
        indicators = analyzer.analyze(symbol, prices)
        
        if RICH_AVAILABLE:
            console.print(f"[bold]Trend:[/bold] {indicators.trend.upper()}")
            console.print(f"[bold]RSI (14):[/bold] {indicators.rsi_14:.1f}" if indicators.rsi_14 else "")
            console.print(f"[bold]SMA 50:[/bold] ${indicators.sma_50:.2f}" if indicators.sma_50 else "")
            console.print(f"[bold]SMA 200:[/bold] ${indicators.sma_200:.2f}" if indicators.sma_200 else "")
            console.print(f"\n[bold]Signals:[/bold]")
            for sig in indicators.signals:
                console.print(f"  • {sig}")
        else:
            print(f"Trend: {indicators.trend}")
            print(f"RSI: {indicators.rsi_14}")
            print(f"Signals: {indicators.signals}")
    
    elif args.type == "fundamental":
        financials = data_manager.get_financials(symbol)
        analyzer = FundamentalAnalyzer()
        analysis = analyzer.analyze(stock, financials)
        
        if RICH_AVAILABLE:
            console.print(f"[bold]Quality Score:[/bold] {analysis.get('quality_score', 0):.0f}/100")
            console.print(f"[bold]Valuation:[/bold] {analysis.get('valuation', {}).get('assessment', 'N/A')}")
            console.print(f"[bold]Profitability:[/bold] {analysis.get('profitability', {}).get('assessment', 'N/A')}")
            console.print(f"[bold]Growth:[/bold] {analysis.get('growth', {}).get('assessment', 'N/A')}")
            console.print(f"[bold]Financial Health:[/bold] {analysis.get('financial_health', {}).get('assessment', 'N/A')}")
        else:
            print(json.dumps(analysis, indent=2, default=str))
    
    elif args.type == "valuation":
        financials = data_manager.get_financials(symbol)
        analyzer = FundamentalAnalyzer()
        valuation = analyzer.dcf_valuation(stock, financials)
        
        if RICH_AVAILABLE:
            console.print(f"[bold]Current Price:[/bold] ${stock.current_price:.2f}")
            console.print(f"[bold]Fair Value:[/bold] ${valuation.fair_value:.2f}")
            console.print(f"[bold]Upside:[/bold] {valuation.upside_potential*100:+.1f}%")
            console.print(f"\n[bold]Scenarios:[/bold]")
            console.print(f"  Bull: ${valuation.bull_case:.2f}")
            console.print(f"  Base: ${valuation.base_case:.2f}")
            console.print(f"  Bear: ${valuation.bear_case:.2f}")
        else:
            print(f"Fair Value: ${valuation.fair_value:.2f}")
            print(f"Upside: {valuation.upside_potential*100:.1f}%")


def run_interactive(args):
    """Run interactive mode."""
    from investment_firm import InvestmentResearchCrew, MarketDataManager
    
    print_banner()
    print_team()
    
    print("\nCommands:")
    print("  research SYMBOL   - Full research report")
    print("  screen SYM1,SYM2  - Screen multiple stocks")
    print("  info SYMBOL       - Quick stock info")
    print("  technical SYMBOL  - Technical analysis")
    print("  quit              - Exit")
    print()
    
    data_manager = MarketDataManager()
    crew = InvestmentResearchCrew()
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold green]>>> [/bold green]")
            else:
                user_input = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break
        
        parts = user_input.strip().split()
        if not parts:
            continue
        
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break
        
        elif cmd == "research" and arg:
            result = crew.research_stock(arg.upper())
            if result.success and result.memo:
                if RICH_AVAILABLE:
                    console.print(Markdown(result.memo.to_markdown()))
                else:
                    print(result.memo.to_markdown())
        
        elif cmd == "screen" and arg:
            symbols = [s.strip().upper() for s in arg.split(",")]
            results = crew.screen_stocks(symbols)
            for r in results:
                print(f"{r.get('symbol')}: ${r.get('price', 0):.2f} → ${r.get('target', 0):.2f} ({r.get('upside', 'N/A')})")
        
        elif cmd == "info" and arg:
            stock = data_manager.get_stock(arg.upper())
            print(f"\n{stock.symbol} - {stock.name}")
            print(f"Price: ${stock.current_price:.2f}")
            print(f"P/E: {stock.pe_ratio:.1f}" if stock.pe_ratio else "")
            print(f"Growth: {stock.revenue_growth*100:.0f}%" if stock.revenue_growth else "")
        
        elif cmd == "technical" and arg:
            from investment_firm import TechnicalAnalyzer
            prices = data_manager.get_price_history(arg.upper())
            analyzer = TechnicalAnalyzer()
            indicators = analyzer.analyze(arg.upper(), prices)
            print(f"Trend: {indicators.trend}, RSI: {indicators.rsi_14:.1f}" if indicators.rsi_14 else "")
        
        else:
            print(f"Unknown command: {cmd}")


def run_serve(args):
    """Run the web server."""
    import uvicorn
    
    print_banner()
    
    print(f"🌐 Starting server at http://localhost:{args.port}")
    print(f"📖 API docs at http://localhost:{args.port}/docs")
    print("\nPress Ctrl+C to stop\n")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Investment Research Firm - AI-Powered Equity Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full research on a stock
    python main.py research NVDA
    
    # Screen multiple stocks
    python main.py screen AAPL,MSFT,GOOGL,AMZN
    
    # Quick analysis
    python main.py analyze NVDA --type technical
    
    # Interactive mode
    python main.py interactive
    
    # Start web server
    python main.py serve
        """
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["gemini", "anthropic", "openai"],
        default="gemini",
        help="LLM provider"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Full research report")
    research_parser.add_argument("symbol", help="Stock symbol")
    research_parser.add_argument("--output", "-o", default="./output", help="Output directory")
    research_parser.add_argument("--format", "-f", choices=["html", "markdown"], default="html")
    
    # Screen command
    screen_parser = subparsers.add_parser("screen", help="Screen multiple stocks")
    screen_parser.add_argument("symbols", help="Comma-separated symbols")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Quick analysis")
    analyze_parser.add_argument("symbol", help="Stock symbol")
    analyze_parser.add_argument("--type", "-t", choices=["technical", "fundamental", "valuation"], default="technical")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "research":
        sys.exit(run_research(args))
    elif args.command == "screen":
        run_screen(args)
    elif args.command == "analyze":
        run_analyze(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
