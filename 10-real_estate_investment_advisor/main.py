#!/usr/bin/env python3
"""
Real Estate Investment Advisor - Command Line Interface

Usage:
    python main.py search Austin --max-price 500000
    python main.py analyze PROPERTY_ID
    python main.py market Austin
    python main.py interactive
    python main.py serve
"""

import argparse
import sys
from pathlib import Path

# Rich console
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_banner():
    """Print the CLI banner."""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold green]🏠 Real Estate Investment Advisor[/bold green]\n"
            "[dim]AI-Powered Property Analysis[/dim]",
            border_style="green"
        ))
    else:
        print("\n" + "="*50)
        print("🏠 Real Estate Investment Advisor")
        print("AI-Powered Property Analysis")
        print("="*50 + "\n")


def print_architecture():
    """Print the supervisor architecture."""
    if RICH_AVAILABLE:
        console.print("""
[dim]Supervisor Pattern Architecture:[/dim]
                ┌─────────────────────┐
                │    [bold]SUPERVISOR[/bold]       │
                │  (Query Router)     │
                └──────────┬──────────┘
                           │
    ┌──────────────────────┼──────────────────────┐
    │                      │                      │
    ▼                      ▼                      ▼
[cyan]MARKET[/cyan]              [yellow]PROPERTY[/yellow]           [green]FINANCIAL[/green]
[cyan]ANALYST[/cyan]             [yellow]EVALUATOR[/yellow]          [green]MODELER[/green]
    │                      │                      │
    └──────────────────────┼──────────────────────┘
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
           [red]RISK[/red]              [magenta]LEGAL[/magenta]
           [red]ASSESSOR[/red]          [magenta]CHECKER[/magenta]
""")


def run_search(args):
    """Search for properties."""
    from real_estate import RealEstateAdvisor
    
    print_banner()
    
    advisor = RealEstateAdvisor()
    
    if RICH_AVAILABLE:
        console.print(f"[bold]Searching for properties in {args.city}...[/bold]\n")
    
    properties = advisor.search_properties(
        city=args.city,
        max_price=args.max_price,
        min_beds=args.min_beds,
    )
    
    if not properties:
        print(f"No properties found in {args.city}")
        return
    
    if RICH_AVAILABLE:
        table = Table(title=f"Properties in {args.city}")
        table.add_column("ID", style="cyan")
        table.add_column("Address", style="white")
        table.add_column("Price", justify="right", style="green")
        table.add_column("Beds/Bath", justify="center")
        table.add_column("Sqft", justify="right")
        table.add_column("Type", style="dim")
        
        for prop in properties[:10]:
            table.add_row(
                prop.property_id[:12],
                str(prop.address)[:35],
                f"${prop.list_price:,.0f}",
                f"{prop.features.bedrooms}/{prop.features.bathrooms}",
                f"{prop.features.sqft:,}",
                prop.property_type.value[:15],
            )
        
        console.print(table)
    else:
        for prop in properties[:10]:
            print(f"{prop.property_id}: {prop.address} - ${prop.list_price:,.0f}")


def run_analyze(args):
    """Analyze a property."""
    from real_estate import RealEstateAdvisor
    
    print_banner()
    print_architecture()
    
    advisor = RealEstateAdvisor()
    
    # Get property
    prop = advisor.get_property(args.property_id)
    
    if not prop:
        # Try to find by partial ID
        all_props = advisor.search_properties()
        for p in all_props:
            if args.property_id in p.property_id:
                prop = p
                break
    
    if not prop:
        print(f"Property not found: {args.property_id}")
        return
    
    # Run analysis
    analysis = advisor.analyze_property(
        prop,
        down_payment_pct=args.down_payment / 100 if args.down_payment else None,
        interest_rate=args.rate / 100 if args.rate else None,
    )
    
    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(analysis.to_markdown())
        print(f"\n✅ Report saved to: {output_path}")
    else:
        if RICH_AVAILABLE:
            console.print(Markdown(analysis.to_markdown()))
        else:
            print(analysis.to_markdown())


def run_market(args):
    """Show market analysis."""
    from real_estate import RealEstateAdvisor
    
    print_banner()
    
    advisor = RealEstateAdvisor()
    market = advisor.get_market_data(args.city)
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Market Analysis: {market.city}, {market.state}[/bold]\n")
        
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        if market.metrics:
            table.add_row("Median Price", f"${market.metrics.median_price:,.0f}")
            table.add_row("Price Change (YoY)", f"{market.metrics.price_change_yoy:+.1f}%")
            table.add_row("Median Rent", f"${market.metrics.median_rent:,.0f}")
            table.add_row("Rent Change (YoY)", f"{market.metrics.rent_change_yoy:+.1f}%")
            table.add_row("Vacancy Rate", f"{market.metrics.vacancy_rate:.1f}%")
            table.add_row("Days on Market", f"{market.metrics.days_on_market_avg}")
        
        table.add_row("Investment Score", f"{market.investment_score}/100")
        table.add_row("Market Type", market.market_type)
        table.add_row("Price Trend", market.price_trend)
        
        console.print(table)
    else:
        print(f"Market: {market.city}, {market.state}")
        print(f"Investment Score: {market.investment_score}/100")


def run_interactive(args):
    """Run interactive mode."""
    from real_estate import RealEstateAdvisor, MARKET_DATA
    
    print_banner()
    print_architecture()
    
    print("\nAvailable Markets:", ", ".join(MARKET_DATA.keys()))
    print("\nCommands:")
    print("  search CITY            - Search properties")
    print("  analyze ID             - Analyze property")
    print("  market CITY            - Market analysis")
    print("  compare CITY           - Compare top properties")
    print("  custom                 - Analyze custom property")
    print("  quit                   - Exit")
    print()
    
    advisor = RealEstateAdvisor()
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold green]🏠 >>> [/bold green]")
            else:
                user_input = input("🏠 >>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Happy investing! 👋")
            break
        
        parts = user_input.strip().split(maxsplit=1)
        if not parts:
            continue
        
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd in ("quit", "exit", "q"):
            print("Goodbye! Happy investing! 👋")
            break
        
        elif cmd == "search" and arg:
            props = advisor.search_properties(city=arg)
            for prop in props[:5]:
                print(f"  {prop.property_id[:12]} | ${prop.list_price:,.0f} | {prop.address}")
        
        elif cmd == "analyze" and arg:
            prop = advisor.get_property(arg)
            if not prop:
                # Find by partial
                all_props = advisor.search_properties()
                for p in all_props:
                    if arg in p.property_id:
                        prop = p
                        break
            
            if prop:
                analysis = advisor.analyze_property(prop)
                if RICH_AVAILABLE:
                    console.print(Markdown(analysis.to_markdown()))
                else:
                    print(analysis.to_markdown())
            else:
                print(f"Property not found: {arg}")
        
        elif cmd == "market" and arg:
            market = advisor.get_market_data(arg)
            print(f"  {market.city}: Score {market.investment_score}/100, {market.market_type}")
            if market.metrics:
                print(f"  Median Price: ${market.metrics.median_price:,} | Rent: ${market.metrics.median_rent:,}")
        
        elif cmd == "compare" and arg:
            props = advisor.search_properties(city=arg)
            comparisons = advisor.compare_properties(props[:5])
            for i, comp in enumerate(comparisons, 1):
                print(f"  {i}. ${comp['property']['price']:,.0f} | Cap: {comp['cap_rate']:.1f}% | CoC: {comp['cash_on_cash']:.1f}%")
        
        elif cmd == "custom":
            print("Enter property details:")
            try:
                address = input("  Address: ")
                city = input("  City: ")
                state = input("  State: ")
                price = float(input("  Price: $"))
                beds = int(input("  Bedrooms: "))
                baths = float(input("  Bathrooms: "))
                sqft = int(input("  Square feet: "))
                
                prop = advisor.create_custom_property(
                    address=address, city=city, state=state, zip_code="00000",
                    price=price, bedrooms=beds, bathrooms=baths, sqft=sqft,
                )
                
                analysis = advisor.analyze_property(prop)
                if RICH_AVAILABLE:
                    console.print(Markdown(analysis.to_markdown()))
                else:
                    print(analysis.to_markdown())
            except ValueError:
                print("Invalid input")
        
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
        description="Real Estate Investment Advisor - AI-Powered Property Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Search for properties
    python main.py search Austin --max-price 500000
    
    # Analyze a property
    python main.py analyze prop-abc123
    python main.py analyze prop-abc123 --down-payment 20 --rate 6.5
    
    # Market analysis
    python main.py market Austin
    
    # Interactive mode
    python main.py interactive
    
    # Web server
    python main.py serve

Available Markets:
    Austin, Phoenix, Tampa, Denver, Nashville, Dallas, Atlanta
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for properties")
    search_parser.add_argument("city", help="City to search")
    search_parser.add_argument("--max-price", type=float, default=float('inf'))
    search_parser.add_argument("--min-beds", type=int, default=0)
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a property")
    analyze_parser.add_argument("property_id", help="Property ID")
    analyze_parser.add_argument("--down-payment", type=float, help="Down payment %")
    analyze_parser.add_argument("--rate", type=float, help="Interest rate %")
    analyze_parser.add_argument("--output", "-o", help="Output file")
    
    # Market command
    market_parser = subparsers.add_parser("market", help="Market analysis")
    market_parser.add_argument("city", help="City name")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "search":
        run_search(args)
    elif args.command == "analyze":
        run_analyze(args)
    elif args.command == "market":
        run_market(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
