#!/usr/bin/env python3
"""
E-commerce Personal Shopper - Command Line Interface

Usage:
    python main.py shop "winter jacket, sustainable, budget $200"
    python main.py search "running shoes"
    python main.py deals
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
            "[bold green]🛍️ Personal Shopper[/bold green]\n"
            "[dim]AI-Powered Shopping Assistant[/dim]",
            border_style="green"
        ))
    else:
        print("\n" + "="*50)
        print("🛍️ Personal Shopper")
        print("AI-Powered Shopping Assistant")
        print("="*50 + "\n")


def print_team():
    """Print the shopping team."""
    if RICH_AVAILABLE:
        console.print("""
[dim]Shopping Team:[/dim]
    🎩 Concierge → [cyan]Style Advisor[/cyan] | [blue]Search Agent[/blue] | [yellow]Deals Finder[/yellow]
                         ↓                    ↓                    ↓
                  [cyan]Price Compare[/cyan]    [blue]Review Analyzer[/blue]
                         └─────────────┬─────────────┘
                                       ↓
                              [green]Recommender[/green]
""")


def run_shop(args):
    """Run shopping query."""
    from shopper import PersonalShopperCrew
    
    print_banner()
    print_team()
    
    query = args.query
    budget = args.budget
    sustainable = args.sustainable
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Shopping for:[/bold] {query}")
        if budget:
            console.print(f"[dim]Budget: ${budget:.2f}[/dim]")
        if sustainable:
            console.print("[dim]♻️ Sustainable products only[/dim]")
        console.print()
    
    shopper = PersonalShopperCrew()
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Shopping assistants working...", total=None)
            result = shopper.shop(query, budget, sustainable)
    else:
        print("Searching...")
        result = shopper.shop(query, budget, sustainable)
    
    # Display results
    if RICH_AVAILABLE:
        console.print(f"\n[bold green]Found {len(result.recommendations)} recommendations![/bold green]\n")
        
        for rec in result.recommendations[:5]:
            # Product panel
            product = rec.product
            
            price_str = f"${rec.final_price:.2f}"
            if rec.final_price < product.price:
                price_str += f" [dim](was ${product.price:.2f})[/dim]"
            
            content = f"""**{product.brand}** | {price_str}

⭐ {product.rating}/5 ({product.review_count:,} reviews)

**Why we recommend:** {', '.join(rec.match_reasons)}
"""
            
            if rec.applicable_deals:
                content += "\n**Deals:**\n"
                for deal in rec.applicable_deals[:2]:
                    content += f"🏷️ {deal.description}"
                    if deal.code:
                        content += f" (code: `{deal.code}`)"
                    content += "\n"
            
            if product.is_sustainable:
                content += f"\n♻️ {', '.join(product.sustainability_certifications[:2])}"
            
            content += f"\n\n🏪 {product.retailer.value.title()}"
            if product.ships_free:
                content += " | 🚚 Free shipping"
            
            emoji = "🥇" if rec.rank == 1 else "🥈" if rec.rank == 2 else "🥉" if rec.rank == 3 else "📦"
            
            console.print(Panel(
                Markdown(content),
                title=f"{emoji} {rec.rank}. {product.name}",
                border_style="green" if rec.rank == 1 else "blue" if rec.rank <= 3 else "white"
            ))
        
        # Deals summary
        if result.deals:
            console.print("\n[bold]💰 Available Deals:[/bold]")
            for deal in result.deals[:3]:
                console.print(f"  🏷️ {deal.description}", end="")
                if deal.code:
                    console.print(f" - code: [bold]{deal.code}[/bold]")
                else:
                    console.print()
    else:
        print(result.to_display())


def run_search(args):
    """Quick search."""
    from shopper import quick_search
    
    print_banner()
    
    products = quick_search(args.query)
    
    if RICH_AVAILABLE:
        table = Table(title=f"Search Results: {args.query}")
        table.add_column("Product", style="cyan")
        table.add_column("Brand")
        table.add_column("Price", justify="right")
        table.add_column("Rating", justify="center")
        table.add_column("Retailer")
        
        for product in products:
            sustainable = "♻️ " if product.is_sustainable else ""
            table.add_row(
                sustainable + product.name,
                product.brand,
                f"${product.price:.2f}",
                f"⭐ {product.rating}",
                product.retailer.value.title(),
            )
        
        console.print(table)
    else:
        for product in products:
            print(f"• {product.name} by {product.brand} - ${product.price:.2f}")


def run_deals(args):
    """Show current deals."""
    from shopper import find_deals
    
    print_banner()
    
    deals = find_deals()
    
    if RICH_AVAILABLE:
        console.print("[bold]🏷️ Current Deals & Coupons[/bold]\n")
        
        for deal in deals:
            content = f"**{deal.description}**\n"
            if deal.code:
                content += f"Code: `{deal.code}`\n"
            if deal.discount_percent:
                content += f"Save: {deal.discount_percent}% off\n"
            elif deal.discount_amount:
                content += f"Save: ${deal.discount_amount:.2f}\n"
            if deal.valid_until:
                content += f"Expires: {deal.valid_until}"
            
            console.print(Panel(content, border_style="yellow"))
    else:
        for deal in deals:
            print(f"🏷️ {deal.description}")
            if deal.code:
                print(f"   Code: {deal.code}")


def run_interactive(args):
    """Interactive shopping mode."""
    from shopper import PersonalShopperCrew, quick_search, find_deals
    
    print_banner()
    print_team()
    
    print("\nCommands:")
    print("  shop QUERY        - Full shopping assistance")
    print("  search QUERY      - Quick product search")
    print("  deals             - Show current deals")
    print("  quit              - Exit")
    print()
    
    shopper = PersonalShopperCrew()
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold green]🛍️ >>> [/bold green]")
            else:
                user_input = input("🛍️ >>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nHappy shopping! 👋")
            break
        
        parts = user_input.strip().split(maxsplit=1)
        if not parts:
            continue
        
        cmd = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ""
        
        if cmd in ("quit", "exit", "q"):
            print("Happy shopping! 👋")
            break
        
        elif cmd == "shop" and query:
            result = shopper.shop(query)
            
            if RICH_AVAILABLE:
                console.print(f"\n[bold]Top Recommendations:[/bold]\n")
                for rec in result.recommendations[:3]:
                    console.print(f"  {rec.rank}. [cyan]{rec.product.name}[/cyan] by {rec.product.brand}")
                    console.print(f"     ${rec.final_price:.2f} | ⭐ {rec.product.rating}/5")
                    if rec.applicable_deals:
                        console.print(f"     🏷️ {rec.applicable_deals[0].description}")
                    console.print()
            else:
                for rec in result.recommendations[:3]:
                    print(f"{rec.rank}. {rec.product.name} - ${rec.final_price:.2f}")
        
        elif cmd == "search" and query:
            products = quick_search(query)
            for product in products[:5]:
                print(f"• {product.name} by {product.brand} - ${product.price:.2f}")
        
        elif cmd == "deals":
            deals = find_deals()
            for deal in deals[:5]:
                print(f"🏷️ {deal.description}", end="")
                if deal.code:
                    print(f" (code: {deal.code})")
                else:
                    print()
        
        else:
            # Treat as shopping query
            if user_input.strip():
                result = shopper.shop(user_input.strip())
                for rec in result.recommendations[:3]:
                    print(f"• {rec.product.name} - ${rec.final_price:.2f}")


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
        description="Personal Shopper - AI Shopping Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full shopping assistance
    python main.py shop "winter jacket, sustainable, budget $200"
    
    # Quick search
    python main.py search "running shoes"
    
    # Show deals
    python main.py deals
    
    # Interactive mode
    python main.py interactive
    
    # Web server
    python main.py serve
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Shop command
    shop_parser = subparsers.add_parser("shop", help="Full shopping assistance")
    shop_parser.add_argument("query", help="What you're looking for")
    shop_parser.add_argument("--budget", "-b", type=float, help="Maximum budget")
    shop_parser.add_argument("--sustainable", "-s", action="store_true", help="Sustainable only")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Quick product search")
    search_parser.add_argument("query", help="Search query")
    
    # Deals command
    deals_parser = subparsers.add_parser("deals", help="Show current deals")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "shop":
        run_shop(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "deals":
        run_deals(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
