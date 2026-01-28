#!/usr/bin/env python3
"""
📦 Supply Chain Optimizer - Command Line Interface

End-to-end supply chain optimization with hierarchical multi-agent system.

Usage:
    python main.py optimize
    python main.py analyze
    python main.py scenario --demand-change 20
    python main.py serve
"""

import argparse
import sys
import os
from datetime import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.tree import Tree
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_banner():
    """Print the application banner."""
    if RICH_AVAILABLE:
        banner = """
[bold blue]
   _____ _    _ _____  _____  _  __     __   _____ _    _          _____ _   _ 
  / ____| |  | |  __ \|  __ \| | \ \   / /  / ____| |  | |   /\   |_   _| \ | |
 | (___ | |  | | |__) | |__) | |  \ \_/ /  | |    | |__| |  /  \    | | |  \| |
  \___ \| |  | |  ___/|  ___/| |   \   /   | |    |  __  | / /\ \   | | | . ` |
  ____) | |__| | |    | |    | |____| |    | |____| |  | |/ ____ \ _| |_| |\  |
 |_____/ \____/|_|    |_|    |______|_|     \_____|_|  |_/_/    \_\_____|_| \_|
[/bold blue]
[bold cyan]                    O P T I M I Z E R[/bold cyan]
        """
        console.print(banner)
        console.print(Panel.fit(
            "[dim]Hierarchical Multi-Agent Supply Chain Optimization[/dim]\n"
            "[green]CrewAI Framework | Strategic → Tactical → Operational[/green]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*60)
        print("       📦 SUPPLY CHAIN OPTIMIZER")
        print("    Hierarchical Multi-Agent System")
        print("="*60 + "\n")


def print_hierarchy():
    """Print the agent hierarchy."""
    if RICH_AVAILABLE:
        tree = Tree("🏢 [bold]Supply Chain Director[/bold] (Strategic)")
        
        # Demand Planning
        demand = tree.add("📊 [cyan]Demand Planning Manager[/cyan]")
        demand.add("📈 Forecast Agent")
        demand.add("🔍 Market Analyst")
        
        # Inventory
        inventory = tree.add("📦 [cyan]Inventory Manager[/cyan]")
        inventory.add("🔄 Replenishment Agent")
        inventory.add("🛡️ Safety Stock Agent")
        
        # Logistics
        logistics = tree.add("🚛 [cyan]Logistics Manager[/cyan]")
        logistics.add("🗺️ Route Optimizer")
        logistics.add("📋 Carrier Selector")
        
        console.print(Panel(tree, title="Agent Hierarchy", border_style="blue"))


def print_results_table(result):
    """Print optimization results in a table."""
    if RICH_AVAILABLE:
        # Summary table
        table = Table(title="📊 Optimization Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Current", style="red")
        table.add_column("Optimized", style="green")
        table.add_column("Change", style="yellow")
        
        table.add_row(
            "Inventory Value",
            f"${result.current_inventory_value:,.0f}",
            f"${result.recommended_inventory_value:,.0f}",
            f"{result.inventory_reduction_pct:+.1f}%"
        )
        table.add_row(
            "Stockout Risk",
            f"{result.current_stockout_risk:.1%}",
            f"{result.recommended_stockout_risk:.1%}",
            f"{(result.recommended_stockout_risk - result.current_stockout_risk)*100:+.1f}pp"
        )
        table.add_row(
            "Shipping Routes",
            str(result.routes_before),
            str(result.routes_after),
            f"{((result.routes_after - result.routes_before)/max(1,result.routes_before))*100:+.1f}%"
        )
        table.add_row(
            "Carbon Footprint",
            "Baseline",
            "-",
            f"{result.carbon_reduction_pct:+.1f}%"
        )
        
        console.print(table)
        
        # Key metrics
        console.print(f"\n💰 [bold green]Working Capital Freed:[/bold green] ${result.working_capital_freed:,.0f}")
        console.print(f"📦 [bold blue]Monthly Shipping Savings:[/bold blue] ${result.shipping_cost_savings:,.0f}")
        
        # Alerts
        if result.alerts:
            console.print(f"\n⚠️ [bold yellow]Alerts ({len(result.alerts)}):[/bold yellow]")
            for alert in result.alerts[:5]:
                console.print(f"  • {alert}")
    else:
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Inventory Reduction: {result.inventory_reduction_pct:.1f}%")
        print(f"Working Capital Freed: ${result.working_capital_freed:,.0f}")
        print(f"Shipping Savings: ${result.shipping_cost_savings:,.0f}")


def run_optimize(args):
    """Run supply chain optimization."""
    from supply_chain import SupplyChainOptimizer, OptimizationConfig
    
    print_banner()
    print_hierarchy()
    
    config = OptimizationConfig(
        verbose=not args.quiet,
        output_dir=args.output or "output",
    )
    
    optimizer = SupplyChainOptimizer(config)
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running optimization...", total=None)
            result = optimizer.optimize()
            progress.remove_task(task)
    else:
        result = optimizer.optimize()
    
    # Display results
    print_results_table(result)
    
    # Save report
    if args.save:
        filepath = optimizer.save_results(result)
        if RICH_AVAILABLE:
            console.print(f"\n📄 Report saved to: [cyan]{filepath}[/cyan]")


def run_analyze(args):
    """Run supply chain analysis."""
    from supply_chain import SupplyChainOptimizer, OptimizationConfig
    from supply_chain.tools import ReportGenerator
    from supply_chain.data import SUPPLIERS, generate_inventory_levels, PRODUCTS
    
    print_banner()
    
    if RICH_AVAILABLE:
        console.print("[bold]Running Supply Chain Analysis...[/bold]\n")
    
    # Inventory analysis
    inventory = generate_inventory_levels()
    
    if RICH_AVAILABLE:
        table = Table(title="📦 Inventory Status", box=box.ROUNDED)
        table.add_column("SKU", style="cyan")
        table.add_column("Product", style="white")
        table.add_column("On Hand", justify="right")
        table.add_column("Safety Stock", justify="right")
        table.add_column("Status", style="bold")
        
        for sku, inv in list(inventory.items())[:10]:
            product = PRODUCTS.get(sku)
            name = product.name[:25] if product else "Unknown"
            
            status_style = {
                "STOCKOUT": "[red]STOCKOUT[/red]",
                "CRITICAL": "[yellow]CRITICAL[/yellow]",
                "REORDER": "[blue]REORDER[/blue]",
                "HEALTHY": "[green]HEALTHY[/green]",
                "OVERSTOCK": "[magenta]OVERSTOCK[/magenta]",
            }.get(inv.stock_status, inv.stock_status)
            
            table.add_row(sku, name, str(inv.on_hand), str(inv.safety_stock), status_style)
        
        console.print(table)
    
    # Supplier analysis
    suppliers = list(SUPPLIERS.values())
    
    if RICH_AVAILABLE:
        console.print("\n")
        table = Table(title="🏭 Supplier Scorecard", box=box.ROUNDED)
        table.add_column("Rank", justify="center")
        table.add_column("Supplier", style="cyan")
        table.add_column("Score", justify="right")
        table.add_column("Quality", justify="right")
        table.add_column("Delivery", justify="right")
        table.add_column("Risk")
        
        sorted_suppliers = sorted(suppliers, key=lambda s: s.overall_score, reverse=True)
        
        for i, s in enumerate(sorted_suppliers, 1):
            table.add_row(
                str(i),
                s.name[:20],
                f"{s.overall_score:.0f}",
                f"{s.quality_score:.0f}",
                f"{s.delivery_score:.0f}",
                s.risk_indicator
            )
        
        console.print(table)


def run_scenario(args):
    """Run scenario analysis."""
    from supply_chain import (
        SupplyChainOptimizer, OptimizationConfig,
        create_demand_shock_scenario, create_supply_disruption_scenario
    )
    
    print_banner()
    
    if RICH_AVAILABLE:
        console.print("[bold]Running Scenario Analysis...[/bold]\n")
    
    config = OptimizationConfig(verbose=True)
    optimizer = SupplyChainOptimizer(config)
    
    # First run base optimization to get forecasts
    optimizer.run_demand_planning()
    
    # Create scenario
    if args.demand_change:
        scenario = create_demand_shock_scenario(args.demand_change)
    elif args.supplier_disruption:
        scenario = create_supply_disruption_scenario(
            args.supplier_disruption,
            args.disruption_days or 30
        )
    else:
        scenario = create_demand_shock_scenario(20)  # Default: 20% surge
    
    # Run scenario
    results = optimizer.run_scenario(scenario)
    
    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold]{scenario.name}[/bold]\n\n{scenario.description}",
            title="📊 Scenario",
            border_style="yellow"
        ))
        
        console.print(f"\n[bold]Impact Analysis:[/bold]")
        console.print(f"  • Products at risk: {results.get('products_at_risk', 0)}")
        
        if results.get('stockout_risk_products'):
            console.print(f"\n[yellow]Products with stockout risk:[/yellow]")
            for p in results['stockout_risk_products'][:5]:
                console.print(f"    • {p['sku']}: shortfall of {p['shortfall']} units")


def run_serve(args):
    """Start the API server."""
    import uvicorn
    
    print_banner()
    
    if RICH_AVAILABLE:
        console.print(f"[cyan]🌐 Starting server at http://localhost:{args.port}[/cyan]")
        console.print(f"[cyan]📖 API docs at http://localhost:{args.port}/docs[/cyan]")
        console.print("\n[dim]Press Ctrl+C to stop[/dim]\n")
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def main():
    parser = argparse.ArgumentParser(
        description="📦 Supply Chain Optimizer - Hierarchical Multi-Agent System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full optimization
    python main.py optimize --save
    
    # Analyze current state
    python main.py analyze
    
    # Run demand surge scenario
    python main.py scenario --demand-change 30
    
    # Run supplier disruption scenario
    python main.py scenario --supplier-disruption SUP-001 --disruption-days 45
    
    # Start web server
    python main.py serve
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Optimize command
    opt_parser = subparsers.add_parser("optimize", help="Run supply chain optimization")
    opt_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    opt_parser.add_argument("--save", "-s", action="store_true", help="Save report to file")
    opt_parser.add_argument("--output", "-o", help="Output directory")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze current supply chain state")
    
    # Scenario command
    scenario_parser = subparsers.add_parser("scenario", help="Run what-if scenario analysis")
    scenario_parser.add_argument("--demand-change", type=float, help="Demand change percentage (e.g., 20 for +20%%)")
    scenario_parser.add_argument("--supplier-disruption", help="Supplier ID for disruption scenario")
    scenario_parser.add_argument("--disruption-days", type=int, default=30, help="Disruption duration in days")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "optimize":
        run_optimize(args)
    elif args.command == "analyze":
        run_analyze(args)
    elif args.command == "scenario":
        run_scenario(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        print_banner()
        if RICH_AVAILABLE:
            print_hierarchy()
        parser.print_help()


if __name__ == "__main__":
    main()
