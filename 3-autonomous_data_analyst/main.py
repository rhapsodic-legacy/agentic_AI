#!/usr/bin/env python3
"""
Autonomous Data Analyst - Command Line Interface

Usage:
    python main.py connect data.csv
    python main.py analyze "What are the top products?"
    python main.py interactive
    python main.py serve
"""

import argparse
import sys
import os
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
            "[bold blue]📊 Autonomous Data Analyst[/bold blue]\n"
            "[dim]Conversational AI for Data Analysis[/dim]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*50)
        print("📊 Autonomous Data Analyst")
        print("Conversational AI for Data Analysis")
        print("="*50 + "\n")


def print_result(result):
    """Print analysis result."""
    if RICH_AVAILABLE:
        if result.success:
            console.print(f"[green]✓ Query returned {result.row_count} rows[/green]")
            
            if result.data and len(result.data) <= 20:
                table = Table(title="Results")
                for col in result.columns:
                    table.add_column(col)
                for row in result.data[:20]:
                    table.add_row(*[str(row.get(c, "")) for c in result.columns])
                console.print(table)
            
            if result.insights:
                console.print("\n[bold]Insights:[/bold]")
                for insight in result.insights[:5]:
                    console.print(f"  • {insight['title']}: {insight['description']}")
        else:
            console.print(f"[red]✗ Error: {result.error}[/red]")
    else:
        if result.success:
            print(f"✓ Query returned {result.row_count} rows")
            if result.data:
                import pandas as pd
                df = pd.DataFrame(result.data[:20])
                print(df.to_string())
            if result.insights:
                print("\nInsights:")
                for insight in result.insights[:5]:
                    print(f"  • {insight['title']}")
        else:
            print(f"✗ Error: {result.error}")


def run_connect(args):
    """Connect to a data source."""
    from data_analyst import AutonomousDataAnalyst
    
    print_banner()
    
    analyst = AutonomousDataAnalyst()
    filepath = args.source
    
    # Determine type
    if filepath.endswith('.csv'):
        success = analyst.connect_csv(filepath)
    elif filepath.endswith(('.xlsx', '.xls')):
        success = analyst.connect_excel(filepath)
    elif filepath.endswith('.parquet'):
        success = analyst.connect_parquet(filepath)
    elif filepath.endswith(('.db', '.sqlite', '.sqlite3')):
        success = analyst.connect_sqlite(filepath)
    else:
        print(f"Unknown file type: {filepath}")
        return 1
    
    if success:
        print(f"✓ Connected to {filepath}")
        print("\nTables:")
        for table in analyst.get_tables():
            print(f"  • {table}")
        print("\n" + analyst.describe())
    else:
        print(f"✗ Failed to connect to {filepath}")
        return 1
    
    return 0


def run_analyze(args):
    """Run an analysis."""
    from data_analyst import AutonomousDataAnalyst, AnalystConfig
    
    print_banner()
    
    config = AnalystConfig(llm_provider=args.provider)
    analyst = AutonomousDataAnalyst(config)
    
    # Connect to source
    if not analyst.connect_csv(args.source) and not analyst.connect_sqlite(args.source):
        print(f"✗ Could not connect to {args.source}")
        return 1
    
    # Run analysis
    question = ' '.join(args.question)
    print(f"\n🔍 Analyzing: {question}\n")
    
    result = analyst.analyze(question)
    print_result(result)
    
    return 0 if result.success else 1


def run_interactive(args):
    """Run interactive mode."""
    from data_analyst import AutonomousDataAnalyst, AnalystConfig
    
    print_banner()
    
    config = AnalystConfig(llm_provider=args.provider)
    analyst = AutonomousDataAnalyst(config)
    
    print("Commands:")
    print("  /connect FILE       - Connect to a data file")
    print("  /tables             - List tables")
    print("  /describe           - Describe data source")
    print("  /profile TABLE      - Profile a table")
    print("  /sql QUERY          - Run SQL query")
    print("  /visualize TYPE X Y - Create a chart")
    print("  /report TITLE       - Generate report")
    print("  /quit               - Exit")
    print("\nOr just type a question to analyze your data!\n")
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold blue]You:[/bold blue] ")
            else:
                user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break
        
        user_input = user_input.strip()
        
        if not user_input:
            continue
        
        # Commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye! 👋")
                break
            
            elif cmd == "/connect" and arg:
                filepath = arg.strip()
                if filepath.endswith('.csv'):
                    success = analyst.connect_csv(filepath)
                elif filepath.endswith(('.xlsx', '.xls')):
                    success = analyst.connect_excel(filepath)
                elif filepath.endswith(('.db', '.sqlite')):
                    success = analyst.connect_sqlite(filepath)
                else:
                    print("Unsupported file type")
                    continue
                
                if success:
                    print(f"✓ Connected to {filepath}")
                    print(f"Tables: {', '.join(analyst.get_tables())}")
                else:
                    print(f"✗ Failed to connect")
            
            elif cmd == "/tables":
                tables = analyst.get_tables()
                if tables:
                    print("Tables:")
                    for t in tables:
                        print(f"  • {t}")
                else:
                    print("No tables found. Connect to a data source first.")
            
            elif cmd == "/describe":
                print(analyst.describe())
            
            elif cmd == "/profile" and arg:
                print(analyst.profile(arg))
            
            elif cmd == "/sql" and arg:
                result = analyst.query(arg)
                print_result(result)
            
            elif cmd == "/visualize" and arg:
                parts = arg.split()
                if len(parts) >= 3:
                    chart_type, x, y = parts[0], parts[1], parts[2]
                    chart = analyst.visualize(chart_type, x, y)
                    if chart.get("filepath"):
                        print(f"✓ Chart saved to {chart['filepath']}")
                    else:
                        print(f"✗ Error: {chart.get('error', 'Unknown error')}")
                else:
                    print("Usage: /visualize TYPE X_COLUMN Y_COLUMN")
            
            elif cmd == "/report" and arg:
                path = analyst.create_report(arg)
                print(f"✓ Report saved to {path}")
            
            else:
                print(f"Unknown command: {cmd}")
        
        else:
            # Natural language query
            if not analyst.data_source or not analyst.data_source.is_connected:
                print("Please connect to a data source first with /connect FILE")
                continue
            
            result = analyst.analyze(user_input)
            print_result(result)
        
        print()


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
        description="Autonomous Data Analyst - Conversational AI for data analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Connect and describe
    python main.py connect sales_data.csv
    
    # Analyze data
    python main.py analyze sales.csv "What are the top 10 products?"
    
    # Interactive mode
    python main.py interactive
    
    # Web server
    python main.py serve
        """
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["gemini", "anthropic", "openai"],
        default="gemini",
        help="LLM provider"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Connect command
    connect_parser = subparsers.add_parser("connect", help="Connect to a data source")
    connect_parser.add_argument("source", help="Data file path")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze data")
    analyze_parser.add_argument("source", help="Data file path")
    analyze_parser.add_argument("question", nargs="+", help="Analysis question")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "connect":
        sys.exit(run_connect(args))
    elif args.command == "analyze":
        sys.exit(run_analyze(args))
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
