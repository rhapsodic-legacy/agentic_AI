#!/usr/bin/env python3
"""
DevOps Incident Response System - Command Line Interface

Usage:
    python main.py simulate api-gateway high_error_rate
    python main.py status
    python main.py health
    python main.py runbooks
    python main.py interactive
    python main.py serve
"""

import argparse
import sys
from datetime import datetime

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
            "[bold red]🚨 Incident Response System[/bold red]\n"
            "[dim]Autonomous DevOps Incident Management[/dim]",
            border_style="red"
        ))
    else:
        print("\n" + "="*50)
        print("🚨 Incident Response System")
        print("Autonomous DevOps Incident Management")
        print("="*50 + "\n")


def print_hierarchy():
    """Print the agent hierarchy."""
    if RICH_AVAILABLE:
        console.print("""
[dim]Agent Hierarchy:[/dim]
                    ┌─────────────────────────┐
                    │  [bold red]INCIDENT COMMANDER[/bold red]  │
                    └────────────┬────────────┘
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
    [cyan]MONITOR[/cyan]             [yellow]DIAGNOSE[/yellow]              [green]COMMS[/green]
         │                       │
         │              ┌────────┴────────┐
         │              ▼                 ▼
         │         [magenta]INFRA FIXER[/magenta]     [blue]APP FIXER[/blue]
         │              │                 │
         └──────────────┴────────┬────────┘
                        [dim](Feedback Loop)[/dim]
""")


def run_simulate(args):
    """Simulate and respond to an incident."""
    from incident_system import IncidentResponseSystem
    
    print_banner()
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Simulating incident:[/bold] {args.incident_type} on {args.service}")
    else:
        print(f"\nSimulating incident: {args.incident_type} on {args.service}")
    
    system = IncidentResponseSystem()
    
    # Simulate the incident
    alert = system.simulate_incident(args.service, args.incident_type)
    
    if RICH_AVAILABLE:
        console.print(f"\n[yellow]⚠️ Alert Generated:[/yellow] {alert.title}")
        console.print(f"[dim]Severity: {alert.severity.value} | Service: {alert.service}[/dim]\n")
    
    # Respond to the incident
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Incident response in progress...", total=None)
            result = system.respond_to_incident(alert)
    else:
        result = system.respond_to_incident(alert)
    
    # Show summary
    if RICH_AVAILABLE:
        console.print(Panel(
            Markdown(result.incident.to_summary()),
            title="📋 Incident Summary",
            border_style="green"
        ))
        
        # Show actions taken
        if result.actions_taken:
            table = Table(title="Actions Taken")
            table.add_column("Action", style="cyan")
            table.add_column("Target", style="white")
            table.add_column("Result", style="green")
            
            for action in result.actions_taken:
                table.add_row(
                    action.action_type.value,
                    action.target_service,
                    "✅ Success" if action.success else "❌ Failed"
                )
            
            console.print(table)
    else:
        print(result.incident.to_summary())
    
    # Generate postmortem if requested
    if args.postmortem:
        postmortem = system.generate_postmortem(result.incident.incident_id)
        if RICH_AVAILABLE:
            console.print(Panel(
                Markdown(postmortem),
                title="📝 Post-Mortem Report",
                border_style="blue"
            ))
        else:
            print("\n" + postmortem)


def run_status(args):
    """Show system status."""
    from incident_system import IncidentResponseSystem, monitoring, infrastructure
    
    print_banner()
    
    system = IncidentResponseSystem()
    status = system.get_system_status()
    
    if RICH_AVAILABLE:
        console.print("[bold]System Status[/bold]\n")
        console.print(f"Active Incidents: {status['active_incidents']}")
        console.print(f"Resolved Today: {status['resolved_today']}")
        console.print("\n" + status['system_health'])
    else:
        print("System Status")
        print(status['system_health'])


def run_health(args):
    """Show service health."""
    from incident_system import TOOLS
    
    print_banner()
    
    health = TOOLS["get_all_services_health"]()
    
    if RICH_AVAILABLE:
        console.print(Panel(health, title="🏥 Infrastructure Health", border_style="green"))
    else:
        print(health)


def run_runbooks(args):
    """List available runbooks."""
    from incident_system import TOOLS
    
    print_banner()
    
    runbooks = TOOLS["list_available_runbooks"]()
    
    if RICH_AVAILABLE:
        console.print(Panel(runbooks, title="📚 Available Runbooks", border_style="cyan"))
    else:
        print(runbooks)


def run_interactive(args):
    """Run interactive mode."""
    from incident_system import IncidentResponseSystem, TOOLS
    
    print_banner()
    print_hierarchy()
    
    print("\nCommands:")
    print("  simulate SERVICE TYPE  - Simulate incident")
    print("  health                 - Show system health")
    print("  health SERVICE         - Check service health")
    print("  logs SERVICE           - Search error logs")
    print("  scale SERVICE N        - Scale service")
    print("  restart SERVICE        - Restart service")
    print("  rollback SERVICE       - Rollback service")
    print("  runbooks               - List runbooks")
    print("  quit                   - Exit")
    print()
    
    system = IncidentResponseSystem()
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold red]🚨 >>> [/bold red]")
            else:
                user_input = input("🚨 >>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Stay vigilant! 👋")
            break
        
        parts = user_input.strip().split()
        if not parts:
            continue
        
        cmd = parts[0].lower()
        args_list = parts[1:] if len(parts) > 1 else []
        
        if cmd in ("quit", "exit", "q"):
            print("Goodbye! Stay vigilant! 👋")
            break
        
        elif cmd == "simulate" and len(args_list) >= 2:
            service, incident_type = args_list[0], args_list[1]
            alert = system.simulate_incident(service, incident_type)
            result = system.respond_to_incident(alert)
            print(f"\n✅ Incident {result.incident.incident_id} resolved")
        
        elif cmd == "health":
            if args_list:
                result = TOOLS["check_service_health"](args_list[0])
            else:
                result = TOOLS["get_all_services_health"]()
            print(result)
        
        elif cmd == "logs" and args_list:
            result = TOOLS["search_error_logs"](args_list[0])
            print(result)
        
        elif cmd == "scale" and len(args_list) >= 2:
            result = TOOLS["scale_service"](args_list[0], int(args_list[1]))
            print(result)
        
        elif cmd == "restart" and args_list:
            result = TOOLS["restart_service"](args_list[0])
            print(result)
        
        elif cmd == "rollback" and args_list:
            result = TOOLS["rollback_service"](args_list[0])
            print(result)
        
        elif cmd == "runbooks":
            result = TOOLS["list_available_runbooks"]()
            print(result)
        
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
        description="Incident Response System - Autonomous DevOps Incident Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Simulate and respond to an incident
    python main.py simulate api-gateway high_error_rate
    
    # Simulate with post-mortem generation
    python main.py simulate api-gateway service_down --postmortem
    
    # Check system health
    python main.py health
    
    # List runbooks
    python main.py runbooks
    
    # Interactive mode
    python main.py interactive
    
    # Web server
    python main.py serve

Incident Types:
    - high_error_rate
    - high_latency
    - service_down
    - high_cpu
    - database_connections
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Simulate command
    sim_parser = subparsers.add_parser("simulate", help="Simulate and respond to an incident")
    sim_parser.add_argument("service", help="Service name (e.g., api-gateway)")
    sim_parser.add_argument("incident_type", help="Incident type (e.g., high_error_rate)")
    sim_parser.add_argument("--postmortem", "-p", action="store_true", help="Generate post-mortem")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Show service health")
    
    # Runbooks command
    runbooks_parser = subparsers.add_parser("runbooks", help="List available runbooks")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "simulate":
        run_simulate(args)
    elif args.command == "status":
        run_status(args)
    elif args.command == "health":
        run_health(args)
    elif args.command == "runbooks":
        run_runbooks(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
