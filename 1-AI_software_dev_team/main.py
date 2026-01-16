#!/usr/bin/env python3
"""
AI Software Development Team - Command Line Interface

Usage:
    python main.py build "Build a REST API for a todo app"
    python main.py build "Build a todo app" --frontend --full
    python main.py interactive
    python main.py serve
"""

import argparse
import sys
import os
import time
from pathlib import Path

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.markdown import Markdown
    from rich.tree import Tree
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
            "[bold blue]🧑‍💻 AI Software Development Team[/bold blue]\n"
            "[dim]Powered by CrewAI | Multi-Agent Development[/dim]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*60)
        print("🧑‍💻 AI Software Development Team")
        print("Powered by CrewAI | Multi-Agent Development")
        print("="*60 + "\n")


def print_team_info(team_info: dict):
    """Print team configuration."""
    if RICH_AVAILABLE:
        table = Table(title="Development Team")
        table.add_column("Role", style="cyan")
        table.add_column("Goal", style="dim")
        
        for agent in team_info["agents"]:
            table.add_row(agent["role"], agent["goal"])
        
        console.print(table)
        console.print(f"\n[dim]LLM Provider: {team_info['llm_provider']}[/dim]")
    else:
        print("\nDevelopment Team:")
        print("-" * 40)
        for agent in team_info["agents"]:
            print(f"  • {agent['role']}")
        print(f"\nLLM Provider: {team_info['llm_provider']}")


def print_result(result):
    """Print the build result."""
    if RICH_AVAILABLE:
        if result.success:
            console.print(Panel(
                f"[green]✅ Build Successful![/green]\n\n"
                f"📁 Output: {result.output_dir}\n"
                f"📄 Files: {len(result.files_created)}\n"
                f"⏱️  Time: {result.total_time_seconds:.1f}s",
                title="Result",
                border_style="green"
            ))
            
            # Show file tree
            tree = Tree(f"📁 {result.project_name}")
            for f in result.files_created:
                rel_path = os.path.relpath(f, result.output_dir)
                tree.add(f"📄 {rel_path}")
            console.print(tree)
        else:
            console.print(Panel(
                f"[red]❌ Build Failed[/red]\n\n"
                f"Errors:\n" + "\n".join(result.errors),
                title="Result",
                border_style="red"
            ))
    else:
        if result.success:
            print(f"\n✅ Build Successful!")
            print(f"   Output: {result.output_dir}")
            print(f"   Files: {len(result.files_created)}")
            print(f"   Time: {result.total_time_seconds:.1f}s")
            print("\nFiles created:")
            for f in result.files_created:
                print(f"   • {os.path.relpath(f, result.output_dir)}")
        else:
            print(f"\n❌ Build Failed")
            for err in result.errors:
                print(f"   Error: {err}")


def run_build(args):
    """Run the build command."""
    from ai_dev_team import AIDevTeam, DevTeamConfig
    
    print_banner()
    
    # Create config
    config = DevTeamConfig(
        llm_provider=args.provider,
        team_type="full" if args.full else "minimal",
        include_frontend=args.frontend,
        output_dir=args.output,
        verbose=args.verbose,
    )
    
    # Create team
    team = AIDevTeam(config)
    
    # Show team info
    print_team_info(team.get_team_info())
    
    # Get project description
    description = ' '.join(args.description)
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Project:[/bold] {description}\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Building project...", total=100)
            
            # Run the build
            result = team.run(description)
            progress.update(task, completed=100)
    else:
        print(f"\nProject: {description}")
        print("Building project...")
        result = team.run(description)
    
    # Print result
    print_result(result)
    
    return 0 if result.success else 1


def run_interactive(args):
    """Run interactive mode."""
    from ai_dev_team import AIDevTeam, DevTeamConfig
    
    print_banner()
    
    # Create team
    config = DevTeamConfig(
        llm_provider=args.provider,
        team_type="backend",
        verbose=args.verbose,
    )
    team = AIDevTeam(config)
    
    print_team_info(team.get_team_info())
    
    print("\n" + "-"*60)
    print("Commands:")
    print("  /build DESCRIPTION  - Build a project")
    print("  /task TYPE CONTEXT  - Run a single task")
    print("  /team [full|minimal|backend] - Change team type")
    print("  /provider [gemini|anthropic|openai] - Change provider")
    print("  /info               - Show team info")
    print("  /quit               - Exit")
    print("-"*60 + "\n")
    
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
            parts = user_input.split(maxsplit=2)
            cmd = parts[0].lower()
            
            if cmd in ("/quit", "/exit", "/q"):
                print("Goodbye! 👋")
                break
            
            elif cmd == "/build" and len(parts) > 1:
                description = parts[1] if len(parts) == 2 else ' '.join(parts[1:])
                print(f"\n🔨 Building: {description}\n")
                result = team.run(description)
                print_result(result)
            
            elif cmd == "/task" and len(parts) > 2:
                task_type = parts[1]
                context = parts[2]
                print(f"\n⚙️ Running {task_type} task...\n")
                try:
                    output = team.run_single_task(task_type, context)
                    if RICH_AVAILABLE:
                        console.print(Markdown(output))
                    else:
                        print(output)
                except Exception as e:
                    print(f"Error: {e}")
            
            elif cmd == "/team" and len(parts) > 1:
                new_type = parts[1]
                if new_type in ["full", "minimal", "backend"]:
                    config.team_type = new_type
                    team = AIDevTeam(config)
                    print(f"✅ Team changed to: {new_type}")
                    print_team_info(team.get_team_info())
                else:
                    print("Invalid team type. Use: full, minimal, or backend")
            
            elif cmd == "/provider" and len(parts) > 1:
                new_provider = parts[1]
                if new_provider in ["gemini", "anthropic", "openai"]:
                    config.llm_provider = new_provider
                    team = AIDevTeam(config)
                    print(f"✅ Provider changed to: {new_provider}")
                else:
                    print("Invalid provider. Use: gemini, anthropic, or openai")
            
            elif cmd == "/info":
                print_team_info(team.get_team_info())
            
            else:
                print(f"Unknown command: {cmd}")
        
        else:
            # Default: treat as build request
            print(f"\n🔨 Building: {user_input}\n")
            result = team.run(user_input)
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


def run_single_task(args):
    """Run a single task."""
    from ai_dev_team import AIDevTeam, DevTeamConfig
    
    print_banner()
    
    config = DevTeamConfig(
        llm_provider=args.provider,
        team_type="full",
        verbose=args.verbose,
    )
    team = AIDevTeam(config)
    
    context = ' '.join(args.context)
    
    print(f"⚙️ Running {args.task} task...\n")
    
    try:
        output = team.run_single_task(args.task, context)
        
        if RICH_AVAILABLE:
            console.print(Markdown(output))
        else:
            print(output)
        
        # Optionally save to file
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"\n✅ Output saved to {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="AI Software Development Team - Build projects with AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick build
    python main.py build "Build a REST API for a todo app"
    
    # Full build with frontend
    python main.py build "Build a todo app" --frontend --full
    
    # Interactive mode
    python main.py interactive
    
    # Single task
    python main.py task requirements "Build a user authentication system"
    
    # Start web server
    python main.py serve
        """
    )
    
    parser.add_argument(
        "--provider", "-p",
        choices=["gemini", "anthropic", "openai"],
        default="gemini",
        help="LLM provider (default: gemini)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build a project from description")
    build_parser.add_argument("description", nargs="+", help="Project description")
    build_parser.add_argument("--frontend", "-f", action="store_true", help="Include frontend development")
    build_parser.add_argument("--full", action="store_true", help="Use full team (all agents)")
    build_parser.add_argument("--output", "-o", default="./output", help="Output directory")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive CLI mode")
    
    # Task command
    task_parser = subparsers.add_parser("task", help="Run a single task")
    task_parser.add_argument(
        "task",
        choices=["requirements", "architecture", "backend", "frontend", 
                 "tests", "devops", "security", "docs"],
        help="Task type to run"
    )
    task_parser.add_argument("context", nargs="+", help="Task context/description")
    task_parser.add_argument("--output", "-o", help="Save output to file")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "build":
        sys.exit(run_build(args))
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "task":
        sys.exit(run_single_task(args))
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
