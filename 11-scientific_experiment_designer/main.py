#!/usr/bin/env python3
"""
Scientific Experiment Designer - Command Line Interface

Usage:
    python main.py design "Research question" --field Biology
    python main.py examples Psychology
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
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_banner():
    """Print the CLI banner."""
    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold cyan]🔬 Scientific Experiment Designer[/bold cyan]\n"
            "[dim]AI-Powered Research Protocol Generation[/dim]",
            border_style="cyan"
        ))
    else:
        print("\n" + "="*50)
        print("🔬 Scientific Experiment Designer")
        print("AI-Powered Research Protocol Generation")
        print("="*50 + "\n")


def print_architecture():
    """Print the nested team architecture."""
    if RICH_AVAILABLE:
        console.print("""
[dim]Nested Team Architecture:[/dim]
┌────────────────────────────────────────────────────────────┐
│              [bold]PRINCIPAL INVESTIGATOR[/bold]                        │
│           (Oversees research project)                      │
└──────────────────────┬─────────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
[cyan]HYPOTHESIS[/cyan]        [yellow]EXPERIMENT[/yellow]         [green]ANALYSIS[/green]
[cyan]TEAM[/cyan]              [yellow]DESIGN TEAM[/yellow]        [green]TEAM[/green]
┌───────────┐    ┌───────────┐    ┌───────────┐
│Literature │    │Protocol   │    │Statistics │
│Gap Finder │    │Controls   │    │Visualizer │
│Hypothesis │    │Safety     │    │Interpreter│
└───────────┘    └───────────┘    └───────────┘
""")


def run_design(args):
    """Design an experiment."""
    from experiment_designer import ExperimentDesigner
    
    print_banner()
    print_architecture()
    
    designer = ExperimentDesigner()
    
    result = designer.design(
        research_question=args.question,
        field=args.field,
        experiment_type=args.type,
        effect_size=args.effect_size,
        duration_days=args.duration,
    )
    
    proposal = result["proposal"]
    
    # Output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(proposal.to_protocol_markdown())
        print(f"\n✅ Protocol saved to: {output_path}")
    else:
        if RICH_AVAILABLE:
            console.print(Markdown(proposal.to_protocol_markdown()))
        else:
            print(proposal.to_protocol_markdown())


def run_examples(args):
    """Show example research questions."""
    from experiment_designer import ExperimentDesigner, EXAMPLE_RESEARCH_QUESTIONS
    
    print_banner()
    
    if RICH_AVAILABLE:
        for field, questions in EXAMPLE_RESEARCH_QUESTIONS.items():
            console.print(f"\n[bold]{field.value}[/bold]")
            for q in questions:
                console.print(f"  • {q}")
    else:
        for field, questions in EXAMPLE_RESEARCH_QUESTIONS.items():
            print(f"\n{field.value}")
            for q in questions:
                print(f"  • {q}")


def run_interactive(args):
    """Interactive mode."""
    from experiment_designer import ExperimentDesigner, ResearchField, ExperimentType
    
    print_banner()
    print_architecture()
    
    print("\nAvailable Fields:", ", ".join([f.value for f in ResearchField]))
    print("\nAvailable Types:", ", ".join([t.value[:20] for t in list(ExperimentType)[:5]]))
    print("\nCommands:")
    print("  design QUESTION     - Design experiment")
    print("  examples [FIELD]    - Show example questions")
    print("  quit                - Exit")
    print()
    
    designer = ExperimentDesigner()
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold cyan]🔬 >>> [/bold cyan]")
            else:
                user_input = input("🔬 >>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Happy researching! 👋")
            break
        
        parts = user_input.strip().split(maxsplit=1)
        if not parts:
            continue
        
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd in ("quit", "exit", "q"):
            print("Goodbye! Happy researching! 👋")
            break
        
        elif cmd == "design" and arg:
            try:
                field = input("Field (Biology/Psychology/Medicine): ").strip() or "Biology"
                result = designer.design(arg, field=field)
                if RICH_AVAILABLE:
                    console.print(Markdown(result["proposal"].to_protocol_markdown()))
                else:
                    print(result["proposal"].to_protocol_markdown())
            except Exception as e:
                print(f"Error: {e}")
        
        elif cmd == "examples":
            from experiment_designer import EXAMPLE_RESEARCH_QUESTIONS
            field_filter = arg.lower() if arg else None
            
            for field, questions in EXAMPLE_RESEARCH_QUESTIONS.items():
                if field_filter and field_filter not in field.value.lower():
                    continue
                print(f"\n{field.value}:")
                for q in questions:
                    print(f"  • {q}")
        
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
        description="Scientific Experiment Designer - AI-Powered Research Protocol Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Design an experiment
    python main.py design "Does cognitive training improve memory?" --field Psychology
    
    # Design with custom parameters
    python main.py design "Effect of drug X on outcome Y" --field Medicine --type RCT --duration 84
    
    # Save protocol to file
    python main.py design "Question" --field Biology -o protocol.md
    
    # Show example questions
    python main.py examples Psychology
    
    # Interactive mode
    python main.py interactive
    
    # Web server
    python main.py serve

Research Fields:
    Biology, Psychology, Medicine, Pharmacology, Chemistry, Physics,
    Neuroscience, Ecology, Genetics, Computer Science, Social Science
    
Experiment Types:
    RCT, Factorial, Crossover, Cohort, Case-Control, Observational
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Design command
    design_parser = subparsers.add_parser("design", help="Design an experiment")
    design_parser.add_argument("question", help="Research question")
    design_parser.add_argument("--field", default="Biology", help="Research field")
    design_parser.add_argument("--type", default="RCT", help="Experiment type")
    design_parser.add_argument("--effect-size", type=float, default=0.5)
    design_parser.add_argument("--duration", type=int, default=56, help="Duration in days")
    design_parser.add_argument("--output", "-o", help="Output file")
    
    # Examples command
    examples_parser = subparsers.add_parser("examples", help="Show example questions")
    examples_parser.add_argument("field", nargs="?", help="Filter by field")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "design":
        run_design(args)
    elif args.command == "examples":
        run_examples(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
