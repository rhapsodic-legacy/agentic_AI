#!/usr/bin/env python3
"""
Research Paper Synthesizer - Command Line Interface

Usage:
    python main.py research "Large Language Models"
    python main.py research "LLMs" --format research_proposal --provider gemini
    python main.py search "transformer architecture" --source arxiv
    python main.py interactive
    python main.py serve
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Rich console for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
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
            "[bold blue]📚 Research Paper Synthesizer[/bold blue]\n"
            "[dim]LangGraph-powered Academic Research Assistant[/dim]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*60)
        print("📚 Research Paper Synthesizer")
        print("LangGraph-powered Academic Research Assistant")
        print("="*60 + "\n")


def print_result(result):
    """Print the synthesis result."""
    if RICH_AVAILABLE:
        # Stats panel
        stats_table = Table(show_header=False, box=None)
        stats_table.add_row("📄 Papers Found", str(result.papers_found))
        stats_table.add_row("📖 Papers Analyzed", str(result.papers_analyzed))
        stats_table.add_row("🏷️ Themes Identified", str(result.themes_identified))
        stats_table.add_row("🔍 Gaps Found", str(result.gaps_found))
        stats_table.add_row("⚖️ Contradictions", str(result.contradictions_found))
        stats_table.add_row("🔄 Iterations", str(result.iterations))
        
        console.print(Panel(stats_table, title="Research Statistics", border_style="green"))
        
        # Output
        console.print("\n[bold]Output:[/bold]\n")
        console.print(Markdown(result.output[:5000]))
        
        if len(result.output) > 5000:
            console.print(f"\n[dim]... (output truncated, {len(result.output)} total chars)[/dim]")
        
        # Citations
        if result.citations:
            console.print("\n[bold]References:[/bold]")
            for i, cite in enumerate(result.citations[:10], 1):
                console.print(f"  [{i}] {cite}")
    else:
        print(f"\n📊 Research Statistics:")
        print(f"   Papers Found: {result.papers_found}")
        print(f"   Papers Analyzed: {result.papers_analyzed}")
        print(f"   Themes: {result.themes_identified}")
        print(f"   Gaps: {result.gaps_found}")
        print(f"   Iterations: {result.iterations}")
        print("\n" + "="*60)
        print(result.output[:3000])
        if len(result.output) > 3000:
            print(f"\n... (truncated)")


def run_research(args):
    """Run the research command."""
    from research_synth import ResearchSynthesizer, SynthesizerConfig
    
    print_banner()
    
    topic = ' '.join(args.topic)
    
    if RICH_AVAILABLE:
        console.print(f"[bold]Topic:[/bold] {topic}")
        console.print(f"[dim]Format: {args.format} | Provider: {args.provider}[/dim]\n")
    else:
        print(f"Topic: {topic}")
        print(f"Format: {args.format} | Provider: {args.provider}\n")
    
    # Create config
    config = SynthesizerConfig(
        llm_provider=args.provider,
        max_iterations=args.iterations,
        enable_arxiv=not args.no_arxiv,
        enable_semantic_scholar=not args.no_s2,
        enable_pubmed=not args.no_pubmed,
    )
    
    # Run synthesis
    synthesizer = ResearchSynthesizer(config)
    
    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Researching...", total=None)
            result = synthesizer.run(
                topic=topic,
                research_question=args.question,
                output_format=args.format,
            )
            progress.update(task, completed=True)
    else:
        result = synthesizer.run(
            topic=topic,
            research_question=args.question,
            output_format=args.format,
        )
    
    # Print result
    print_result(result)
    
    # Save output
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = ''.join(c if c.isalnum() else '_' for c in topic[:30])
        output_path = f"./output/{safe_topic}_{timestamp}.md"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(result.output)
    
    print(f"\n✅ Output saved to: {output_path}")
    
    return 0 if result.success else 1


def run_search(args):
    """Run a quick paper search."""
    from research_synth.sources import MultiSourceSearch
    
    print_banner()
    
    query = ' '.join(args.query)
    
    sources = {}
    if args.source in ['all', 'arxiv']:
        sources['arxiv'] = True
    if args.source in ['all', 's2', 'semantic_scholar']:
        sources['semantic_scholar'] = True
    if args.source in ['all', 'pubmed']:
        sources['pubmed'] = True
    
    search = MultiSourceSearch(
        enable_arxiv=sources.get('arxiv', False) or args.source == 'all',
        enable_semantic_scholar=sources.get('semantic_scholar', False) or args.source == 'all',
        enable_pubmed=sources.get('pubmed', False) or args.source == 'all',
    )
    
    print(f"🔍 Searching for: {query}\n")
    
    papers = search.search(query, max_results_per_source=args.max)
    
    if RICH_AVAILABLE:
        table = Table(title=f"Found {len(papers)} papers")
        table.add_column("Year", style="cyan", width=6)
        table.add_column("Title", style="white")
        table.add_column("Source", style="green", width=10)
        table.add_column("Citations", justify="right", width=8)
        
        for paper in papers[:20]:
            table.add_row(
                str(paper.year),
                paper.title[:60] + "..." if len(paper.title) > 60 else paper.title,
                paper.source.value,
                str(paper.citations_count),
            )
        
        console.print(table)
    else:
        print(f"Found {len(papers)} papers:\n")
        for i, paper in enumerate(papers[:20], 1):
            print(f"{i}. [{paper.year}] {paper.title[:70]}")
            print(f"   Source: {paper.source.value} | Citations: {paper.citations_count}")
            print()
    
    return 0


def run_interactive(args):
    """Run interactive mode."""
    from research_synth import ResearchSynthesizer, SynthesizerConfig
    
    print_banner()
    
    config = SynthesizerConfig(
        llm_provider=args.provider,
        max_iterations=3,
    )
    synthesizer = ResearchSynthesizer(config)
    
    print("Commands:")
    print("  /research TOPIC     - Research a topic")
    print("  /format FORMAT      - Set output format")
    print("  /provider PROVIDER  - Change LLM provider")
    print("  /graph              - Show workflow graph")
    print("  /quit               - Exit")
    print()
    
    current_format = "literature_review"
    
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
            
            elif cmd == "/research" and arg:
                print(f"\n🔬 Researching: {arg}\n")
                result = synthesizer.run(arg, output_format=current_format)
                print_result(result)
            
            elif cmd == "/format" and arg:
                formats = ["literature_review", "research_proposal", "blog_post", 
                          "executive_summary", "annotated_bibliography"]
                if arg in formats:
                    current_format = arg
                    print(f"✅ Format set to: {current_format}")
                else:
                    print(f"Invalid format. Choose from: {', '.join(formats)}")
            
            elif cmd == "/provider" and arg:
                if arg in ["gemini", "anthropic", "openai"]:
                    config.llm_provider = arg
                    synthesizer = ResearchSynthesizer(config)
                    print(f"✅ Provider changed to: {arg}")
                else:
                    print("Invalid provider. Use: gemini, anthropic, or openai")
            
            elif cmd == "/graph":
                print(synthesizer.get_graph_visualization())
            
            else:
                print(f"Unknown command: {cmd}")
        
        else:
            # Treat as research topic
            print(f"\n🔬 Researching: {user_input}\n")
            result = synthesizer.run(user_input, output_format=current_format)
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
        description="Research Paper Synthesizer - AI-powered academic research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Research a topic
    python main.py research "Large Language Models in Healthcare"
    
    # Create a research proposal
    python main.py research "AI Ethics" --format research_proposal
    
    # Search for papers
    python main.py search "transformer architecture" --source arxiv
    
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
        help="LLM provider (default: gemini)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Research command
    research_parser = subparsers.add_parser("research", help="Research a topic")
    research_parser.add_argument("topic", nargs="+", help="Research topic")
    research_parser.add_argument(
        "--format", "-f",
        choices=["literature_review", "research_proposal", "blog_post", 
                 "executive_summary", "annotated_bibliography"],
        default="literature_review",
        help="Output format"
    )
    research_parser.add_argument("--question", "-q", help="Specific research question")
    research_parser.add_argument("--output", "-o", help="Output file path")
    research_parser.add_argument("--iterations", "-i", type=int, default=3, help="Max iterations")
    research_parser.add_argument("--no-arxiv", action="store_true", help="Disable Arxiv")
    research_parser.add_argument("--no-s2", action="store_true", help="Disable Semantic Scholar")
    research_parser.add_argument("--no-pubmed", action="store_true", help="Disable PubMed")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for papers")
    search_parser.add_argument("query", nargs="+", help="Search query")
    search_parser.add_argument(
        "--source", "-s",
        choices=["all", "arxiv", "s2", "pubmed"],
        default="all",
        help="Paper source"
    )
    search_parser.add_argument("--max", "-m", type=int, default=5, help="Max results per source")
    
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
    elif args.command == "search":
        sys.exit(run_search(args))
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
