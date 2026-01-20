#!/usr/bin/env python3
"""
Legal Document Analyzer - Command Line Interface

Usage:
    python main.py analyze document.txt
    python main.py sample saas
    python main.py classify document.txt
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
            "[bold blue]⚖️ Legal Document Analyzer[/bold blue]\n"
            "[dim]AI-Powered Contract Analysis[/dim]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*50)
        print("⚖️ Legal Document Analyzer")
        print("AI-Powered Contract Analysis")
        print("="*50 + "\n")


def print_state_machine():
    """Print the state machine diagram."""
    if RICH_AVAILABLE:
        console.print("""
[dim]State Machine Pipeline:[/dim]
    📄 UPLOAD → 📝 PARSE → 🏷️ CLASSIFY
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   NDA               SAAS    ...
                    │                   │
                    └─────────┬─────────┘
                              │
                         ⚠️ RISK ASSESS
                              │
                         ✅ COMPLIANCE
                              │
                         ⚖️ COMPARE
                              │
                         📊 REPORT
""")


def run_analyze(args):
    """Analyze a document."""
    from legal_analyzer import LegalDocumentAnalyzer
    
    print_banner()
    
    # Read document
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: File not found: {args.file}")
            sys.exit(1)
        
        with open(filepath, 'r') as f:
            text = f.read()
        
        filename = filepath.name
    else:
        print("Error: Please provide a document file")
        sys.exit(1)
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Analyzing:[/bold] {filename}")
        console.print(f"[dim]Size: {len(text):,} characters[/dim]\n")
    
    analyzer = LegalDocumentAnalyzer()
    
    # Run analysis
    report = analyzer.analyze(text, compare_to_standard=not args.no_compare)
    
    if not report:
        print("Analysis failed.")
        sys.exit(1)
    
    # Display results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            f.write(report.to_markdown())
        print(f"\n✅ Report saved to: {output_path}")
    else:
        if RICH_AVAILABLE:
            console.print(Markdown(report.to_markdown()))
        else:
            print(report.to_markdown())


def run_sample(args):
    """Analyze a sample document."""
    from legal_analyzer import LegalDocumentAnalyzer, DocumentType
    
    print_banner()
    print_state_machine()
    
    type_map = {
        "saas": DocumentType.SAAS,
        "nda": DocumentType.NDA,
        "employment": DocumentType.EMPLOYMENT,
    }
    
    doc_type = type_map.get(args.type.lower())
    if not doc_type:
        print(f"Unknown document type: {args.type}")
        print(f"Available: {list(type_map.keys())}")
        sys.exit(1)
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Analyzing sample:[/bold] {doc_type.value}\n")
    
    analyzer = LegalDocumentAnalyzer()
    report = analyzer.get_sample_analysis(doc_type)
    
    if report:
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                f.write(report.to_markdown())
            print(f"\n✅ Report saved to: {output_path}")
        else:
            if RICH_AVAILABLE:
                console.print(Markdown(report.to_markdown()))
            else:
                print(report.to_markdown())


def run_classify(args):
    """Quick document classification."""
    from legal_analyzer import classify_document_type
    
    print_banner()
    
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
    
    with open(filepath, 'r') as f:
        text = f.read()
    
    doc_type, confidence = classify_document_type(text)
    
    if RICH_AVAILABLE:
        console.print(f"\n[bold]Document Type:[/bold] {doc_type.value}")
        console.print(f"[bold]Confidence:[/bold] {confidence:.0%}")
    else:
        print(f"\nDocument Type: {doc_type.value}")
        print(f"Confidence: {confidence:.0%}")


def run_interactive(args):
    """Interactive mode."""
    from legal_analyzer import LegalDocumentAnalyzer, DocumentType
    
    print_banner()
    print_state_machine()
    
    print("\nCommands:")
    print("  analyze FILE         - Analyze a document")
    print("  sample TYPE          - Analyze sample (saas/nda/employment)")
    print("  classify FILE        - Quick classification")
    print("  quit                 - Exit")
    print()
    
    analyzer = LegalDocumentAnalyzer()
    
    while True:
        try:
            if RICH_AVAILABLE:
                user_input = console.input("[bold blue]⚖️ >>> [/bold blue]")
            else:
                user_input = input("⚖️ >>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break
        
        parts = user_input.strip().split(maxsplit=1)
        if not parts:
            continue
        
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd in ("quit", "exit", "q"):
            print("Goodbye! 👋")
            break
        
        elif cmd == "analyze" and arg:
            try:
                with open(arg, 'r') as f:
                    text = f.read()
                report = analyzer.analyze(text)
                if report and RICH_AVAILABLE:
                    console.print(Markdown(report.to_markdown()))
                elif report:
                    print(report.to_markdown())
            except FileNotFoundError:
                print(f"File not found: {arg}")
        
        elif cmd == "sample" and arg:
            type_map = {"saas": DocumentType.SAAS, "nda": DocumentType.NDA, "employment": DocumentType.EMPLOYMENT}
            doc_type = type_map.get(arg.lower())
            if doc_type:
                report = analyzer.get_sample_analysis(doc_type)
                if report and RICH_AVAILABLE:
                    console.print(Markdown(report.to_markdown()))
                elif report:
                    print(report.to_markdown())
            else:
                print(f"Unknown type. Available: {list(type_map.keys())}")
        
        elif cmd == "classify" and arg:
            try:
                from legal_analyzer import classify_document_type
                with open(arg, 'r') as f:
                    text = f.read()
                doc_type, confidence = classify_document_type(text)
                print(f"Type: {doc_type.value} ({confidence:.0%})")
            except FileNotFoundError:
                print(f"File not found: {arg}")
        
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
        description="Legal Document Analyzer - AI-Powered Contract Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a document
    python main.py analyze contract.txt
    
    # Analyze with output file
    python main.py analyze contract.txt -o report.md
    
    # Analyze sample document
    python main.py sample saas
    python main.py sample nda
    
    # Quick classification
    python main.py classify document.txt
    
    # Interactive mode
    python main.py interactive
    
    # Web server
    python main.py serve

Document Types:
    - NDA (Non-Disclosure Agreement)
    - SaaS Agreement
    - Employment Contract
    - Lease Agreement
    - Purchase Agreement
    - Terms of Service
    - Privacy Policy
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a document")
    analyze_parser.add_argument("file", help="Document file to analyze")
    analyze_parser.add_argument("--output", "-o", help="Output file for report")
    analyze_parser.add_argument("--no-compare", action="store_true", help="Skip comparison to standard")
    
    # Sample command
    sample_parser = subparsers.add_parser("sample", help="Analyze a sample document")
    sample_parser.add_argument("type", help="Document type (saas, nda, employment)")
    sample_parser.add_argument("--output", "-o", help="Output file for report")
    
    # Classify command
    classify_parser = subparsers.add_parser("classify", help="Quick document classification")
    classify_parser.add_argument("file", help="Document file to classify")
    
    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "sample":
        run_sample(args)
    elif args.command == "classify":
        run_classify(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
