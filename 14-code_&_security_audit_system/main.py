#!/usr/bin/env python3
"""
🔒 Code Review & Security Audit System - Command Line Interface

Automated code review with security vulnerability detection.

Usage:
    python main.py review ./src
    python main.py review file.py
    python main.py serve
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich.syntax import Syntax
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
   ██████╗ ██████╗ ██████╗ ███████╗    ██████╗ ███████╗██╗   ██╗██╗███████╗██╗    ██╗
  ██╔════╝██╔═══██╗██╔══██╗██╔════╝    ██╔══██╗██╔════╝██║   ██║██║██╔════╝██║    ██║
  ██║     ██║   ██║██║  ██║█████╗      ██████╔╝█████╗  ██║   ██║██║█████╗  ██║ █╗ ██║
  ██║     ██║   ██║██║  ██║██╔══╝      ██╔══██╗██╔══╝  ╚██╗ ██╔╝██║██╔══╝  ██║███╗██║
  ╚██████╗╚██████╔╝██████╔╝███████╗    ██║  ██║███████╗ ╚████╔╝ ██║███████╗╚███╔███╔╝
   ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝    ╚═╝  ╚═╝╚══════╝  ╚═══╝  ╚═╝╚══════╝ ╚══╝╚══╝
[/bold blue]
[bold cyan]              & Security Audit System[/bold cyan]
        """
        console.print(banner)
        console.print(Panel.fit(
            "[dim]AutoGen Pipeline Architecture | OWASP Top 10 Detection[/dim]\n"
            "[green]Parallel Analysis → Aggregation → Prioritization → Fix Suggestion → Report[/green]",
            border_style="blue"
        ))
    else:
        print("\n" + "="*60)
        print("       🔒 CODE REVIEW & SECURITY AUDIT SYSTEM")
        print("    AutoGen Pipeline Architecture")
        print("="*60 + "\n")


def print_pipeline():
    """Print the pipeline architecture."""
    if not RICH_AVAILABLE:
        return
    
    pipeline = """
┌─────────────┐
│   CODE      │
│   INPUT     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                    PARALLEL ANALYSIS                        │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐│
│  │  SYNTAX   │  │ SECURITY  │  │   STYLE   │  │   PERF    ││
│  │  CHECKER  │  │  SCANNER  │  │  CHECKER  │  │  ANALYZER ││
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘│
└────────┴──────────────┴──────────────┴──────────────┴───────┘
                               │
                    ┌──────────▼──────────┐
                    │     AGGREGATOR      │
                    └──────────┬──────────┘
                    ┌──────────▼──────────┐
                    │   PRIORITIZER       │
                    └──────────┬──────────┘
                    ┌──────────▼──────────┐
                    │   FIX SUGGESTER     │
                    └──────────┬──────────┘
                    ┌──────────▼──────────┐
                    │   REPORT GENERATOR  │
                    └─────────────────────┘
    """
    console.print(Panel(pipeline, title="Pipeline Architecture", border_style="blue"))


def print_results(report):
    """Print review results."""
    if not RICH_AVAILABLE:
        print(report.to_markdown())
        return
    
    analysis = report.analysis
    
    # Summary table
    table = Table(title="📊 Review Summary", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Files Analyzed", str(analysis.files_analyzed))
    table.add_row("Total Lines", f"{analysis.total_lines:,}")
    table.add_row("Total Issues", str(analysis.issue_count))
    table.add_row("Quality Score", f"{analysis.quality_metrics.overall_score:.1f}/10")
    
    console.print(table)
    
    # Severity breakdown
    severity_table = Table(title="🎯 Issues by Severity", box=box.ROUNDED)
    severity_table.add_column("Severity", style="bold")
    severity_table.add_column("Count", justify="right")
    severity_table.add_column("Bar")
    
    max_count = max(analysis.critical_count, analysis.high_count, analysis.medium_count, analysis.low_count, 1)
    
    severity_table.add_row(
        "🔴 Critical", str(analysis.critical_count),
        "█" * int(analysis.critical_count / max_count * 20),
        style="red"
    )
    severity_table.add_row(
        "🟠 High", str(analysis.high_count),
        "█" * int(analysis.high_count / max_count * 20),
        style="yellow"
    )
    severity_table.add_row(
        "🟡 Medium", str(analysis.medium_count),
        "█" * int(analysis.medium_count / max_count * 20),
        style="bright_yellow"
    )
    severity_table.add_row(
        "🔵 Low", str(analysis.low_count),
        "█" * int(analysis.low_count / max_count * 20),
        style="blue"
    )
    
    console.print(severity_table)
    
    # Critical/High issues
    critical_high = [i for i in analysis.issues if i.severity.value in ['critical', 'high']]
    
    if critical_high:
        console.print(f"\n[bold red]⚠️ Critical & High Severity Issues ({len(critical_high)})[/bold red]\n")
        
        for issue in critical_high[:10]:
            severity_style = "red" if issue.severity.value == "critical" else "yellow"
            
            console.print(Panel(
                f"[bold]{issue.title}[/bold]\n\n"
                f"{issue.description}\n\n"
                f"📍 [cyan]{issue.location}[/cyan]\n"
                f"🏷️ Category: {issue.category.value}\n"
                f"🔧 Fix: {issue.fix_suggestion}",
                title=f"{issue.severity_emoji} {issue.severity.value.upper()}",
                border_style=severity_style,
            ))
    
    # Security issues
    security = analysis.security_issues
    if security:
        console.print(f"\n[bold]🔒 Security Vulnerabilities ({len(security)})[/bold]\n")
        
        sec_table = Table(box=box.SIMPLE)
        sec_table.add_column("CWE")
        sec_table.add_column("Issue")
        sec_table.add_column("Location")
        sec_table.add_column("Severity")
        
        for issue in security[:10]:
            sec_table.add_row(
                issue.cwe_id or "N/A",
                issue.title[:40],
                str(issue.location)[:30],
                issue.severity.value,
            )
        
        console.print(sec_table)
    
    # Recommendations
    if report.recommendations:
        console.print("\n[bold]💡 Recommendations[/bold]\n")
        for i, rec in enumerate(report.recommendations, 1):
            console.print(f"  {i}. {rec}")
    
    # Executive Summary
    console.print("\n")
    console.print(Panel(
        report.executive_summary,
        title="Executive Summary",
        border_style="green" if analysis.critical_count == 0 else "red",
    ))


def run_review(args):
    """Run code review."""
    from code_review import CodeReviewEngine, ReviewConfig
    
    print_banner()
    
    if args.pipeline:
        print_pipeline()
    
    config = ReviewConfig(
        verbose=not args.quiet,
        check_security=not args.no_security,
        check_style=not args.no_style,
        check_performance=not args.no_performance,
        output_dir=args.output or "output",
    )
    
    engine = CodeReviewEngine(config)
    
    if RICH_AVAILABLE and not args.quiet:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running code review...", total=None)
            report = engine.review(args.path)
            progress.remove_task(task)
    else:
        report = engine.review(args.path)
    
    # Display results
    print_results(report)
    
    # Save report
    if args.save:
        filepath = engine.save_report(report, args.format)
        if RICH_AVAILABLE:
            console.print(f"\n📄 Report saved to: [cyan]{filepath}[/cyan]")
    
    # Exit code based on critical issues
    if args.fail_on_critical and report.analysis.critical_count > 0:
        sys.exit(1)


def run_scan_code(args):
    """Scan code from stdin or argument."""
    from code_review import review_code
    
    print_banner()
    
    if args.code:
        code = args.code
    else:
        code = sys.stdin.read()
    
    report = review_code(code, args.language)
    print_results(report)


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


def run_security_checks(args):
    """List available security checks."""
    print_banner()
    
    from code_review.analyzers.security import (
        SQL_INJECTION_PATTERNS, COMMAND_INJECTION_PATTERNS,
        XSS_PATTERNS, AUTH_PATTERNS, CRYPTO_PATTERNS,
        DATA_EXPOSURE_PATTERNS, ACCESS_CONTROL_PATTERNS,
        DESERIALIZATION_PATTERNS
    )
    
    categories = {
        "SQL Injection": SQL_INJECTION_PATTERNS,
        "Command Injection": COMMAND_INJECTION_PATTERNS,
        "XSS": XSS_PATTERNS,
        "Authentication/Secrets": AUTH_PATTERNS,
        "Cryptographic Issues": CRYPTO_PATTERNS,
        "Data Exposure": DATA_EXPOSURE_PATTERNS,
        "Access Control": ACCESS_CONTROL_PATTERNS,
        "Deserialization": DESERIALIZATION_PATTERNS,
    }
    
    if RICH_AVAILABLE:
        for cat_name, patterns in categories.items():
            console.print(f"\n[bold cyan]{cat_name}[/bold cyan]")
            for pattern in patterns:
                console.print(f"  • {pattern.name} ({pattern.cwe_id}) - {pattern.severity.value}")
    else:
        for cat_name, patterns in categories.items():
            print(f"\n{cat_name}")
            for pattern in patterns:
                print(f"  • {pattern.name} ({pattern.cwe_id}) - {pattern.severity.value}")


def main():
    parser = argparse.ArgumentParser(
        description="🔒 Code Review & Security Audit System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Review a file or directory
    python main.py review ./src
    python main.py review auth_service.py
    
    # Review with specific options
    python main.py review ./src --no-style --save
    
    # Review and fail CI if critical issues
    python main.py review ./src --fail-on-critical
    
    # Scan code directly
    echo "password = 'secret123'" | python main.py scan --language python
    
    # List available security checks
    python main.py checks
    
    # Start web server
    python main.py serve
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Review command
    review_parser = subparsers.add_parser("review", help="Review code files or directory")
    review_parser.add_argument("path", help="File or directory to review")
    review_parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output")
    review_parser.add_argument("--save", "-s", action="store_true", help="Save report to file")
    review_parser.add_argument("--format", "-f", choices=["markdown", "json"], default="markdown")
    review_parser.add_argument("--output", "-o", help="Output directory")
    review_parser.add_argument("--pipeline", "-p", action="store_true", help="Show pipeline diagram")
    review_parser.add_argument("--no-security", action="store_true", help="Skip security checks")
    review_parser.add_argument("--no-style", action="store_true", help="Skip style checks")
    review_parser.add_argument("--no-performance", action="store_true", help="Skip performance checks")
    review_parser.add_argument("--fail-on-critical", action="store_true", help="Exit with code 1 if critical issues found")
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan code from stdin or argument")
    scan_parser.add_argument("--code", "-c", help="Code to scan")
    scan_parser.add_argument("--language", "-l", default="python", choices=["python", "javascript", "typescript"])
    
    # Checks command
    checks_parser = subparsers.add_parser("checks", help="List available security checks")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    
    args = parser.parse_args()
    
    if args.command == "review":
        run_review(args)
    elif args.command == "scan":
        run_scan_code(args)
    elif args.command == "checks":
        run_security_checks(args)
    elif args.command == "serve":
        run_serve(args)
    else:
        print_banner()
        if RICH_AVAILABLE:
            print_pipeline()
        parser.print_help()


if __name__ == "__main__":
    main()
